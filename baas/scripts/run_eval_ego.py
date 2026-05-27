"""Systematic ego policy evaluation — baseline driving quality without adversaries.

Runs k episodes from varied seeds and reports the metrics you need to decide
whether the ego is good enough to freeze for adversarial benchmarking, and to
compare DQN vs PPO.

Decision criteria (from MADS research design):
  - p_collision_baseline < 0.10  (ego rarely crashes on its own)
  - mean_speed > 20.0 m/s        (actually drives, not crawls)
  - p_completion > 0.80          (completes most episodes without crashing)
  - feasibility_with_simple_adv in [0.20, 0.70]  (challenging but solvable)

Usage:
    # Evaluate DQN ego:
    python baas/scripts/run_eval_ego.py \\
        --config configs/benchmark_v1.yaml \\
        --ego-policy checkpoints/ego_dqn/checkpoints/ego_dqn_final.zip \\
        --output evals/ego_dqn_eval.json

    # Evaluate PPO ego:
    python baas/scripts/run_eval_ego.py \\
        --config configs/benchmark_v1.yaml \\
        --ego-policy checkpoints/ego_ppo/checkpoints/ego_ppo_final.zip \\
        --policy-type ppo \\
        --output evals/ego_ppo_eval.json

    # Compare both:
    python baas/scripts/run_eval_ego.py --compare evals/ego_dqn_eval.json evals/ego_ppo_eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _eval_ego(
    model: Any,
    env_cfg_dict: Dict[str, Any],
    env_id: str,
    *,
    n_episodes: int,
    seed: int,
) -> Dict[str, Any]:
    """Run n_episodes and return aggregate stats."""
    import gymnasium as gym
    import highway_env  # noqa: F401

    returns, speeds, steps_list = [], [], []
    collisions = 0
    completions = 0

    for ep in range(n_episodes):
        env = gym.make(env_id, config=env_cfg_dict)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_steps = 0
        crashed = False
        speed_sum = 0.0

        while not done:
            action, _ = model.predict(obs[None, ...] if obs.ndim == 3 else obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).flat[0]))
            ep_return += float(reward)
            ep_steps += 1
            speed_sum += float(info.get("speed", 0.0))
            done = terminated or truncated
            if info.get("crashed", False):
                crashed = True

        env.close()
        returns.append(ep_return)
        steps_list.append(ep_steps)
        speeds.append(speed_sum / max(1, ep_steps))
        if crashed:
            collisions += 1
        else:
            completions += 1

    return {
        "n_episodes": n_episodes,
        "p_collision": round(collisions / n_episodes, 4),
        "p_completion": round(completions / n_episodes, 4),
        "mean_return": round(float(np.mean(returns)), 3),
        "std_return": round(float(np.std(returns)), 3),
        "mean_speed_mps": round(float(np.mean(speeds)), 3),
        "mean_episode_steps": round(float(np.mean(steps_list)), 1),
    }


def _passes_criteria(stats: Dict[str, Any]) -> List[str]:
    """Return list of failed criteria (empty = ready to freeze)."""
    failures = []
    if stats["p_collision"] >= 0.10:
        failures.append(f"p_collision={stats['p_collision']:.3f} >= 0.10 (crashes too often)")
    if stats["mean_speed_mps"] < 18.0:
        failures.append(f"mean_speed={stats['mean_speed_mps']:.1f} m/s < 18 (too slow)")
    if stats["p_completion"] < 0.80:
        failures.append(f"p_completion={stats['p_completion']:.3f} < 0.80 (too many crashes)")
    return failures


def _cmd_eval(args: argparse.Namespace) -> None:
    import highway_env  # noqa: F401

    from baas.evaluation.config_loader import load_config
    from baas.training.config import GRAYSCALE_OBS_CFG

    bundle = load_config(args.config)
    ego_cfg = bundle.ego_training
    duration_s = float(ego_cfg.horizon_steps) / float(ego_cfg.policy_frequency)

    env_cfg_dict = {
        "observation": GRAYSCALE_OBS_CFG,
        "action": {"type": "DiscreteMetaAction"},
        "collision_reward": ego_cfg.collision_reward,
        "lane_change_reward": ego_cfg.lane_change_reward,
        "high_speed_reward": ego_cfg.high_speed_reward,
        "right_lane_reward": ego_cfg.right_lane_reward,
        "policy_frequency": ego_cfg.policy_frequency,
        "simulation_frequency": ego_cfg.simulation_frequency,
        "duration": duration_s,
        "controlled_vehicles": 1,
        "vehicles_count": 1 + (args.background_traffic if args.background_traffic is not None else ego_cfg.background_traffic),
    }

    if args.policy_type == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(str(args.ego_policy))
    else:
        from stable_baselines3 import PPO
        model = PPO.load(str(args.ego_policy))

    logger.info("Evaluating %s ego over %d episodes...", args.policy_type.upper(), args.episodes)
    stats = _eval_ego(model, env_cfg_dict, ego_cfg.env_id, n_episodes=args.episodes, seed=args.seed)
    failures = _passes_criteria(stats)

    result = {
        "policy": str(args.ego_policy),
        "policy_type": args.policy_type,
        "stats": stats,
        "ready_to_freeze": len(failures) == 0,
        "failed_criteria": failures,
    }

    print("\n" + "=" * 50)
    print(f"  Ego evaluation: {args.policy_type.upper()}")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k:<28} {v}")
    print("-" * 50)
    if failures:
        print("  NOT READY — failed criteria:")
        for f in failures:
            print(f"    x {f}")
        print("  → Retrain with more steps or adjust hyperparameters.")
    else:
        print("  READY TO FREEZE")
    print("=" * 50 + "\n")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Saved to %s", args.output)


def _cmd_compare(args: argparse.Namespace) -> None:
    """Print side-by-side comparison of two eval JSON files."""
    results = [json.loads(Path(p).read_text()) for p in args.compare]
    keys = list(results[0]["stats"].keys())
    labels = [r["policy_type"].upper() for r in results]

    print("\n" + "=" * 60)
    print(f"  Ego comparison: {' vs '.join(labels)}")
    print("=" * 60)
    print(f"  {'Metric':<28} " + "  ".join(f"{l:>10}" for l in labels))
    print("-" * 60)
    for k in keys:
        vals = [str(r["stats"].get(k, "--")) for r in results]
        print(f"  {k:<28} " + "  ".join(f"{v:>10}" for v in vals))
    print("-" * 60)

    # Recommendation
    scores = []
    for r in results:
        s = r["stats"]
        # Higher is better: completion rate + normalised speed - collision rate
        score = s["p_completion"] + (s["mean_speed_mps"] / 30.0) - s["p_collision"]
        scores.append(score)

    winner_idx = int(np.argmax(scores))
    print(f"\n  Recommendation: use {labels[winner_idx]} ego")
    print(f"  (score: {scores[0]:.3f} vs {scores[1]:.3f})")

    for r in results:
        if not r["ready_to_freeze"]:
            print(f"\n  WARNING: {r['policy_type'].upper()} failed criteria:")
            for f in r["failed_criteria"]:
                print(f"    x {f}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare ego policies")
    sub = parser.add_subparsers(dest="cmd")

    # eval subcommand
    ev = sub.add_parser("eval", help="Evaluate one ego policy")
    ev.add_argument("--config", required=True, type=Path)
    ev.add_argument("--ego-policy", required=True, type=Path)
    ev.add_argument("--policy-type", choices=["dqn", "ppo"], default="dqn")
    ev.add_argument("--episodes", type=int, default=50)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--output", type=Path, default=None)
    ev.add_argument("--background-traffic", type=int, default=None,
                    help="Override background vehicle count (default: from config)")

    # compare subcommand
    cmp = sub.add_parser("compare", help="Compare two eval JSON files")
    cmp.add_argument("compare", nargs=2, type=Path, metavar="EVAL_JSON")

    # Shortcut: no subcommand defaults to eval
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ego-policy", type=Path)
    parser.add_argument("--policy-type", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--background-traffic", type=int, default=None,
                        help="Override background vehicle count (default: from config)")
    parser.add_argument("--compare", nargs=2, type=Path, metavar="EVAL_JSON")

    args = parser.parse_args()

    if args.compare:
        _cmd_compare(args)
    elif args.ego_policy:
        _cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
