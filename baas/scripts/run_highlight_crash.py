"""Render a crash episode as a GIF with one specific traffic car highlighted yellow.

Workflow:
  1. Run run_export_gif.py to get the plain crash GIF. Note the seed from the log.

  2. Run --inspect to list every traffic vehicle (including mid-episode spawns):
       python baas/scripts/run_highlight_crash.py \\
           --config configs/benchmark_v1.yaml \\
           --ego-policy checkpoints/ego_dqn_v3/checkpoints/ego_dqn_final.zip \\
           --seed 1 --background-traffic 15 --inspect --output gifs/

  3. Try all candidates (renders one GIF per vehicle):
       python baas/scripts/run_highlight_crash.py \\
           --config configs/benchmark_v1.yaml \\
           --ego-policy checkpoints/ego_dqn_v3/checkpoints/ego_dqn_final.zip \\
           --seed 1 --background-traffic 15 --try-candidates --output gifs/ego_failure_highlighted.gif

  4. Pick the right one, then render the final version:
       python baas/scripts/run_highlight_crash.py \\
           --config configs/benchmark_v1.yaml \\
           --ego-policy checkpoints/ego_dqn_v3/checkpoints/ego_dqn_final.zip \\
           --seed 1 --background-traffic 15 --target-idx 21 --output gifs/ego_failure_highlighted.gif
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ego(env):
    unwrapped = env.unwrapped
    ego = getattr(unwrapped, "vehicle", None)
    controlled = list(getattr(unwrapped, "controlled_vehicles", []) or [])
    if ego is None and controlled:
        ego = controlled[0]
    return ego


def _non_ego_vehicles(env) -> list:
    try:
        ego = _get_ego(env)
        controlled = list(getattr(env.unwrapped, "controlled_vehicles", []) or [])
        all_v = list(getattr(getattr(env.unwrapped, "road", None), "vehicles", []) or [])
        return [v for v in all_v if v is not ego and v not in controlled]
    except Exception:
        return []


def _pos(v) -> Optional[Tuple[float, float]]:
    p = getattr(v, "position", None)
    return (float(p[0]), float(p[1])) if p is not None else None


@dataclass
class VehicleRecord:
    idx: int
    first_step: int          # step when this vehicle first appeared
    first_pos: Tuple[float, float]   # position at first_step
    is_crash_candidate: bool = False


def _make_env_and_model(ego_cfg, background_traffic, policy_type, ego_policy_path):
    import gymnasium as gym
    import highway_env  # noqa: F401
    from baas.training.config import GRAYSCALE_OBS_CFG

    duration_s = float(ego_cfg.horizon_steps) / float(ego_cfg.policy_frequency)
    env_config = {
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
        "vehicles_count": 1 + background_traffic,
    }
    env = gym.make(ego_cfg.env_id, config=env_config, render_mode="rgb_array")

    if policy_type == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(str(ego_policy_path))
    else:
        from stable_baselines3 import PPO
        model = PPO.load(str(ego_policy_path))

    return env, model


# ---------------------------------------------------------------------------
# Dry run: record ALL vehicles (initial + mid-episode spawns)
# ---------------------------------------------------------------------------

def _dry_run_all_vehicles(
    model, env, seed: int
) -> Tuple[int, List[VehicleRecord]]:
    """Run episode without rendering.

    Tracks every non-ego vehicle that ever appears (including mid-episode spawns)
    using object references as keys — no id() to avoid GC/reuse bugs.

    Returns (crash_step, records) where records lists every vehicle with its
    first-appearance step/position and whether it was a crash candidate.
    """
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)

    # vehicle object -> VehicleRecord (populated as vehicles appear)
    seen: Dict[object, VehicleRecord] = {}
    idx_counter = 0
    pre_crashed: set = set()   # vehicle objects already crashed before ego-crash step
    done = False
    steps = 0
    crash_step = -1

    while not done:
        # Register any newly appeared vehicles
        for v in _non_ego_vehicles(env):
            if v not in seen:
                p = _pos(v)
                if p is not None:
                    seen[v] = VehicleRecord(
                        idx=idx_counter,
                        first_step=steps,
                        first_pos=p,
                    )
                    idx_counter += 1
            # Track already-crashed
            if getattr(v, "crashed", False):
                pre_crashed.add(v)

        action, _ = model.predict(obs[None, ...] if obs.ndim == 3 else obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(np.asarray(action).flat[0]))
        steps += 1
        done = terminated or truncated

        if info.get("crashed", False):
            crash_step = steps
            # The crash partner is whichever non-ego vehicle also has crashed=True.
            vehicles_now = _non_ego_vehicles(env)
            for v in vehicles_now:
                if getattr(v, "crashed", False) and v in seen:
                    seen[v].is_crash_candidate = True
            break

    records = sorted(seen.values(), key=lambda r: r.idx)
    return crash_step, records


# ---------------------------------------------------------------------------
# Render pass: highlight one specific vehicle (by first-appearance step+pos)
# ---------------------------------------------------------------------------

def _render_highlighted(
    model, env, seed: int,
    target_first_step: int,
    target_first_pos: Tuple[float, float],
    fps: int, slow_fps: int, freeze_frames: int,
    out: Path,
) -> bool:
    """Render with one specific vehicle highlighted yellow throughout.

    Finds the target vehicle by position-matching at target_first_step, then
    tracks that exact object. All other traffic is forced blue. At crash the
    colors are reset so highway-env's default red shows for crashed vehicles.
    """
    import imageio.v2 as imageio

    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)

    frames = []
    done = False
    steps = 0
    target_vehicle = None

    while not done:
        vehicles = _non_ego_vehicles(env)

        # Find target vehicle by position match at its first-appearance step
        if target_vehicle is None and steps == target_first_step:
            best, best_dist = None, float("inf")
            for v in vehicles:
                p = _pos(v)
                if p is not None:
                    d = ((p[0] - target_first_pos[0]) ** 2 + (p[1] - target_first_pos[1]) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best = v
            if best is not None and best_dist < 2.0:
                target_vehicle = best
                logger.info("Target vehicle locked at step=%d  pos=(%.4f,%.4f)  match_dist=%.4f",
                            steps, target_first_pos[0], target_first_pos[1], best_dist)
            else:
                logger.warning("Could not match target vehicle at step=%d (best_dist=%.2f)",
                               steps, best_dist)

        # Force all traffic blue; highlight target yellow
        for v in vehicles:
            v.color = (100, 200, 255)  # type: ignore[assignment]
        if target_vehicle is not None and any(v is target_vehicle for v in vehicles):
            target_vehicle.color = (255, 255, 0)  # type: ignore[assignment]

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs[None, ...] if obs.ndim == 3 else obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(np.asarray(action).flat[0]))
        steps += 1
        done = terminated or truncated

        if info.get("crashed", False):
            for v in _non_ego_vehicles(env):
                v.color = None  # type: ignore[assignment]  # let crash go red
            crash_frame = env.render()
            if crash_frame is not None:
                frames.extend([crash_frame] * freeze_frames)
            break

    if not frames:
        return False

    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), frames, fps=fps)
    logger.info("Saved %s  (%d frames @ %d fps)", out, len(frames), fps)
    imageio.mimsave(str(out.with_stem(out.stem + "_slow")), frames, fps=slow_fps)
    return True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_inspect(args, ego_cfg, background_traffic) -> None:
    env, model = _make_env_and_model(ego_cfg, background_traffic,
                                     args.policy_type, args.ego_policy)
    crash_step, records = _dry_run_all_vehicles(model, env, args.seed)

    # Also render crash frame for visual reference
    np.random.seed(args.seed)
    obs, _ = env.reset(seed=args.seed)
    done = False
    crash_frame = None
    while not done:
        action, _ = model.predict(obs[None, ...] if obs.ndim == 3 else obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(np.asarray(action).flat[0]))
        done = terminated or truncated
        if info.get("crashed", False):
            crash_frame = env.render()
            break
    env.close()

    print()
    print("=" * 70)
    print(f"  Crash at seed={args.seed}  step={crash_step}"
          f"  total vehicles seen: {len(records)}")
    print("=" * 70)
    print(f"  {'idx':<5}  {'first_step':>10}  {'first_x':>10}  {'first_y':>8}  note")
    print("-" * 70)
    for r in records:
        marker = "  *** CRASH CANDIDATE ***" if r.is_crash_candidate else ""
        print(f"  {r.idx:<5}  {r.first_step:>10}  "
              f"{r.first_pos[0]:>10.4f}  {r.first_pos[1]:>8.4f}{marker}")
    print("=" * 70)
    print()
    candidates = [r for r in records if r.is_crash_candidate]
    if candidates:
        print(f"  Run --try-candidates to render all {len(records)} vehicles,")
        print(f"  or use --target-idx {candidates[0].idx} to render just the candidate.")
    print()

    if crash_frame is not None and args.output:
        import imageio.v2 as imageio
        frame_path = Path(args.output).parent / f"crash_inspect_seed{args.seed}.png"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(str(frame_path), crash_frame)
        print(f"  Crash frame saved to: {frame_path}")
        print()


def cmd_try_candidates(args, ego_cfg, background_traffic) -> None:
    env, model = _make_env_and_model(ego_cfg, background_traffic,
                                     args.policy_type, args.ego_policy)
    crash_step, records = _dry_run_all_vehicles(model, env, args.seed)
    env.close()

    out_base = Path(args.output)
    n = len(records)
    print(f"\nRendering {n} vehicles (crash step={crash_step}):\n")

    for r in records:
        marker = " *** CRASH CANDIDATE ***" if r.is_crash_candidate else ""
        out = out_base.with_stem(f"{out_base.stem}_v{r.idx}")
        print(f"  v{r.idx}  first_step={r.first_step}"
              f"  pos=({r.first_pos[0]:.2f},{r.first_pos[1]:.2f}){marker}  → {out.name}")
        env2, model2 = _make_env_and_model(ego_cfg, background_traffic,
                                            args.policy_type, args.ego_policy)
        _render_highlighted(model2, env2, args.seed,
                            r.first_step, r.first_pos,
                            args.fps, args.slow_fps, args.freeze_frames, out)
        env2.close()

    print(f"\nDone. Open the GIFs and find the one where the yellow car is the crash car.")
    print(f"Then re-run with --target-idx N to get just that one.")


def cmd_target(args, ego_cfg, background_traffic) -> None:
    env, model = _make_env_and_model(ego_cfg, background_traffic,
                                     args.policy_type, args.ego_policy)
    _, records = _dry_run_all_vehicles(model, env, args.seed)
    env.close()

    rec = next((r for r in records if r.idx == args.target_idx), None)
    if rec is None:
        logger.error("No vehicle with idx=%d. Run --inspect to see available indices.",
                     args.target_idx)
        return

    logger.info("Rendering vehicle idx=%d  first_step=%d  pos=(%.4f,%.4f)",
                rec.idx, rec.first_step, rec.first_pos[0], rec.first_pos[1])
    env2, model2 = _make_env_and_model(ego_cfg, background_traffic,
                                        args.policy_type, args.ego_policy)
    _render_highlighted(model2, env2, args.seed,
                        rec.first_step, rec.first_pos,
                        args.fps, args.slow_fps, args.freeze_frames,
                        Path(args.output))
    env2.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a crash episode with one traffic car highlighted yellow"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ego-policy", required=True, type=Path)
    parser.add_argument("--policy-type", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--background-traffic", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--slow-fps", type=int, default=5)
    parser.add_argument("--freeze-frames", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true",
                      help="List all vehicles (initial + mid-episode spawns) with indices")
    mode.add_argument("--try-candidates", action="store_true",
                      help="Render one GIF per vehicle so you can pick the right one")
    mode.add_argument("--target-idx", type=int, metavar="N",
                      help="Render final GIF highlighting vehicle N (idx from --inspect)")

    args = parser.parse_args()

    if (args.try_candidates or args.target_idx is not None) and args.output is None:
        parser.error("--output is required with --try-candidates and --target-idx")

    import highway_env  # noqa: F401
    from baas.evaluation.config_loader import load_config

    bundle = load_config(args.config)
    ego_cfg = bundle.ego_training
    background_traffic = (args.background_traffic if args.background_traffic is not None
                          else ego_cfg.background_traffic)

    if args.inspect:
        cmd_inspect(args, ego_cfg, background_traffic)
    elif args.try_candidates:
        cmd_try_candidates(args, ego_cfg, background_traffic)
    else:
        cmd_target(args, ego_cfg, background_traffic)


if __name__ == "__main__":
    main()
