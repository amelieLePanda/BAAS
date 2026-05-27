"""Watch a trained ego policy drive live in the highway-env window.

Runs N episodes with human rendering so you can visually assess behaviour.
No adversaries — baseline driving quality only.

Usage:
    python baas/scripts/run_watch_ego.py \\
        --config configs/benchmark_v1.yaml \\
        --ego-policy checkpoints/ego_dqn/checkpoints/ego_dqn_final.zip \\
        --episodes 5

    # PPO ego:
    python baas/scripts/run_watch_ego.py \\
        --config configs/benchmark_v1.yaml \\
        --ego-policy checkpoints/ego_ppo/checkpoints/ego_ppo_final.zip \\
        --policy-type ppo
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch ego policy drive live")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ego-policy", required=True, type=Path)
    parser.add_argument("--policy-type", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--background-traffic", type=int, default=None,
                        help="Override background vehicle count (default: from config)")
    args = parser.parse_args()

    import gymnasium as gym
    import highway_env  # noqa: F401

    from baas.evaluation.config_loader import load_config
    from baas.training.config import GRAYSCALE_OBS_CFG

    bundle = load_config(args.config)
    ego_cfg = bundle.ego_training
    duration_s = float(ego_cfg.horizon_steps) / float(ego_cfg.policy_frequency)

    config = {
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

    env = gym.make(ego_cfg.env_id, config=config, render_mode="human")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        steps = 0
        collided = False

        while not done:
            # DQN/PPO both accept a batch dim
            action, _ = model.predict(obs[None, ...] if obs.ndim == 3 else obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).flat[0]))
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
            if info.get("crashed", False):
                collided = True

        logger.info(
            "Episode %d/%d  steps=%d  return=%.2f  crashed=%s",
            ep + 1, args.episodes, steps, total_reward, collided,
        )

    env.close()


if __name__ == "__main__":
    main()
