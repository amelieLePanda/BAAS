"""CLI entry point for PPO adversary training.

Usage:
    python baas/scripts/run_ppo_adversary.py \\
        --config configs/benchmark_v1.yaml \\
        --ego-policy checkpoints/ego_dqn.zip \\
        --output runs/ppo_adversary/run_000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO adversary")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ego-policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from baas.adapters.highway_env.adapter import HighwayEnvAdapter
    from baas.core.ego_policy import DQNEgoPolicy
    from baas.evaluation.config_loader import load_config
    from baas.methods.ppo_adversary.train import train_ppo_adversary

    bundle = load_config(args.config)
    ppo_cfg = bundle.ppo_adversary
    ppo_cfg.seed = args.seed  # type: ignore[misc]

    adapter = HighwayEnvAdapter()
    ego = DQNEgoPolicy(str(args.ego_policy))
    train_ppo_adversary(adapter, ego, ppo_cfg, bundle.env, output_dir=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
