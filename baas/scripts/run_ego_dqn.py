"""CLI entry point for ego DQN training.

All environment settings come from the benchmark YAML to guarantee the trained
ego uses the exact same frequencies and horizon as every adversarial method.

Usage:
    python baas/scripts/run_ego_dqn.py \\
        --config configs/benchmark_v1.yaml \\
        --output checkpoints/ego_dqn

    # Optional overrides (rarely needed):
    python baas/scripts/run_ego_dqn.py \\
        --config configs/benchmark_v1.yaml \\
        --output checkpoints/ego_dqn \\
        --total-steps 500000 \\
        --seed 1
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ego DQN policy")
    parser.add_argument("--config", required=True, type=Path,
                        help="Benchmark YAML - single source of truth for env settings")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--background-traffic", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import dataclasses

    from baas.evaluation.config_loader import load_config
    from baas.training.train_ego_dqn import train_ego_dqn

    bundle = load_config(args.config)

    overrides = {"total_timesteps": args.total_steps, "seed": args.seed}
    if args.background_traffic is not None:
        overrides["background_traffic"] = args.background_traffic
    if args.device is not None:
        overrides["device"] = args.device
    cfg = dataclasses.replace(bundle.ego_training, **overrides)
    checkpoint = train_ego_dqn(cfg, args.output, seed=args.seed)
    print(f"Saved: {checkpoint}")


if __name__ == "__main__":
    main()
