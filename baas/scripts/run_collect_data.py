"""CLI entry point for ego data collection.

Runs the ego policy through a set of episodes and saves kinematics,
actions, and optionally frames as .npz files. Used to build datasets
for ego training, world-model pre-training, or RLHF labelling.

Usage:
    python baas/scripts/run_collect_data.py \\
        --config configs/benchmark_v1.yaml \\
        --ego-policy checkpoints/ego_dqn.zip \\
        --output runs/data/ego_episodes \\
        --n-episodes 1000

    # Include active adversaries:
    python baas/scripts/run_collect_data.py ... --n-adversaries 1

    # Record RGB frames (large files):
    python baas/scripts/run_collect_data.py ... --record-frames
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ego episode data")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ego-policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-episodes", type=int, default=1000)
    parser.add_argument("--n-adversaries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-frames", action="store_true",
                        help="Save RGB frames per step (increases file size significantly)")
    args = parser.parse_args()

    from baas.adapters.highway_env.adapter import HighwayEnvAdapter
    from baas.core.ego_policy import DQNEgoPolicy
    from baas.core.metrics import IncidentThresholds
    from baas.training.collect_data import collect_episodes
    from baas.training.config import EgoTrainConfig

    cfg = yaml.safe_load(args.config.read_text())

    train_cfg = EgoTrainConfig(
        env_id=cfg["env"]["env_id"],
        policy_frequency=cfg["env"]["policy_frequency"],
        simulation_frequency=cfg["env"]["simulation_frequency"],
        horizon_steps=cfg["env"]["horizon_steps"],
        background_traffic=cfg["env"].get("background_traffic", 5),
    )
    inc = cfg["incident"]
    thresholds = IncidentThresholds(
        critical_dist=inc["critical_dist"],
        nuisance_dist=inc["nuisance_dist"],
        ttc_crit_s=inc["ttc_crit_s"],
        dx_near_m=inc["dx_near_m"],
        dy_near_m=inc["dy_near_m"],
    )

    adapter = HighwayEnvAdapter()
    ego = DQNEgoPolicy(str(args.ego_policy))

    collect_episodes(
        cfg=train_cfg,
        output_dir=args.output,
        adapter=adapter,
        ego_policy=ego,
        thresholds=thresholds,
        n_episodes=args.n_episodes,
        n_adversaries=args.n_adversaries,
        seed=args.seed,
        record_frames=args.record_frames,
    )


if __name__ == "__main__":
    main()
