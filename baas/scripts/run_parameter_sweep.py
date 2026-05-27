"""CLI entry point for the parameter sweep baseline.

Tries a grid of adversary initial conditions and picks the worst outcome
per rollout. Single adversary only. Outputs results.json for use with
run_benchmark.py summarise.

Usage:
    python baas/scripts/run_parameter_sweep.py \\
        --config configs/benchmark_v1.yaml \\
        --ego-policy checkpoints/ego_dqn.zip \\
        --output runs/parameter_sweep/run_000
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter sweep adversary baseline")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ego-policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from baas.adapters.highway_env.adapter import HighwayEnvAdapter
    from baas.core.ego_policy import DQNEgoPolicy
    from baas.evaluation.benchmark import make_rollout_specs_from_config
    from baas.evaluation.config_loader import load_config
    from baas.methods.parameter_sweep.search import run_parameter_sweep

    bundle = load_config(args.config)
    config_sha1 = hashlib.sha1(args.config.read_bytes()).hexdigest()

    specs = make_rollout_specs_from_config(bundle.raw, n_adversaries=0)
    adapter = HighwayEnvAdapter()
    ego = DQNEgoPolicy(str(args.ego_policy))

    args.output.mkdir(parents=True, exist_ok=True)
    run_parameter_sweep(
        specs=specs,
        adapter=adapter,
        ego_policy=ego,
        thresholds=bundle.thresholds,
        cfg=bundle.parameter_sweep,
        env_cfg=bundle.env,
        output_dir=args.output,
        config=bundle.raw,
        config_sha1=config_sha1,
    )


if __name__ == "__main__":
    main()
