"""CLI entry point for MAP-Elites adversary search.

Usage:
    python baas/scripts/run_map_elites.py \\
        --config configs/benchmark_v1.yaml \\
        --ego-policy checkpoints/ego_dqn.zip \\
        --n-adversaries 1 \\
        --output runs/map_elites/run_000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAP-Elites adversary search")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ego-policy", required=True, type=Path)
    parser.add_argument("--n-adversaries", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from baas.adapters.highway_env.adapter import HighwayEnvAdapter
    from baas.core.ego_policy import DQNEgoPolicy
    from baas.evaluation.benchmark import make_rollout_specs_from_config, save_rollout_specs
    from baas.evaluation.config_loader import load_config
    from baas.methods.map_elites.genome import ActionSeqGenomeSpec
    from baas.methods.map_elites.search import run_map_elites

    bundle = load_config(args.config)
    raw = bundle.raw

    adapter = HighwayEnvAdapter()
    ego = DQNEgoPolicy(str(args.ego_policy))

    specs = make_rollout_specs_from_config(raw, n_adversaries=args.n_adversaries)
    args.output.mkdir(parents=True, exist_ok=True)
    specs_sha = save_rollout_specs(specs, args.output / "rollout_specs.json")
    logger.info("Rollout specs SHA-1: %s", specs_sha)

    me_raw = raw.get("map_elites", {})
    genome_spec = ActionSeqGenomeSpec(
        horizon_steps=bundle.env.horizon_steps,
        n_adversaries=args.n_adversaries,
        n_blocks=me_raw.get("n_blocks", 20),
        block_size=me_raw.get("block_size", 3),
    )

    result = run_map_elites(
        specs=specs,
        adapter=adapter,
        ego_policy=ego,
        thresholds=bundle.thresholds,
        genome_spec=genome_spec,
        me_cfg=bundle.map_elites,
        env_cfg=bundle.env,
        rng_seed=args.seed,
        output_dir=args.output,
    )

    logger.info("Done. Archive size: %d  QD-score: %.3f", result["archive_size"], result["qd_score"])


if __name__ == "__main__":
    main()
