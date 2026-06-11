"""CLI entry point for QD-RL adversary training.

QD-RL maintains a MAP-Elites archive during PPO training. Each archive cell
holds a distinct adversary policy trained with a different set of reward
shaping parameters. The solution space is [r_critical, r_adv_close, r_adv_nuis].

Usage:
    python baas/scripts/run_qdrl.py \\
        --config configs/benchmark_v1.yaml \\
        --ego-policy checkpoints/ego_dqn.zip \\
        --n-adversaries 1 \\
        --output runs/qdrl/run_000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="QD-RL adversary training")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ego-policy", required=True, type=Path)
    parser.add_argument("--n-adversaries", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None,
                        help="Override cfg.device (e.g. 'cuda', 'cpu'). Default: use yaml value.")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Parallel workers for burst training. Default: min(n_candidates, cpu_count).")
    args = parser.parse_args()

    from baas.adapters.highway_env.adapter import HighwayEnvAdapter
    from baas.core.ego_policy import DQNEgoPolicy
    from baas.evaluation.benchmark import make_rollout_specs_from_config
    from baas.evaluation.config_loader import load_config
    from baas.methods.qdrl.search import run_qdrl
    import dataclasses

    bundle = load_config(args.config)
    qdrl_cfg = bundle.qdrl
    qdrl_cfg.n_adversaries = args.n_adversaries  # type: ignore[misc]
    qdrl_cfg.seed = args.seed  # type: ignore[misc]
    if args.device is not None:
        qdrl_cfg = dataclasses.replace(qdrl_cfg, device=args.device)

    specs = make_rollout_specs_from_config(bundle.raw, n_adversaries=args.n_adversaries)
    adapter = HighwayEnvAdapter()
    ego = DQNEgoPolicy(str(args.ego_policy))

    dist_max = bundle.raw.get("diversity", {}).get("dist_max", 60.0)

    args.output.mkdir(parents=True, exist_ok=True)
    run_qdrl(
        specs, adapter, ego, bundle.thresholds, qdrl_cfg,
        output_dir=args.output, seed=args.seed, env_cfg=bundle.env,
        dist_max=dist_max,
        ego_policy_path=str(args.ego_policy),
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
