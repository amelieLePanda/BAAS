"""CLI entry point for KING-light adversary optimisation.

Single adversary only. Not comparable to QD methods on diversity metrics.

Workflow:
  1. Optimise an adversary action sequence on one rollout spec via the
     BicycleProxy gradient search.
  2. Evaluate the best found sequence across all k rollout specs.
  3. Save results in the standard MADS format (same as every other method).

Usage:
    python baas/scripts/run_king_light.py \\
        --config configs/benchmark_v1.yaml \\
        --ego-policy checkpoints/ego_dqn.zip \\
        --output runs/king_light/run_000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KING-light adversary optimisation")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ego-policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0,
                        help="Index into rollout_specs used for optimisation")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    from baas.adapters.highway_env.adapter import HighwayEnvAdapter
    from baas.core.ego_policy import DQNEgoPolicy
    from baas.evaluation.benchmark import make_rollout_specs_from_config, save_rollout_specs
    from baas.evaluation.config_loader import load_config
    from baas.evaluation.run_io import sha1_file, update_run_status, write_run_meta
    from baas.evaluation.runner import action_seq_controllers, evaluate_artefact, save_results
    from baas.methods.king_light.optimiser import run_king_light

    bundle = load_config(args.config)
    kl_cfg = bundle.king_light
    if args.device is not None:
        import dataclasses
        kl_cfg = dataclasses.replace(kl_cfg, device=args.device)

    specs = make_rollout_specs_from_config(bundle.raw, n_adversaries=1)
    args.output.mkdir(parents=True, exist_ok=True)
    save_rollout_specs(specs, args.output / "rollout_specs.json")

    write_run_meta(
        args.output,
        method="king_light",
        config=kl_cfg,
        seed=args.seed,
        n_adversaries=1,
    )

    adapter = HighwayEnvAdapter()
    ego = DQNEgoPolicy(str(args.ego_policy))

    # Optimise on one spec chosen by --seed
    opt_spec = specs[args.seed % len(specs)]
    logger.info("Optimising on rollout_index=%d  env_seed=%d", opt_spec.rollout_index, opt_spec.env_seed)
    opt_result = run_king_light(opt_spec, adapter, ego, bundle.thresholds, kl_cfg, env_cfg=bundle.env)

    # adv_actions_idx is (T, N_adv) from optimiser — wrap as per-adversary flat list
    raw_seqs = opt_result["adv_actions_idx"]  # list of T lists, each [action_for_adv_0]
    adv_seqs = [[step[0] for step in raw_seqs]]  # [[a0, a1, ..., aT]] for adversary 0

    horizon = specs[0].horizon_steps

    def _make_controllers(spec):  # noqa: ARG001
        return action_seq_controllers(adv_seqs, horizon)

    # Evaluate the found sequence across all rollout specs
    logger.info("Evaluating found sequence on %d specs...", len(specs))
    results = evaluate_artefact(
        specs,
        adapter,
        ego,
        bundle.thresholds,
        _make_controllers,
        env_cfg=bundle.env,
        n_feasibility_reruns=bundle.raw.get("feasibility", {}).get("n_feasibility_reruns", 0),
    )

    config_sha1 = sha1_file(args.config)
    save_results(
        results,
        args.output / "results.json",
        method="king_light",
        config=bundle.raw,
        config_sha1=config_sha1,
        specs=specs,
    )

    p_crit = sum(1 for r in results if r.metrics.critical_incident) / max(1, len(results))
    update_run_status(
        args.output,
        status="complete",
        extra={
            "n_rollouts": len(results),
            "p_critical": round(p_crit, 4),
            "opt_collided": opt_result["metrics"]["collided"],
            "opt_min_dist": opt_result["metrics"]["min_dist"],
        },
    )
    logger.info(
        "Done. p_critical=%.3f  opt_collided=%s  opt_min_dist=%.3f  output=%s",
        p_crit,
        opt_result["metrics"]["collided"],
        opt_result["metrics"]["min_dist"],
        args.output,
    )


if __name__ == "__main__":
    main()
