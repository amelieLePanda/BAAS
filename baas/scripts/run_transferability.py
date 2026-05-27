"""CLI entry point for transferability evaluation.

Evaluates the same adversarial artefact against two ego policies and reports
the delta in critical incident rate (black-box minus trained ego).

Usage:
    python baas/scripts/run_transferability.py \\
        --config configs/benchmark_v1.yaml \\
        --method map_elites \\
        --artefact runs/map_elites/run_000/archive.json \\
        --trained-ego checkpoints/ego_dqn.zip \\
        --blackbox-ego checkpoints/ego_ppo.zip \\
        --specs runs/map_elites/run_000/rollout_specs.json \\
        --output runs/transferability/map_elites.json

    python baas/scripts/run_transferability.py \\
        --config configs/benchmark_v1.yaml \\
        --method ppo_adversary \\
        --artefact runs/ppo_adversary/run_000/ppo_adv_final.zip \\
        --trained-ego checkpoints/ego_dqn.zip \\
        --blackbox-ego checkpoints/ego_ppo.zip \\
        --specs runs/map_elites/run_000/rollout_specs.json \\
        --output runs/transferability/ppo.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ["map_elites", "ppo_adversary", "qdrl", "king_light", "cma_es", "novelty_search"]


def _make_controllers_map_elites(
    artefact_path: Path,
    cfg: dict,
    n_adversaries: int,
) -> Callable:
    """Load the best elite from an archive.json and return a make_controllers factory."""
    from baas.evaluation.runner import action_seq_controllers
    from baas.methods.map_elites.genome import ActionSeqGenomeSpec

    data = json.loads(artefact_path.read_text(encoding="utf-8"))
    records = data.get("archive", [])
    if not records:
        raise RuntimeError(f"No elites found in archive: {artefact_path}")

    best = max(records, key=lambda r: float(r.get("objective", float("-inf"))))

    sol_keys = sorted(k for k in best if k.startswith("solution_"))
    if not sol_keys:
        raise RuntimeError("Archive record has no solution_N columns.")
    solution = np.array([float(best[k]) for k in sol_keys], dtype=np.float32)

    me_raw = cfg.get("map_elites", {})
    genome_spec = ActionSeqGenomeSpec(
        horizon_steps=cfg["env"]["horizon_steps"],
        n_adversaries=n_adversaries,
        n_blocks=me_raw.get("n_blocks", 20),
        block_size=me_raw.get("block_size", 3),
    )
    genome = genome_spec.from_continuous(solution)
    seqs = genome_spec.decode(genome)
    horizon = cfg["env"]["horizon_steps"]

    logger.info(
        "MAP-Elites best elite: objective=%.3f  solution=%s",
        float(best.get("objective", float("nan"))), solution.tolist(),
    )

    def make_controllers(spec: Any) -> List[Any]:
        return action_seq_controllers(seqs, horizon)

    return make_controllers


def _make_controllers_ppo(artefact_path: Path, device: str = "cpu") -> Callable:
    """Load a PPO .zip and return a make_controllers factory."""
    from stable_baselines3 import PPO

    ppo = PPO.load(str(artefact_path), device=device)

    def _ctrl(obs: Any) -> int:
        act, _ = ppo.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        return int(np.asarray(act).reshape(-1)[0])

    def make_controllers(spec: Any) -> List[Any]:
        return [_ctrl]

    return make_controllers


def _make_controllers_king_light(artefact_path: Path, n_adversaries: int) -> Callable:
    """Load a KING-light result JSON and return a make_controllers factory."""
    from baas.evaluation.runner import action_seq_controllers

    data = json.loads(artefact_path.read_text(encoding="utf-8"))
    adv_actions_idx: List[List[int]] = data.get("adv_actions_idx", [])
    if not adv_actions_idx:
        raise RuntimeError(f"No adv_actions_idx in: {artefact_path}")

    if isinstance(adv_actions_idx[0], int):
        adv_actions_idx = [adv_actions_idx]

    horizon = len(adv_actions_idx[0])

    def make_controllers(spec: Any) -> List[Any]:
        return action_seq_controllers(adv_actions_idx, horizon)

    return make_controllers


def main() -> None:
    parser = argparse.ArgumentParser(description="Transferability evaluation")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--method", required=True, choices=SUPPORTED_METHODS,
        help="Adversary method whose artefact we are evaluating",
    )
    parser.add_argument("--artefact", required=True, type=Path, help="Archive or model path")
    parser.add_argument("--trained-ego", required=True, type=Path)
    parser.add_argument("--blackbox-ego", required=True, type=Path)
    parser.add_argument("--specs", required=True, type=Path, help="rollout_specs.json")
    parser.add_argument("--n-adversaries", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import yaml
    from baas.adapters.highway_env.adapter import HighwayEnvAdapter
    from baas.adapters.highway_env.config import HighwayEnvBenchmarkConfig
    from baas.core.ego_policy import DQNEgoPolicy
    from baas.core.metrics import IncidentThresholds
    from baas.evaluation.benchmark import load_rollout_specs
    from baas.evaluation.transferability import evaluate_transferability

    cfg = yaml.safe_load(args.config.read_text())
    inc = cfg["incident"]
    thresholds = IncidentThresholds(
        critical_dist=inc["critical_dist"],
        nuisance_dist=inc["nuisance_dist"],
        ttc_crit_s=inc["ttc_crit_s"],
        dx_near_m=inc["dx_near_m"],
        dy_near_m=inc["dy_near_m"],
    )
    env_cfg = HighwayEnvBenchmarkConfig(
        env_id=cfg["env"]["env_id"],
        policy_frequency=cfg["env"]["policy_frequency"],
        simulation_frequency=cfg["env"]["simulation_frequency"],
        horizon_steps=cfg["env"]["horizon_steps"],
    )

    specs = load_rollout_specs(args.specs)
    adapter = HighwayEnvAdapter()
    trained_ego = DQNEgoPolicy(str(args.trained_ego))
    blackbox_ego = DQNEgoPolicy(str(args.blackbox_ego))

    if args.method == "map_elites":
        make_controllers = _make_controllers_map_elites(args.artefact, cfg, args.n_adversaries)
    elif args.method in ("ppo_adversary", "qdrl"):
        make_controllers = _make_controllers_ppo(args.artefact, device=args.device)
    elif args.method in ("king_light", "cma_es", "novelty_search"):
        make_controllers = _make_controllers_king_light(args.artefact, args.n_adversaries)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    results = evaluate_transferability(
        specs=specs,
        adapter=adapter,
        trained_ego=trained_ego,
        blackbox_ego=blackbox_ego,
        thresholds=thresholds,
        make_controllers=make_controllers,
        env_cfg=env_cfg,
    )

    logger.info(
        "Transferability:  p_critical trained=%.3f  blackbox=%.3f  delta=%.3f",
        results["p_critical_trained"],
        results["p_critical_blackbox"],
        results["delta_p_critical"],
    )

    payload = {
        "method": args.method,
        "artefact": str(args.artefact),
        "trained_ego": str(args.trained_ego),
        "blackbox_ego": str(args.blackbox_ego),
        "specs": str(args.specs),
        **results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Written to %s", args.output)


if __name__ == "__main__":
    main()
