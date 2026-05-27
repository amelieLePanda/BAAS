"""MAP-Elites search loop using pyribs.

Archive tracks phi = (min_ego_adv_dist, TTCI) as behaviour descriptors.
Genomes are discrete action sequences encoded by ActionSeqGenomeSpec.
Emitter is CMA-ME by default, or Gaussian for ablations.

Multi-adversary: genome shape (n_blocks, n_adv); adversary coordination is
enforced in the fitness function, adv-adv collisions are penalised.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from baas.core.ego_policy import EgoPolicy
from baas.core.env_adapter import EnvAdapter
from baas.core.metrics import IncidentThresholds
from baas.core.types import RolloutSpec
from baas.evaluation.run_io import save_snapshot, update_run_status, write_run_meta
from baas.evaluation.runner import action_seq_controllers, evaluate_artefact
from baas.methods.map_elites.genome import ActionSeqGenomeSpec

logger = logging.getLogger(__name__)


@dataclass
class MapElitesConfig:
    """Hyperparameters for the MAP-Elites search loop."""

    n_iterations: int = 5000
    batch_size: int = 30
    initial_population: int = 100
    sigma0: float = 1.0           # CMA-ME initial step size
    emitter_type: str = "cma_me"  # "cma_me" | "random"

    # phi-space archive grid
    dist_bins: int = 10
    ttci_bins: int = 10
    dist_max: float = 60.0

    nuisance_penalty: float = 0.5
    snapshot_every: int = 500     # save archive snapshot every N iterations (0 = disabled)


def _fitness(metrics: Any, nuisance_penalty: float = 0.5) -> float:
    """Scalar fitness for the archive.

    Matches paper Eq. (6): J(τ) = α·I[CI] + β·I[coll] − δ·I[nuis].
    +1.0 for critical incident (α), +0.5 for ego collision (β).
    Penalised by nuisance_penalty when adversary crashes without ego safety impact (δ).
    """
    f = 0.0
    if metrics.critical_incident:
        f += 1.0
    if metrics.ego_collision:
        f += 0.5
    if metrics.adv_nuisance_crash:
        f -= nuisance_penalty
    return float(f)


def _behaviour_descriptor(metrics: Any, dist_max: float, horizon_steps: int) -> List[float]:
    """Map metrics to normalised 2-D behaviour descriptor phi in [0,1]^2."""
    d = metrics.min_dist_ego_adv
    ttci = float(metrics.time_to_critical_incident_adv_steps)
    d_norm = float(np.clip(d, 0.0, dist_max) / dist_max) if np.isfinite(d) else 1.0
    ttci_norm = float(np.clip(ttci, 1.0, horizon_steps) / horizon_steps)
    return [d_norm, ttci_norm]


def run_map_elites(
    specs: List[RolloutSpec],
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    thresholds: IncidentThresholds,
    genome_spec: ActionSeqGenomeSpec,
    me_cfg: MapElitesConfig,
    *,
    env_cfg: Any = None,
    rng_seed: int = 0,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run MAP-Elites search and return the archive contents.

    Each genome is evaluated against a randomly sampled spec from the list.
    Returns a dict with 'archive', 'archive_size', 'qd_score'.
    """
    try:
        from ribs.archives import GridArchive
        from ribs.emitters import EvolutionStrategyEmitter, GaussianEmitter
        from ribs.schedulers import Scheduler
    except ImportError as e:
        raise ImportError("ribs not installed.  Run: pip install ribs") from e

    if output_dir is not None:
        write_run_meta(
            output_dir,
            method="map_elites",
            config=me_cfg,
            seed=rng_seed,
            n_adversaries=genome_spec.n_adversaries,
        )

    rng = np.random.default_rng(rng_seed)
    horizon = specs[0].horizon_steps

    archive = GridArchive(
        solution_dim=genome_spec.solution_dim,
        dims=[me_cfg.dist_bins, me_cfg.ttci_bins],
        ranges=[(0.0, 1.0), (0.0, 1.0)],
        seed=rng_seed,
    )

    if me_cfg.emitter_type == "cma_me":
        emitters = [
            EvolutionStrategyEmitter(
                archive=archive,
                x0=np.zeros(genome_spec.solution_dim, dtype=np.float32),
                sigma0=me_cfg.sigma0,
                batch_size=me_cfg.batch_size,
                seed=rng_seed + i,
            )
            for i in range(1)
        ]
    else:
        emitters = [
            GaussianEmitter(
                archive=archive,
                x0=np.zeros(genome_spec.solution_dim, dtype=np.float32),
                sigma=me_cfg.sigma0,
                batch_size=me_cfg.batch_size,
                seed=rng_seed,
            )
        ]

    scheduler = Scheduler(archive, emitters)

    for genome in [genome_spec.sample_random(rng) for _ in range(me_cfg.initial_population)]:
        seqs = genome_spec.decode(genome)
        spec = specs[int(rng.integers(0, len(specs)))]
        ctrlrs = action_seq_controllers(seqs, horizon)
        results = evaluate_artefact(
            [spec], adapter, ego_policy, thresholds,
            make_controllers=lambda s, _c=ctrlrs: _c,
            env_cfg=env_cfg,
        )
        if not results:
            continue
        m = results[0].metrics
        archive.add_single(
            solution=genome_spec.to_continuous(genome),
            objective=_fitness(m, me_cfg.nuisance_penalty),
            measures=_behaviour_descriptor(m, me_cfg.dist_max, horizon),
        )

    logger.info("Initial population done.  Archive size: %d", len(archive))

    for iteration in range(me_cfg.n_iterations):
        solutions = scheduler.ask()
        objectives, measures = [], []

        for sol in solutions:
            genome = genome_spec.from_continuous(sol)
            seqs = genome_spec.decode(genome)
            spec = specs[int(rng.integers(0, len(specs)))]
            ctrlrs = action_seq_controllers(seqs, horizon)
            results = evaluate_artefact(
                [spec], adapter, ego_policy, thresholds,
                make_controllers=lambda s, _c=ctrlrs: _c,
                env_cfg=env_cfg,
            )
            if results:
                m = results[0].metrics
                objectives.append(_fitness(m, me_cfg.nuisance_penalty))
                measures.append(_behaviour_descriptor(m, me_cfg.dist_max, horizon))
            else:
                objectives.append(-1.0)
                measures.append([1.0, 1.0])

        scheduler.tell(objectives, measures)

        if (iteration + 1) % 100 == 0:
            logger.info(
                "Iteration %d/%d  archive: %d  QD-score: %.3f",
                iteration + 1, me_cfg.n_iterations, len(archive), archive.stats.qd_score,
            )

        snap_every = int(me_cfg.snapshot_every)
        if output_dir is not None and snap_every > 0 and (iteration + 1) % snap_every == 0:
            elites_snap = archive.data(return_type="pandas")
            snap_records = elites_snap.to_dict(orient="records") if elites_snap is not None else []
            save_snapshot(
                output_dir,
                {"iteration": iteration + 1, "archive_size": len(archive),
                 "qd_score": archive.stats.qd_score, "archive": snap_records},
                name=f"archive_it{iteration + 1:05d}",
            )

    elites = archive.data(return_type="pandas")
    archive_records = elites.to_dict(orient="records") if elites is not None else []

    final = {
        "method": "map_elites",
        "archive": archive_records,
        "archive_size": len(archive),
        "qd_score": archive.stats.qd_score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "archive.json").write_text(
            json.dumps(final, indent=2, default=str), encoding="utf-8",
        )
        update_run_status(output_dir, status="complete",
                          extra={"archive_size": len(archive), "qd_score": archive.stats.qd_score})
        logger.info("Archive saved to %s/archive.json", output_dir)

    return {
        "archive": archive_records,
        "archive_size": len(archive),
        "qd_score": archive.stats.qd_score,
    }
