"""Parameter sweep baseline - single adversary only.

Tries every (dx, dy, dv) combination from a grid by repositioning a background
vehicle after reset and letting it drive freely with IDM/MOBIL behaviour. The
ego is controlled by its policy as normal. The worst outcome per rollout is kept.

Not comparable to QD methods on diversity metrics - useful as a naive worst-case
baseline only.
"""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from baas.core.ego_policy import EgoPolicy
from baas.core.env_adapter import EnvAdapter
from baas.core.metrics import EpisodeMetrics, IncidentThresholds
from baas.core.rollout import run_episode
from baas.core.types import EpisodeResult, RolloutSpec
from baas.evaluation.runner import estimate_feasibility, save_results
from baas.methods.parameter_sweep.config import ParameterSweepConfig

import baas

logger = logging.getLogger(__name__)


def _worst_score(m: EpisodeMetrics) -> Tuple[float, float, float]:
    """Lower score = worse for the ego. Used to select the hardest grid point."""
    tcrit = m.time_to_critical_incident_steps
    if tcrit is None or not np.isfinite(tcrit):
        tcrit = float(m.steps)

    mind = m.min_dist_ego_any if (m.min_dist_ego_any is not None and np.isfinite(m.min_dist_ego_any)) else 1e9
    minttc = m.min_ttc_ego_any if (m.min_ttc_ego_any is not None and np.isfinite(m.min_ttc_ego_any)) else 1e9

    return (float(tcrit), float(mind), float(minttc))


def run_parameter_sweep(
    specs: List[RolloutSpec],
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    thresholds: IncidentThresholds,
    cfg: ParameterSweepConfig,
    *,
    env_cfg: Any = None,
    output_dir: Path,
    config: dict,
    config_sha1: str,
    n_feasibility_reruns: int = 0,
) -> List[EpisodeResult]:
    """Run the parameter sweep and write results to output_dir.

    specs must have n_adversaries=0 (only ego is a controlled vehicle).
    The background vehicle at road.vehicles[1] is repositioned for each grid point.
    Returns the list of worst-case EpisodeResults, one per rollout spec.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(cfg.delta_x_grid, cfg.delta_y_grid, cfg.delta_v_grid))
    logger.info("Grid: %d combinations × %d rollout specs", len(grid), len(specs))

    best_results: List[EpisodeResult] = []
    best_params: List[dict] = []

    for spec in specs:
        best_score: Optional[Tuple[float, float, float]] = None
        best_result: Optional[EpisodeResult] = None
        best_dx, best_dy, best_dv = 0.0, 0.0, 0.0

        for dx, dy, dv in grid:
            def _post_reset(unwrapped: Any, _dx=dx, _dy=dy, _dv=dv) -> None:
                adapter.apply_background_perturbation(unwrapped, 1, _dx, _dy, _dv)

            try:
                result = run_episode(
                    spec=spec,
                    adapter=adapter,
                    ego_policy=ego_policy,
                    adv_controllers=[],
                    thresholds=thresholds,
                    env_cfg=env_cfg,
                    stop_on_critical=False,
                    post_reset_fn=_post_reset,
                )
            except Exception as exc:
                logger.debug(
                    "Rollout %d dx=%.1f dy=%.1f dv=%.1f failed: %s",
                    spec.rollout_index, dx, dy, dv, exc,
                )
                continue

            score = _worst_score(result.metrics)
            if best_score is None or score < best_score:
                best_score = score
                best_result = result
                best_dx, best_dy, best_dv = dx, dy, dv

        if best_result is None:
            logger.warning("All grid points failed for rollout %d, skipping.", spec.rollout_index)
            continue

        if n_feasibility_reruns > 0:
            def _best_post_reset(unwrapped: Any, _dx=best_dx, _dy=best_dy, _dv=best_dv) -> None:
                adapter.apply_background_perturbation(unwrapped, 1, _dx, _dy, _dv)

            feasibility, label = estimate_feasibility(
                spec, adapter, ego_policy, thresholds,
                make_controllers=lambda s: [],
                env_cfg=env_cfg,
                n_reruns=n_feasibility_reruns,
                post_reset_fn=_best_post_reset,
            )
            best_result.metrics.feasibility = feasibility
            best_result.metrics.difficulty_label = label
            logger.debug(
                "Rollout %d feasibility=%.2f  label=%s",
                spec.rollout_index, feasibility, label,
            )

        best_results.append(best_result)
        best_params.append({
            "rollout_index": spec.rollout_index,
            "dx": best_dx,
            "dy": best_dy,
            "dv": best_dv,
            "critical": best_result.metrics.critical_incident,
        })

        logger.info(
            "Rollout %d/%d  best=(dx=%.1f dy=%.1f dv=%.1f)  critical=%s",
            spec.rollout_index + 1, len(specs),
            best_dx, best_dy, best_dv,
            best_result.metrics.critical_incident,
        )

    save_results(
        best_results,
        output_dir / "results.json",
        method="parameter_sweep",
        config=config,
        config_sha1=config_sha1,
        specs=specs,
    )

    artefact = {
        "baas_version": baas.__version__,
        "method": "parameter_sweep",
        "grid": {
            "delta_x_grid": cfg.delta_x_grid,
            "delta_y_grid": cfg.delta_y_grid,
            "delta_v_grid": cfg.delta_v_grid,
        },
        "best_params": best_params,
    }
    (output_dir / "parameter_sweep_artefact.json").write_text(
        json.dumps(artefact, indent=2), encoding="utf-8"
    )

    p_critical = sum(1 for r in best_results if r.metrics.critical_incident) / max(1, len(best_results))
    logger.info(
        "Done. p_critical=%.3f  (%d/%d rollouts)",
        p_critical, int(p_critical * len(best_results)), len(best_results),
    )

    return best_results
