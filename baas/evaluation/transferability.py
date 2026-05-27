"""Cross-policy transferability evaluation.

Measures whether adversarial scenarios found against one ego policy also
fail a held-out black-box ego. A high delta indicates the discovered failures
generalise beyond the policy used during search.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from baas.core.ego_policy import EgoPolicy
from baas.core.env_adapter import EnvAdapter
from baas.core.metrics import IncidentThresholds
from baas.core.types import RolloutSpec
from baas.evaluation.runner import AdvController, evaluate_artefact

logger = logging.getLogger(__name__)


def evaluate_transferability(
    specs: List[RolloutSpec],
    adapter: EnvAdapter,
    trained_ego: EgoPolicy,
    blackbox_ego: EgoPolicy,
    thresholds: IncidentThresholds,
    make_controllers: Any,
    *,
    env_cfg: Any = None,
) -> Dict[str, float]:
    """Evaluate the same adversarial artefact against two ego policies.

    Returns p_critical_trained, p_critical_blackbox, delta_p_critical,
    p_collision_trained, p_collision_blackbox.
    """
    trained_results = evaluate_artefact(
        specs, adapter, trained_ego, thresholds, make_controllers, env_cfg=env_cfg
    )
    blackbox_results = evaluate_artefact(
        specs, adapter, blackbox_ego, thresholds, make_controllers, env_cfg=env_cfg
    )

    def _rate(results, key: str) -> float:
        if not results:
            return float("nan")
        return float(sum(getattr(r.metrics, key) for r in results) / len(results))

    p_crit_trained  = _rate(trained_results, "critical_incident")
    p_crit_blackbox = _rate(blackbox_results, "critical_incident")
    p_coll_trained  = _rate(trained_results, "ego_collision")
    p_coll_blackbox = _rate(blackbox_results, "ego_collision")

    return {
        "p_critical_trained":   round(p_crit_trained, 4),
        "p_critical_blackbox":  round(p_crit_blackbox, 4),
        "delta_p_critical":     round(p_crit_blackbox - p_crit_trained, 4),
        "p_collision_trained":  round(p_coll_trained, 4),
        "p_collision_blackbox": round(p_coll_blackbox, 4),
    }
