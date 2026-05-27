"""Collect adversarial scenarios into a reusable dataset.

Wraps evaluate_artefact and packages results as ScenarioRecords.
The dataset is used for ego retraining, RLHF labelling, and paper reporting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List

from baas.core.ego_policy import EgoPolicy
from baas.core.env_adapter import EnvAdapter
from baas.core.metrics import IncidentThresholds
from baas.core.types import RolloutSpec
from baas.data.schema import ScenarioRecord
from baas.evaluation.runner import AdvController, evaluate_artefact


def collect_scenarios(
    method: str,
    specs: List[RolloutSpec],
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    thresholds: IncidentThresholds,
    make_controllers: Callable[[RolloutSpec], List[AdvController]],
    adversary_artefact: Any,
    *,
    baas_version: str,
    config_sha1: str,
    rollout_specs_sha1: str,
    env_cfg: Any = None,
) -> List[ScenarioRecord]:
    """Run evaluation and package results as ScenarioRecords."""
    from baas.evaluation.diversity import ego_relative_trajectory

    results = evaluate_artefact(
        specs, adapter, ego_policy, thresholds, make_controllers, env_cfg=env_cfg
    )

    records: List[ScenarioRecord] = []
    for r in results:
        records.append(ScenarioRecord(
            method=method,
            baas_version=baas_version,
            config_sha1=config_sha1,
            rollout_specs_sha1=rollout_specs_sha1,
            rollout_spec=r.spec,
            ego_trace=r.ego_trace,
            adv_traces=r.adv_traces,
            adversary_artefact=adversary_artefact,
            outcome=r.metrics,
            phi=r.metrics.phi,
        ))
    return records
