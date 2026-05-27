"""Dataset schema: one record per generated adversarial scenario."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from baas.core.metrics import EpisodeMetrics
from baas.core.types import RolloutSpec


@dataclass
class ScenarioRecord:
    """A single adversarial scenario stored in the dataset.

    Produced by data/collector.py during or after a search run.
    The label field is filled by data/labeller.py if human feedback is collected.
    """

    method: str
    baas_version: str
    config_sha1: str
    rollout_specs_sha1: str

    rollout_spec: RolloutSpec

    # Trajectories: (T, 4) float32, columns [x, y, psi, v]
    ego_trace: np.ndarray
    adv_traces: List[np.ndarray]

    # Adversary artefact (method-specific):
    # action-sequence methods: List[int] of length horizon_steps * n_adv
    # RL methods: checkpoint path string
    adversary_artefact: Any

    outcome: EpisodeMetrics

    phi: Optional[Tuple[float, float]]

    label: Optional[str] = None
    label_notes: Optional[str] = None
