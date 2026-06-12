"""Shared dataclasses for the whole framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class AgentState:
    """Kinematic snapshot of one vehicle. SI units (m, rad, m/s)."""
    x: float
    y: float
    psi: float
    v: float


@dataclass(frozen=True)
class Perturb:
    """Initial-state perturbation applied after env reset.

    agents[0] is ego (dy, dv), agents[1:] are adversaries in order.
    """
    agents: Tuple[Tuple[float, float], ...]

    @classmethod
    def from_flat(
        cls,
        dy_ego: float,
        dv_ego: float,
        dy_adv: float,
        dv_adv: float,
        n_adversaries: int = 1,
    ) -> "Perturb":
        agents: Tuple[Tuple[float, float], ...] = ((dy_ego, dv_ego),) + tuple(
            (dy_adv, dv_adv) for _ in range(n_adversaries)
        )
        return cls(agents=agents)


@dataclass(frozen=True)
class RolloutSpec:
    """Everything needed to reproduce one evaluation rollout.

    horizon_steps is always in policy steps, never seconds.
    The only place that converts to seconds is adapters/highway_env/config.py.
    """
    rollout_index: int
    env_seed: int
    env_id: str
    horizon_steps: int
    n_adversaries: int
    background_traffic: int
    perturb: Perturb
    tag: Optional[str] = None


@dataclass
class EpisodeResult:
    """Full result of one simulated episode.

    Trace arrays are (T, 4) float32, columns: [x, y, psi, v].
    T is actual steps run, which may be less than horizon_steps if the episode ended early.
    """
    spec: RolloutSpec
    metrics: "EpisodeMetrics"

    ego_trace: np.ndarray
    adv_traces: List[np.ndarray]

    ego_actions: np.ndarray
    adv_actions: np.ndarray

    frames: Optional[List[np.ndarray]] = None
    rewards: Optional[np.ndarray] = None
