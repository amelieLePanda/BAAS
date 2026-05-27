"""Episode metrics and incident detection helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IncidentThresholds:
    """Incident detection thresholds, always loaded from config."""
    critical_dist: float = 6.0
    nuisance_dist: float = 12.0
    ttc_crit_s: float = 1.5
    dx_near_m: float = 5.0
    dy_near_m: float = 1.0
    v_stop_mps: float = 1.0
    stop_streak_steps: int = 5
    hard_brake_a: float = -4.0
    brake_stop_prox_ttc_s: float = 2.0
    brake_stop_prox_dx_m: float = 5.0
    brake_stop_prox_dy_m: float = 1.0


@dataclass
class EpisodeMetrics:
    ego_collision: bool
    ego_offroad: Optional[bool]
    adv_ego_collision: bool
    adv_adv_collision_count: int

    critical_incident: bool
    critical_incident_adv: bool

    termination_reason: str

    time_to_collision_steps: int
    time_to_critical_incident_steps: int
    time_to_critical_incident_adv_steps: int

    min_dist_ego_any: float
    min_dist_ego_adv: float
    min_ttc_ego_any: float
    min_ttc_ego_adv: float

    ego_speed_min: float
    ego_speed_mean: float
    ego_accel_min: float
    ego_hard_brake_count: int
    ego_near_stop_steps: int

    adv_crashed: bool
    adv_nuisance_crash: bool
    max_adv_accel: float
    max_adv_jerk: float

    episode_return: float
    steps: int

    phi: Optional[Tuple[float, float]] = None

    feasibility: Optional[float] = None
    difficulty_label: Optional[str] = None


def classify_difficulty(feasibility: float) -> str:
    if feasibility < 0.10:
        return "degenerate"
    if feasibility < 0.30:
        return "hardcore"
    if feasibility < 0.50:
        return "hard"
    if feasibility < 0.70:
        return "medium"
    if feasibility < 1.00:
        return "easy"
    return "trivial"


def dist_pair(v1: Any, v2: Any) -> float:
    dx = float(v1.position[0] - v2.position[0])
    dy = float(v1.position[1] - v2.position[1])
    return float(np.hypot(dx, dy))


def min_dist_ego_any(ego: Any, vehicles: List[Any]) -> float:
    md = float("inf")
    ex, ey = float(ego.position[0]), float(ego.position[1])
    for v in vehicles:
        if v is ego:
            continue
        d = float(np.hypot(float(v.position[0]) - ex, float(v.position[1]) - ey))
        if d < md:
            md = d
    return float(md) if np.isfinite(md) else float("nan")


def ttc_pair_longitudinal(ego: Any, other: Any, dy_gate: float) -> float:
    """Longitudinal TTC. Returns nan when outside lateral gate or not closing.

    highway-env convention: x is longitudinal, y is lateral.
    """
    ex, ey = float(ego.position[0]), float(ego.position[1])
    ox, oy = float(other.position[0]), float(other.position[1])

    if abs(oy - ey) > dy_gate:
        return float("nan")

    dx = ox - ex
    rel_v = float(getattr(ego, "speed", 0.0)) - float(getattr(other, "speed", 0.0))

    if dx > 0.0 and rel_v > 1e-6:
        return dx / rel_v
    if dx < 0.0 and rel_v < -1e-6:
        return (-dx) / (-rel_v)
    return float("nan")


def min_ttc_ego_any(ego: Any, vehicles: List[Any], dy_gate: float) -> float:
    best = float("inf")
    for v in vehicles:
        if v is ego:
            continue
        t = ttc_pair_longitudinal(ego, v, dy_gate=dy_gate)
        if np.isfinite(t) and 0.0 < t < best:
            best = t
    return float(best) if np.isfinite(best) and best < float("inf") else float("nan")


def proximity_flag(ego: Any, vehicles: List[Any], *, min_ttc_any: float, th: IncidentThresholds) -> bool:
    """True when ego is in a safety-relevant proximity zone."""
    if np.isfinite(min_ttc_any) and min_ttc_any < th.brake_stop_prox_ttc_s:
        return True
    ex, ey = float(ego.position[0]), float(ego.position[1])
    for v in vehicles:
        if v is ego:
            continue
        dx = float(v.position[0]) - ex
        dy = float(v.position[1]) - ey
        if abs(dy) <= th.brake_stop_prox_dy_m and abs(dx) <= th.brake_stop_prox_dx_m:
            return True
    return False


def ego_offroad_best_effort(ego: Any) -> Optional[bool]:
    """Returns True if ego is off-road, None if the flag is not available."""
    try:
        on_road = getattr(ego, "on_road", None)
        if on_road is not None:
            return not bool(on_road)
    except Exception:
        pass
    return None
