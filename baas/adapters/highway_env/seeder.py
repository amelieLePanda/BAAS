"""Initial-state sampling for highway-env episodes.

Migrated from king_light/highway_seed.py with the torch dependency removed.
Returns numpy arrays throughout.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def sample_initial_states(
    env: "gymnasium.Env",
    n_adversaries: int,
    *,
    seed: int = 0,
    min_dist: float = 20.0,
    dx_min: float = 15.0,
    dx_max: float = 80.0,
    max_abs_dv: float = 8.0,
    max_tries: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reset the env and extract kinematic initial states for ego and adversaries.

    Tries up to max_tries resets until a valid configuration is found.

    Returns ego_state (4,) float32 as [x, y, psi, speed] and
    adv_states (n_adversaries, 4) float32 in the same layout.
    """
    if n_adversaries <= 0:
        raise ValueError("n_adversaries must be >= 1")

    unwrapped = env.unwrapped

    def _to_state(v: object) -> list:
        x, y = v.position  # type: ignore[attr-defined]
        return [float(x), float(y), float(v.heading), float(v.speed)]  # type: ignore[attr-defined]

    def _valid(ego: object, other: object) -> bool:
        ex, ey = float(ego.position[0]), float(ego.position[1])  # type: ignore[attr-defined]
        ox, oy = float(other.position[0]), float(other.position[1])  # type: ignore[attr-defined]
        dx = ox - ex
        dist = float(np.hypot(dx, oy - ey))
        if dist < min_dist or dx < dx_min or dx > dx_max:
            return False
        return abs(float(other.speed - ego.speed)) <= max_abs_dv  # type: ignore[attr-defined]

    ego_state: Optional[np.ndarray] = None
    adv_states: Optional[np.ndarray] = None

    for attempt in range(int(max_tries)):
        obs, _info = env.reset(seed=int(seed + attempt))

        try:
            ego = unwrapped.vehicle
            vehicles = list(unwrapped.road.vehicles)

            candidates = []
            for v in vehicles:
                if v is ego:
                    continue
                if _valid(ego, v):
                    ex, ey = float(ego.position[0]), float(ego.position[1])
                    d = float(np.hypot(float(v.position[0]) - ex, float(v.position[1]) - ey))
                    candidates.append((d, v))

            if len(candidates) < n_adversaries:
                continue

            candidates.sort(key=lambda t: t[0])
            chosen = [v for _, v in candidates[:n_adversaries]]

            ego_state = np.array(_to_state(ego), dtype=np.float32)
            adv_states = np.array([_to_state(v) for v in chosen], dtype=np.float32)
            logger.debug("Valid init found at attempt %d (seed=%d)", attempt, seed + attempt)
            break

        except Exception:
            # Fallback using raw observation array; constraint enforcement is best-effort
            arr = np.array(obs, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= (n_adversaries + 1) and arr.shape[1] >= 4:
                ego_state = arr[0, :4].astype(np.float32)
                adv_states = arr[1 : n_adversaries + 1, :4].astype(np.float32)
                break

    if ego_state is None or adv_states is None:
        raise RuntimeError(
            f"sample_initial_states: no valid configuration within max_tries={max_tries} "
            f"(min_dist={min_dist}, dx=[{dx_min}, {dx_max}], max_abs_dv={max_abs_dv})."
        )

    return ego_state, adv_states
