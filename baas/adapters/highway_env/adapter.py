"""Concrete EnvAdapter implementation for highway-env.

This is the only file in MADS that imports highway_env directly.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np

from baas.core.env_adapter import EnvAdapter
from baas.core.types import AgentState, Perturb

import highway_env  # noqa: F401  (registers gym envs as a side-effect)

logger = logging.getLogger(__name__)

# DiscreteMetaAction alphabet: 0 LANE_LEFT  1 IDLE  2 LANE_RIGHT  3 FASTER  4 SLOWER
_N_META_ACTIONS = 5
_STEER_RAW = 1.5
_ACC_RAW = 2.0
_BRAKE_RAW = -2.0

_ACTION_MAP: Dict[int, Tuple[float, float]] = {
    0: (-_STEER_RAW, 0.0),
    1: (0.0, 0.0),
    2: (_STEER_RAW, 0.0),
    3: (0.0, _ACC_RAW),
    4: (0.0, _BRAKE_RAW),
}

_MULTI_AGENT_ACTION_CFG: Dict[str, Any] = {
    "action": {
        "type": "MultiAgentAction",
        "action_config": {"type": "DiscreteMetaAction"},
    },
}


class HighwayEnvAdapter(EnvAdapter):
    """EnvAdapter for the highway-env / highway-v0 family."""

    def make_env(
        self,
        cfg: Any,
        n_adversaries: int = 1,
        render_mode: Optional[str] = None,
    ) -> gymnasium.Env:
        """Create a highway-env environment with explicit timing config.

        policy_frequency and simulation_frequency are always set explicitly.
        """
        if cfg is None:
            env_config: Dict[str, Any] = {}
            env_id = "highway-v0"
        elif hasattr(cfg, "base_env_config"):
            env_config = cfg.base_env_config(n_adversaries=n_adversaries)
            env_id = cfg.env_id
        elif isinstance(cfg, dict):
            env_config = dict(cfg)
            env_id = env_config.pop("env_id", "highway-v0")
        else:
            raise TypeError(f"Unsupported cfg type: {type(cfg)}")

        if n_adversaries >= 1:
            env_config.update(_MULTI_AGENT_ACTION_CFG)
            env_config["controlled_vehicles"] = 1 + n_adversaries

        env = gymnasium.make(env_id, render_mode=render_mode, config=env_config)
        logger.debug("Created env %s  n_adv=%d", env_id, n_adversaries)
        return env

    def extract_state(self, vehicle: Any) -> AgentState:
        """Extract (x, y, psi, v) from a highway-env vehicle."""
        return AgentState(
            x=float(vehicle.position[0]),
            y=float(vehicle.position[1]),
            psi=float(getattr(vehicle, "heading", 0.0)),
            v=float(getattr(vehicle, "speed", 0.0)),
        )

    def apply_perturbation(self, unwrapped: Any, p: Perturb) -> None:
        """Apply lateral and speed offsets to controlled vehicles post-reset.

        In highway-env, position[1] is lateral and speed is a float attribute.
        """
        controlled = list(getattr(unwrapped, "controlled_vehicles", []))
        if not controlled:
            ego = getattr(unwrapped, "vehicle", None)
            controlled = [ego] if ego is not None else []

        for i, (dy, dv) in enumerate(p.agents):
            if i >= len(controlled):
                break
            v = controlled[i]
            if v is None:
                continue
            v.position[1] = float(v.position[1]) + float(dy)
            v.speed = max(0.0, float(getattr(v, "speed", 0.0)) + float(dv))

    def apply_background_perturbation(
        self, unwrapped: Any, vehicle_index: int, dx: float, dy: float, dv: float
    ) -> None:
        """Reposition a background (uncontrolled) vehicle after reset.

        vehicle_index is into road.vehicles, where index 0 is the ego.
        Used by the parameter sweep baseline.
        """
        vehicles = list(getattr(getattr(unwrapped, "road", None), "vehicles", []) or [])
        if vehicle_index >= len(vehicles):
            logger.debug("apply_background_perturbation: vehicle_index %d out of range (%d vehicles)", vehicle_index, len(vehicles))
            return
        v = vehicles[vehicle_index]
        v.position[0] = float(v.position[0]) + float(dx)
        v.position[1] = float(v.position[1]) + float(dy)
        v.speed = max(0.0, float(getattr(v, "speed", 0.0)) + float(dv))

    def get_kinematic_obs(
        self,
        unwrapped: Any,
        observer: Any,
        vehicles_all: List[Any],
        n_obs: int = 5,
    ) -> np.ndarray:
        """Return a (n_obs, 5) kinematic observation centred on observer.

        Features per row: [presence, dx, dy, dvx, dvy] normalised.
        Matches the shape of highway-env KinematicsObservation (n_vehicles, n_features).
        """
        NORM = np.array([1.0, 100.0, 10.0, 30.0, 10.0], dtype=np.float32)

        obs_x = float(observer.position[0])
        obs_y = float(observer.position[1])
        heading = float(getattr(observer, "heading", 0.0))
        spd = float(getattr(observer, "speed", 0.0))
        obs_vx = spd * float(np.cos(heading))
        obs_vy = spd * float(np.sin(heading))

        rows: List[List[float]] = [[1.0, 0.0, 0.0, 0.0, 0.0]]

        others = sorted(
            [v for v in vehicles_all if v is not observer],
            key=lambda v: float(np.hypot(
                float(v.position[0]) - obs_x,
                float(v.position[1]) - obs_y,
            )),
        )

        for v in others[: n_obs - 1]:
            h = float(getattr(v, "heading", 0.0))
            s = float(getattr(v, "speed", 0.0))
            rows.append([
                1.0,
                float(v.position[0]) - obs_x,
                float(v.position[1]) - obs_y,
                s * float(np.cos(h)) - obs_vx,
                s * float(np.sin(h)) - obs_vy,
            ])

        while len(rows) < n_obs:
            rows.append([0.0, 0.0, 0.0, 0.0, 0.0])

        result = np.array(rows[:n_obs], dtype=np.float32)
        result /= NORM
        return result

    def is_crashed(self, vehicle: Any) -> bool:
        return bool(getattr(vehicle, "crashed", False))

    def get_controlled_vehicles(self, unwrapped: Any) -> List[Any]:
        """Return controlled vehicles, ego at index 0."""
        return list(getattr(unwrapped, "controlled_vehicles", []) or [])

    def action_space_size(self) -> int:
        return _N_META_ACTIONS

    def action_map(self) -> Dict[int, Tuple[float, float]]:
        """Return action_id mapped to (steer_raw, accel_raw).

        Used by KING-light's BicycleProxy.
        """
        return dict(_ACTION_MAP)
