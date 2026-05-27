"""Single source of truth for highway-env timing and benchmark configuration.

This is the only file in MADS that converts horizon_steps to seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


# Canonical GrayscaleObservation config used by all ego policies and env setup.
# Imported by training/config.py and used in base_env_config() below.
GRAYSCALE_OBS_CFG: Dict[str, Any] = {
    "type": "GrayscaleObservation",
    "observation_shape": (128, 64),  # (W, H) as expected by highway-env
    "stack_size": 4,
    "weights": [0.2989, 0.5870, 0.1140],
    "scaling": 1.75,
}

# LidarObservation config for the lidar ablation study.
# Returns shape (cells, 2): [normalised_distance, normalised_relative_speed] per cell.
# Flat dim per agent = 64 * 2 = 128.  For N adversaries: N * 128.
LIDAR_OBS_CFG: Dict[str, Any] = {
    "type": "LidarObservation",
    "cells": 64,
    "maximum_range": 60.0,
    "normalize": True,
}

# Number of features per lidar cell (distance + speed)
LIDAR_FEATURES = 2
# Flat observation dim for a single agent with the canonical lidar config
LIDAR_OBS_DIM = LIDAR_OBS_CFG["cells"] * LIDAR_FEATURES  # 128


@dataclass(frozen=True)
class HighwayEnvBenchmarkConfig:
    """All parameters needed to run a highway-env benchmark experiment.

    Loaded from a YAML file once at experiment start.
    """

    env_id: str = "highway-v0"

    # Always set explicitly even when matching highway-env defaults.
    # policy_frequency=2 is per the official GrayscaleObservation/CNN example in the highway-env docs.
    # simulation_frequency=15 is the highway-v0 default.
    policy_frequency: int = 2
    simulation_frequency: int = 15

    # Observation mode: "grayscale" (default, CNN ego) or "lidar" (MLP ego, ablation study)
    obs_mode: str = "grayscale"

    # Episode length in policy steps (canonical unit throughout MADS)
    horizon_steps: int = 240  # = 120 s at policy_frequency=2

    background_traffic: int = 6
    vehicles_density: float = 1.0

    # Rollout set parameters
    k: int = 30
    base_seed: int = 0
    env_seed_mode: str = "jitter"   # "jitter" | "fixed"
    salt: int = 0

    # Initial-state perturbation ranges
    ego_dy_eps: float = 0.5
    ego_dv_eps: float = 1.0
    adv_dy_eps: float = 0.5
    adv_dv_eps: float = 1.0

    # Incident thresholds
    critical_dist: float = 6.0
    nuisance_dist: float = 12.0
    ttc_crit_s: float = 1.5
    dx_near_m: float = 5.0
    dy_near_m: float = 1.0

    def duration_seconds(self) -> int:
        """Convert horizon_steps to seconds for highway-env's duration field.

        This is the only place in MADS that performs this conversion.
        """
        secs = float(self.horizon_steps) / float(self.policy_frequency)
        return max(1, int(round(secs)))

    def base_env_config(
        self,
        n_adversaries: int = 1,
        render_agent_observations: bool = False,
        obs_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the env config dict passed to gymnasium.make().

        obs_mode: "grayscale" (default, GrayscaleObservation + CNN ego) or
                  "lidar" (LidarObservation + MLP ego, ablation study).
                  Defaults to self.obs_mode when not provided.
        """
        _obs_mode = obs_mode if obs_mode is not None else self.obs_mode
        cfg: Dict[str, Any] = {
            "policy_frequency": self.policy_frequency,
            "simulation_frequency": self.simulation_frequency,
            "duration": self.duration_seconds(),
            "vehicles_count": 1 + n_adversaries + self.background_traffic,
            "controlled_vehicles": 1 + n_adversaries,
            "vehicles_density": self.vehicles_density,
        }
        if render_agent_observations:
            cfg["render_agent_observations"] = True

        obs_cfg = LIDAR_OBS_CFG if _obs_mode == "lidar" else GRAYSCALE_OBS_CFG

        if n_adversaries >= 1:
            # MultiAgentObservation wraps the per-agent obs type, returning a
            # tuple (ego_obs, adv_obs, ...) rather than a single array.
            cfg["observation"] = {
                "type": "MultiAgentObservation",
                "observation_config": obs_cfg,
            }
        else:
            cfg["observation"] = obs_cfg
        return cfg
