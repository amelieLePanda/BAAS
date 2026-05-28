"""Shared ego-training configuration.

policy_frequency=2 must match the evaluation environment exactly.
All DQN hyperparameters and reward settings come from the benchmark YAML
via EgoTrainConfig - no hardcoded values outside this dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass

from baas.adapters.highway_env.config import GRAYSCALE_OBS_CFG  # noqa: F401 (re-exported)


@dataclass
class EgoTrainConfig:
    # Environment
    env_id: str = "highway-v0"
    policy_frequency: int = 2
    simulation_frequency: int = 15
    horizon_steps: int = 240
    background_traffic: int = 6
    total_timesteps: int = 300_000
    seed: int = 0
    device: str = "auto"

    # DQN hyperparameters (proven settings for CNN ego on highway-env)
    learning_rate: float = 5e-4
    lr_schedule: str = "constant"   # "constant" or "linear"
    buffer_size: int = 15_000
    learning_starts: int = 200
    batch_size: int = 32
    gamma: float = 0.8
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 50
    exploration_fraction: float = 0.7
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.01

    # Reward shaping
    collision_reward: float = -3.0
    lane_change_reward: float = -0.1
    high_speed_reward: float = 0.1
    right_lane_reward: float = 0.1
