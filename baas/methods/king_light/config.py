"""KING-light configuration.

Single adversary only. Gradient-based proxy optimisation is not compatible
with multi-agent coordination constraints.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class KingLightConfig:
    # KING-light is single-adversary only.
    n_adversaries: int = 1

    n_opt_steps: int = 400
    lr: float = 0.05
    device: str = "cpu"

    attack_horizon: int = 60

    gumbel_tau: float = 1.0

    max_steer: float = 0.35
    acc_lo: float = -6.0
    acc_hi: float = 6.0

    collision_radius: float = 2.0

    warmup_steps: int = 5
    warmup_penalty: float = 1.0

    progress_weight: float = 0.05
