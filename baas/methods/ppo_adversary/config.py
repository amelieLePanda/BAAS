"""PPO adversary training configuration."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PPOAdvConfig:
    n_adversaries: int = 1
    total_timesteps: int = 500_000
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 2048
    batch_size: int = 64
    ent_coef: float = 0.01
    clip_range: float = 0.2
    min_ttf_steps: int = 10       # early-collision down-weighting threshold (t_min)
    # Reward shaping coefficients (see Eq. 4 in paper)
    w_proximity: float = 0.05     # weight on ψ(d_ego,adv) proximity shaping
    c_time_step: float = 0.001    # per-step time cost
    r_critical: float = 1.5       # terminal reward on first critical incident
    r_adv_crash_close: float = -1.0    # penalty when adversary crashes near ego
    r_adv_crash_nuisance: float = -6.0 # penalty when adversary crashes far from ego
    device: str = "auto"
    seed: int = 0
