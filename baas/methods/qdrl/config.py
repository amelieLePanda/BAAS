"""QD-RL configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class QDRLConfig:
    n_adversaries: int = 1

    n_iters: int = 80
    n_emitters: int = 4
    emitter_batch_size: int = 4
    sigma0: float = 0.8
    snapshot_every: int = 5

    archive_grid_dims: List[int] = field(default_factory=lambda: [10, 10])
    dist_range: List[float] = field(default_factory=lambda: [0.0, 50.0])
    tcrit_range: List[float] = field(default_factory=list)

    burst_steps: int = 25_000
    n_envs: int = 4
    eval_rollouts: int = 6

    r_critical_range: List[float] = field(default_factory=lambda: [1.0, 2.0])
    r_adv_close_range: List[float] = field(default_factory=lambda: [-2.0, -0.2])
    r_adv_nuis_range: List[float] = field(default_factory=lambda: [-10.0, -2.0])

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_n_steps: int = 1024
    ppo_batch_size: int = 256
    ent_coef: float = 0.02
    clip_range: float = 0.2

    device: str = "auto"
    seed: int = 0
