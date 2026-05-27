"""BicycleProxy: differentiable bicycle-model surrogate for KING-light.

Migrated from king_light/bicycle.py with logic unchanged.
Single adversary only; not used in multi-agent QD methods.
"""
from __future__ import annotations

import torch


def softclip(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Smooth clamp via sigmoid, keeping gradients flowing at the boundaries."""
    return lo + (hi - lo) * torch.sigmoid(x)


class BicycleProxy(torch.nn.Module):
    """Differentiable bicycle-model dynamics.

    Used by the KING-light optimiser to roll out adversary trajectories under
    Gumbel-Softmax actions and backpropagate through them.

    State layout: (..., 4) as x, y, psi, v.
    Action layout: (..., 2) as steer_raw, accel_raw.
    """

    def __init__(
        self,
        wheelbase: float = 2.7,
        max_steer: float = 0.5,
        acc_lo: float = -6.0,
        acc_hi: float = 3.0,
        dt: float = 0.25,
    ) -> None:
        super().__init__()
        self.L = wheelbase
        self.max_steer = max_steer
        self.acc_lo = acc_lo
        self.acc_hi = acc_hi
        self.dt = dt

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Advance state by one dt.

        Uses unbind to avoid in-place writes into view tensors,
        which is important for correct autograd through the rollout loop.
        """
        steer = self.max_steer * torch.tanh(action[..., 0])
        accel = softclip(action[..., 1], self.acc_lo, self.acc_hi)
        x, y, psi, v = state.unbind(-1)
        beta = torch.atan(0.5 * torch.tan(steer))
        dx = v * torch.cos(psi + beta)
        dy = v * torch.sin(psi + beta)
        dpsi = (v / self.L) * torch.sin(beta) * 2.0
        return torch.stack(
            [x + dx * self.dt, y + dy * self.dt, psi + dpsi * self.dt, v + accel * self.dt],
            dim=-1,
        )
