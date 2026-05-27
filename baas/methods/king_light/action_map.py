"""META_TO_PROXY tensor for KING-light.

Built from adapter.action_map() so the adapter remains the single source
of truth for the DiscreteMetaAction to (steer, accel) mapping.
"""
from __future__ import annotations

import torch

from baas.core.env_adapter import EnvAdapter


def get_meta_to_proxy(adapter: EnvAdapter, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return META_TO_PROXY shaped (N_ACTIONS, 2) from the adapter's action map."""
    amap = adapter.action_map()
    rows = [amap[i] for i in range(len(amap))]
    return torch.tensor(rows, dtype=torch.float32, device=device)
