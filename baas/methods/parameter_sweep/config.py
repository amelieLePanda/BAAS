"""Config for the parameter sweep baseline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ParameterSweepConfig:
    """Grid over adversary initial conditions. Single adversary only."""

    # Longitudinal offsets to try (metres, relative to reset position).
    delta_x_grid: List[float] = field(default_factory=lambda: [-40.0, -20.0, 0.0, 20.0])

    # Lateral offsets to try (metres).
    delta_y_grid: List[float] = field(default_factory=lambda: [-3.5, 0.0, 3.5])

    # Speed offsets to try (m/s).
    delta_v_grid: List[float] = field(default_factory=lambda: [-5.0, 0.0, 5.0])
