"""Abstract environment adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np

import gymnasium

from baas.core.types import AgentState, Perturb


class EnvAdapter(ABC):

    @abstractmethod
    def make_env(
        self,
        cfg: Any,
        n_adversaries: int = 1,
        render_mode: Optional[str] = None,
    ) -> gymnasium.Env:
        """Build and return a Gymnasium environment.

        policy_frequency and simulation_frequency must always be set explicitly,
        never relying on environment defaults.
        """
        ...

    @abstractmethod
    def extract_state(self, vehicle: Any) -> AgentState:
        """Read (x, y, psi, v) from a single vehicle object."""
        ...

    @abstractmethod
    def apply_perturbation(self, unwrapped: Any, p: Perturb) -> None:
        """Apply (dy, dv) offsets to controlled vehicles after reset.

        p.agents[0] is ego, p.agents[1:] are adversaries in order.
        """
        ...

    @abstractmethod
    def is_crashed(self, vehicle: Any) -> bool:
        ...

    @abstractmethod
    def get_controlled_vehicles(self, unwrapped: Any) -> List[Any]:
        """Return controlled vehicles with ego at index 0."""
        ...

    @abstractmethod
    def action_space_size(self) -> int:
        ...

    @abstractmethod
    def action_map(self) -> Dict[int, Tuple[float, float]]:
        """action_id mapped to (steer_raw, accel_raw), used by KING-light proxy."""
        ...

    def apply_background_perturbation(
        self, unwrapped: Any, vehicle_index: int, dx: float, dy: float, dv: float
    ) -> None:
        """Reposition a background (uncontrolled) vehicle after reset.

        No-op by default. Implemented in HighwayEnvAdapter for the parameter sweep.
        """

    def get_kinematic_obs(
        self,
        unwrapped: Any,
        observer: Any,
        vehicles_all: List[Any],
        n_obs: int = 5,
    ) -> "np.ndarray":
        """Return a (n_obs, 5) kinematic observation centred on observer.

        Features per row: [presence, dx, dy, dvx, dvy] normalised.
        Row 0 is the observer itself (dx=dy=dvx=dvy=0).
        Rows 1..n_obs-1 are nearby vehicles sorted by distance. Zero-padded.

        Returns zeros by default. Implemented in HighwayEnvAdapter.
        """
        import numpy as np
        return np.zeros((n_obs, 5), dtype=np.float32)
