"""Ego policy interface and SB3 wrappers (DQN and PPO)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class EgoPolicy(ABC):

    @abstractmethod
    def act(self, obs: Any, *, deterministic: bool = True) -> int:
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (e.g. recurrent hidden states)."""
        ...


class DQNEgoPolicy(EgoPolicy):

    def __init__(self, model_path: str, device: str = "auto") -> None:
        from stable_baselines3 import DQN
        self._model = DQN.load(model_path, device=device)
        logger.info("Loaded DQN ego from %s", model_path)

    def act(self, obs: Any, *, deterministic: bool = True) -> int:
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return int(action)

    def reset(self) -> None:
        pass


class PPOEgoPolicy(EgoPolicy):

    def __init__(self, model_path: str) -> None:
        from stable_baselines3 import PPO
        self._model = PPO.load(model_path)
        logger.info("Loaded PPO ego from %s", model_path)

    def act(self, obs: Any, *, deterministic: bool = True) -> int:
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return int(action)

    def reset(self) -> None:
        pass
