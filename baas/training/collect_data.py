"""Data collection pipeline: records frames and kinematics per step.

Saves one .npz per episode. Used for ego training, world-model pre-training,
and RLHF labelling.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from baas.core.env_adapter import EnvAdapter
from baas.core.ego_policy import EgoPolicy
from baas.core.metrics import IncidentThresholds
from baas.core.rollout import run_episode
from baas.core.types import Perturb, RolloutSpec
from baas.evaluation.benchmark import make_rollout_specs as _make_rollout_specs
from baas.training.config import EgoTrainConfig

logger = logging.getLogger(__name__)


def _make_idle_controllers(n_adversaries: int) -> List[Any]:
    """Return controllers that always play IDLE (action 1)."""
    return [lambda obs: 1 for _ in range(n_adversaries)]


def _save_episode(
    result: Any,
    path: Path,
    *,
    record_frames: bool = False,
) -> None:
    """Save one episode's data to a .npz file."""
    arrays: dict = {
        "ego_trace": result.ego_trace,
        "ego_actions": result.ego_actions,
        "adv_actions": result.adv_actions,
    }

    if result.adv_traces:
        arrays["adv_traces"] = np.stack(result.adv_traces, axis=1)  # (T, N, 4)

    if record_frames and result.frames:
        arrays["frames"] = np.stack(result.frames, axis=0)

    try:
        m = result.metrics
        metrics_dict = {
            "ego_collision": m.ego_collision,
            "critical_incident": m.critical_incident,
            "episode_return": m.episode_return,
            "steps": m.steps,
        }
        if m.phi is not None:
            metrics_dict["phi"] = list(m.phi)
    except Exception:
        metrics_dict = {}

    arrays["metrics_json"] = np.array(json.dumps(metrics_dict))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)


def collect_episodes(
    cfg: EgoTrainConfig,
    output_dir: Path,
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    thresholds: IncidentThresholds,
    *,
    n_episodes: int = 1000,
    n_adversaries: int = 0,
    seed: int = 0,
    record_frames: bool = False,
) -> None:
    """Run n_episodes and save each to output_dir as episode_XXXXXX.npz.

    With n_adversaries=0, adversaries play IDLE, useful for collecting clean
    ego driving data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _make_rollout_specs(
        env_id=cfg.env_id,
        horizon_steps=cfg.horizon_steps,
        n_adversaries=n_adversaries,
        background_traffic=cfg.background_traffic,
        k=n_episodes,
        base_seed=seed,
        env_seed_mode="jitter",
        salt=0,
    )

    saved = 0
    for spec in specs:
        controllers = _make_idle_controllers(n_adversaries) if n_adversaries > 0 else []

        try:
            result = run_episode(
                spec=spec,
                adapter=adapter,
                ego_policy=ego_policy,
                adv_controllers=controllers,
                thresholds=thresholds,
                record_frames=record_frames,
                stop_on_critical=False,
            )
        except Exception as exc:
            logger.warning("Episode %d failed: %s", spec.rollout_index, exc)
            continue

        out_path = output_dir / f"episode_{spec.rollout_index:06d}.npz"
        _save_episode(result, out_path, record_frames=record_frames)
        saved += 1

        if saved % 100 == 0:
            logger.info("Saved %d / %d episodes", saved, n_episodes)

    logger.info("Collection complete: %d episodes saved to %s", saved, output_dir)
