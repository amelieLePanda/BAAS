"""Render adversarial episodes to GIF or MP4."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

import numpy as np

from baas.core.ego_policy import EgoPolicy
from baas.core.env_adapter import EnvAdapter
from baas.core.metrics import IncidentThresholds
from baas.core.rollout import AdvController, run_episode
from baas.core.types import EpisodeResult, RolloutSpec

logger = logging.getLogger(__name__)


def render_episode(
    spec: RolloutSpec,
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    adv_controllers: List[AdvController],
    thresholds: IncidentThresholds,
    *,
    env_cfg: Any = None,
) -> EpisodeResult:
    """Re-run an episode with render_mode='rgb_array' and collect frames.

    Runs to completion so the full replay is captured.
    """
    return run_episode(
        spec=spec,
        adapter=adapter,
        ego_policy=ego_policy,
        adv_controllers=adv_controllers,
        thresholds=thresholds,
        env_cfg=env_cfg,
        render_mode="rgb_array",
        record_frames=True,
        stop_on_critical=False,
    )


def frames_to_gif(frames: List[np.ndarray], path: Path, *, fps: int = 10) -> None:
    """Write a list of RGB frames to a GIF file. Requires imageio."""
    try:
        import imageio.v2 as imageio  # type: ignore[import]
    except ImportError:
        raise ImportError("imageio is required for GIF export.  pip install imageio")

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps)
    logger.info("GIF saved to %s (%d frames @ %d fps)", path, len(frames), fps)


def frames_to_mp4(frames: List[np.ndarray], path: Path, *, fps: int = 10) -> None:
    """Write a list of RGB frames to an MP4 file. Requires imageio[ffmpeg]."""
    try:
        import imageio.v2 as imageio  # type: ignore[import]
    except ImportError:
        raise ImportError("imageio is required for MP4 export.  pip install 'imageio[ffmpeg]'")

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    logger.info("MP4 saved to %s (%d frames @ %d fps)", path, len(frames), fps)


def render_episode_to_gif(
    spec: RolloutSpec,
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    adv_controllers: List[AdvController],
    thresholds: IncidentThresholds,
    output_path: Path,
    *,
    env_cfg: Any = None,
    fps: int = 10,
) -> EpisodeResult:
    """Convenience: render an episode and save directly to GIF."""
    result = render_episode(spec, adapter, ego_policy, adv_controllers, thresholds, env_cfg=env_cfg)
    if result.frames:
        frames_to_gif(result.frames, output_path, fps=fps)
    else:
        logger.warning("No frames captured; is the env render_mode set correctly?")
    return result
