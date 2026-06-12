"""Render adversarial episodes to GIF or MP4."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional

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
    post_reset_fn: Optional[Callable[[Any], None]] = None,
) -> EpisodeResult:
    """Re-run an episode with render_mode='rgb_array' and collect frames."""
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
        post_reset_fn=post_reset_fn,
    )


def frames_to_gif(frames: List[np.ndarray], path: Path, *, fps: int = 10) -> None:
    """Write RGB frames to a GIF. Requires imageio."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise ImportError("pip install imageio")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps)
    logger.info("GIF saved to %s (%d frames @ %d fps)", path, len(frames), fps)


def frames_to_mp4(frames: List[np.ndarray], path: Path, *, fps: int = 10) -> None:
    """Write RGB frames to MP4. Requires imageio[ffmpeg]."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise ImportError("pip install 'imageio[ffmpeg]'")
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
    post_reset_fn: Optional[Callable[[Any], None]] = None,
) -> EpisodeResult:
    """Render an episode and save to GIF."""
    result = render_episode(
        spec, adapter, ego_policy, adv_controllers, thresholds,
        env_cfg=env_cfg, post_reset_fn=post_reset_fn,
    )
    if result.frames:
        frames_to_gif(result.frames, output_path, fps=fps)
    else:
        logger.warning("No frames captured. Check env render_mode.")
    return result


def resolve_replay_inputs(
    scenario_id: str,
    catalogue_path: Path,
    adapter: EnvAdapter,
    device: str = "cpu",
) -> "tuple[dict, RolloutSpec, List[AdvController], Optional[Callable[[Any], None]], Any, IncidentThresholds]":
    """Resolve everything needed to deterministically re-run a cataloged scenario.

    Looks up scenario_id, reconstructs adversary controllers from the stored
    artefact, finds the matching RolloutSpec, and derives env_cfg/thresholds
    from the run's own stored config. Returns
    (row, spec, adv_controllers, post_reset_fn, env_cfg, thresholds).
    """
    from baas.evaluation.benchmark import load_rollout_specs
    from baas.evaluation.config_loader import config_from_dict

    catalogue = json.loads(Path(catalogue_path).read_text(encoding="utf-8"))
    row = next((r for r in catalogue if r["scenario_id"] == scenario_id), None)
    if row is None:
        raise KeyError(f"scenario_id not found: {scenario_id}")

    method = row["method"]
    rollout_index = int(row["rollout_index"])
    release_dir = Path(catalogue_path).parent
    results_path = release_dir / row["results_path"]
    run_dir = results_path.parent

    results_json = json.loads(results_path.read_text(encoding="utf-8"))
    run_cfg = results_json["config"]
    bundle = config_from_dict(run_cfg)
    env_cfg = bundle.env
    thresholds = bundle.thresholds

    # Load the RolloutSpec for this rollout; fall back to regenerating from config
    # if rollout_specs.json was not saved by the run (older ppo_adversary / parameter_sweep runs).
    specs_path = run_dir / "rollout_specs.json"
    if specs_path.exists():
        specs = load_rollout_specs(specs_path)
    else:
        from baas.evaluation.benchmark import make_rollout_specs_from_config
        n_adv = int(results_json.get("n_adversaries", 1))
        specs = make_rollout_specs_from_config(run_cfg, n_adversaries=n_adv)
    spec = next(s for s in specs if s.rollout_index == rollout_index)

    # Reconstruct artefact path
    artefact_ref = row.get("artefact_reference", "")
    artefact_path = (release_dir / artefact_ref) if artefact_ref else None

    # Build controllers and optional post_reset_fn
    post_reset_fn = None
    adv_controllers: List[Any] = []

    if method == "parameter_sweep":
        if artefact_path and artefact_path.exists():
            art = json.loads(artefact_path.read_text(encoding="utf-8"))
            params = next(p for p in art["best_params"] if p["rollout_index"] == rollout_index)
            dx, dy, dv = float(params["dx"]), float(params["dy"]), float(params["dv"])
            post_reset_fn = lambda u: adapter.apply_background_perturbation(u, 1, dx, dy, dv)
        else:
            raise FileNotFoundError(f"Artefact not found for {scenario_id}: {artefact_ref}")

    elif method == "ppo_adversary":
        from baas.evaluation.controllers import make_controllers_ppo
        if not artefact_path or not artefact_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {artefact_ref}")
        adv_controllers = make_controllers_ppo(artefact_path, device)(spec)

    elif method == "qdrl":
        from baas.evaluation.controllers import make_controllers_ppo
        if not artefact_path or not artefact_path.exists():
            raise FileNotFoundError(f"QD-RL archive not found: {artefact_ref}")
        archive = json.loads(artefact_path.read_text(encoding="utf-8"))
        cells = archive["archive"]["cells"]
        best = max(cells, key=lambda c: float(c.get("objective", float("-inf"))))
        policy_path = Path(best["policy_path"])
        if not policy_path.exists():
            policy_path = release_dir / best["policy_path"]
        adv_controllers = make_controllers_ppo(policy_path, device)(spec)

    elif method == "map_elites":
        from baas.evaluation.controllers import make_controllers_map_elites
        if not artefact_path or not artefact_path.exists():
            raise FileNotFoundError(f"Archive not found: {artefact_ref}")
        adv_controllers = make_controllers_map_elites(artefact_path, run_cfg, 1)(spec)

    elif method == "king_light":
        from baas.evaluation.controllers import make_controllers_action_seq
        if not artefact_path or not artefact_path.exists():
            raise FileNotFoundError(
                f"KING-light artefact not found for {scenario_id}. "
                "Existing seed runs were produced before artefact saving was added."
            )
        adv_controllers = make_controllers_action_seq(artefact_path)(spec)

    else:
        raise ValueError(f"Replay not implemented for method: {method}")

    return row, spec, adv_controllers, post_reset_fn, env_cfg, thresholds


def replay_by_scenario_id(
    scenario_id: str,
    catalogue_path: Path,
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    thresholds: IncidentThresholds,
    output_path: Path,
    *,
    env_cfg: Any = None,
    device: str = "cpu",
    fps: int = 10,
) -> EpisodeResult:
    """Replay a scenario from the catalogue and save to GIF.

    Looks up scenario_id, reconstructs adversary controllers from the stored
    artefact, finds the matching RolloutSpec, and renders to output_path.
    """
    _row, spec, adv_controllers, post_reset_fn, resolved_env_cfg, _thresholds = resolve_replay_inputs(
        scenario_id, catalogue_path, adapter, device=device,
    )

    return render_episode_to_gif(
        spec, adapter, ego_policy, adv_controllers, thresholds, output_path,
        env_cfg=env_cfg if env_cfg is not None else resolved_env_cfg,
        fps=fps, post_reset_fn=post_reset_fn,
    )
