"""Execute rollout specs against a search method and collect results.

Takes a frozen list of RolloutSpec objects and an adversary artefact, runs
each rollout through core/rollout.py, and saves results with full provenance.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import dataclasses

import numpy as np

import baas
from baas.core.ego_policy import EgoPolicy
from baas.core.env_adapter import EnvAdapter
from baas.core.metrics import EpisodeMetrics, IncidentThresholds, classify_difficulty
from baas.core.rollout import AdvController, run_episode
from baas.core.types import EpisodeResult, RolloutSpec
from baas.evaluation.benchmark import rollout_specs_sha1

logger = logging.getLogger(__name__)


def _metrics_to_dict(m: EpisodeMetrics) -> Dict[str, Any]:
    """Serialise EpisodeMetrics, replacing NaN/Inf with None for JSON."""
    d: Dict[str, Any] = {
        "ego_collision": m.ego_collision,
        "ego_offroad": m.ego_offroad,
        "adv_ego_collision": m.adv_ego_collision,
        "adv_adv_collision_count": m.adv_adv_collision_count,
        "critical_incident": m.critical_incident,
        "critical_incident_adv": m.critical_incident_adv,
        "termination_reason": m.termination_reason,
        "time_to_collision_steps": m.time_to_collision_steps,
        "time_to_critical_incident_steps": m.time_to_critical_incident_steps,
        "time_to_critical_incident_adv_steps": m.time_to_critical_incident_adv_steps,
        "min_dist_ego_any": m.min_dist_ego_any,
        "min_dist_ego_adv": m.min_dist_ego_adv,
        "min_ttc_ego_any": m.min_ttc_ego_any,
        "min_ttc_ego_adv": m.min_ttc_ego_adv,
        "ego_speed_min": m.ego_speed_min,
        "ego_speed_mean": m.ego_speed_mean,
        "ego_accel_min": m.ego_accel_min,
        "ego_hard_brake_count": m.ego_hard_brake_count,
        "ego_near_stop_steps": m.ego_near_stop_steps,
        "adv_crashed": m.adv_crashed,
        "adv_nuisance_crash": m.adv_nuisance_crash,
        "max_adv_accel": m.max_adv_accel,
        "max_adv_jerk": m.max_adv_jerk,
        "episode_return": m.episode_return,
        "steps": m.steps,
        "phi": list(m.phi) if m.phi is not None else None,
        "feasibility": m.feasibility,
        "difficulty_label": m.difficulty_label,
    }
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}


def action_seq_controllers(
    action_seqs: List[List[int]],
    horizon_steps: int,
) -> List[AdvController]:
    """Wrap per-adversary action sequences as stateful controller callables.

    Each controller is a closure that returns the pre-determined action at
    step t, ignoring the observation. Create a fresh list for each episode.
    """
    controllers: List[AdvController] = []
    for seq in action_seqs:
        _seq = list(seq)
        _t = [0]

        def _ctrl(obs: Any, _seq=_seq, _t=_t) -> int:
            t = _t[0]
            _t[0] += 1
            return int(_seq[t]) if t < len(_seq) else 1  # IDLE fallback

        controllers.append(_ctrl)
    return controllers


_FEASIBILITY_SEED_OFFSET = 10_000


def estimate_feasibility(
    spec: RolloutSpec,
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    thresholds: IncidentThresholds,
    make_controllers: Callable[[RolloutSpec], List[AdvController]],
    *,
    env_cfg: Any = None,
    n_reruns: int = 10,
    post_reset_fn: Any = None,
) -> tuple:
    """Estimate feasibility for one scenario.

    Replays the same adversary n_reruns times with varied ego env seeds.
    Returns (feasibility, difficulty_label) where feasibility is the fraction of
    reruns in which the ego survives.

    post_reset_fn: optional callable passed to run_episode. Used by parameter
    sweep to reposition the background vehicle after each env reset.
    """
    survived = 0
    for i in range(n_reruns):
        rerun_spec = dataclasses.replace(
            spec, env_seed=spec.env_seed + _FEASIBILITY_SEED_OFFSET + i
        )
        controllers = make_controllers(rerun_spec)
        try:
            result = run_episode(
                spec=rerun_spec,
                adapter=adapter,
                ego_policy=ego_policy,
                adv_controllers=controllers,
                thresholds=thresholds,
                env_cfg=env_cfg,
                stop_on_critical=False,
                post_reset_fn=post_reset_fn,
            )
            if not result.metrics.ego_collision and result.metrics.ego_offroad is not True:
                survived += 1
        except Exception as exc:
            logger.debug("Feasibility rerun %d failed: %s", i, exc)

    feasibility = survived / n_reruns
    return feasibility, classify_difficulty(feasibility)


def evaluate_artefact(
    specs: List[RolloutSpec],
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    thresholds: IncidentThresholds,
    make_controllers: Callable[[RolloutSpec], List[AdvController]],
    *,
    env_cfg: Any = None,
    stop_on_critical: bool = False,
    n_feasibility_reruns: int = 0,
    adv_obs_extractor: Any = None,
) -> List[EpisodeResult]:
    """Run every rollout spec and return results.

    make_controllers is called fresh per rollout so controllers are stateless
    across episodes. stop_on_critical=False (default) runs full episodes so
    adversaries get to act - critical_any from background traffic no longer
    terminates the episode prematurely. Set n_feasibility_reruns > 0 to
    estimate feasibility and difficulty label for each result.
    """
    results: List[EpisodeResult] = []
    for spec in specs:
        controllers = make_controllers(spec)
        try:
            result = run_episode(
                spec=spec,
                adapter=adapter,
                ego_policy=ego_policy,
                adv_controllers=controllers,
                thresholds=thresholds,
                env_cfg=env_cfg,
                stop_on_critical=stop_on_critical,
                adv_obs_extractor=adv_obs_extractor,
            )
        except Exception as exc:
            logger.warning("Rollout %d failed: %s", spec.rollout_index, exc)
            continue

        if n_feasibility_reruns > 0:
            feasibility, label = estimate_feasibility(
                spec, adapter, ego_policy, thresholds, make_controllers,
                env_cfg=env_cfg, n_reruns=n_feasibility_reruns,
            )
            result.metrics.feasibility = feasibility
            result.metrics.difficulty_label = label
            logger.debug(
                "Rollout %d feasibility=%.2f  label=%s",
                spec.rollout_index, feasibility, label,
            )

        results.append(result)
        logger.info(
            "Rollout %d/%d  critical=%s  feasibility=%s  reason=%s",
            spec.rollout_index + 1, len(specs),
            result.metrics.critical_incident,
            f"{result.metrics.feasibility:.2f}" if result.metrics.feasibility is not None else "n/a",
            result.metrics.termination_reason,
        )
    return results


_TRACE_DOWNSAMPLE = 30  # steps stored per trace (matches Fréchet computation default)


def _downsample_trace(arr: np.ndarray, n: int) -> List[List[float]]:
    """Downsample a (T, D) trace to n rows and convert to nested list for JSON."""
    T = len(arr)
    if T == 0:
        return []
    if T <= n:
        return arr.tolist()
    idx = np.round(np.linspace(0, T - 1, n)).astype(int)
    return arr[idx].tolist()


def save_results(
    results: List[EpisodeResult],
    path: Path,
    *,
    method: str,
    config: Dict[str, Any],
    config_sha1: str,
    specs: List[RolloutSpec],
    run_id: str = "",
    trace_downsample: int = _TRACE_DOWNSAMPLE,
) -> None:
    """Write evaluation results to JSON with full provenance block.

    Ego and adversary traces are downsampled to trace_downsample steps and stored
    per rollout so that diversity metrics (Fréchet, dispersion) can be computed
    in summarise.py without re-running the environment.
    """
    specs_sha = rollout_specs_sha1(specs)

    rollout_entries = []
    for r in results:
        entry: Dict[str, Any] = {
            "rollout_index": r.spec.rollout_index,
            "env_seed": r.spec.env_seed,
            "metrics": _metrics_to_dict(r.metrics),
        }
        # Store downsampled traces for diversity analysis
        if r.ego_trace is not None and len(r.ego_trace) > 0:
            entry["ego_trace_ds"] = _downsample_trace(
                np.asarray(r.ego_trace, dtype=np.float32), trace_downsample
            )
        if r.adv_traces:
            entry["adv_traces_ds"] = [
                _downsample_trace(
                    np.asarray(tr, dtype=np.float32), trace_downsample
                )
                for tr in r.adv_traces
            ]
        rollout_entries.append(entry)

    n_adversaries = specs[0].n_adversaries if specs else 0
    payload: Dict[str, Any] = {
        "baas_version": baas.__version__,
        "method": method,
        "n_adversaries": n_adversaries,
        "run_id": run_id,
        "config_sha1": config_sha1,
        "rollout_specs_sha1": specs_sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "rollouts": rollout_entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved %d rollout results to %s", len(results), path)
