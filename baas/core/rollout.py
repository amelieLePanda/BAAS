"""Environment-agnostic episode runner.

run_episode is the single entry point used by all search methods and the
evaluation framework.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable, List, Optional

import numpy as np

from baas.core.env_adapter import EnvAdapter
from baas.core.ego_policy import EgoPolicy
from baas.core.metrics import (
    EpisodeMetrics,
    IncidentThresholds,
    dist_pair,
    ego_offroad_best_effort,
    min_dist_ego_any,
    min_ttc_ego_any,
    proximity_flag,
    ttc_pair_longitudinal,
)
from baas.core.types import EpisodeResult, RolloutSpec

logger = logging.getLogger(__name__)

_EGO_COLOUR = (50, 200, 0)    # green - matches highway-env default ego colour
_ADV_COLOUR = (225, 225, 0)   # yellow
_TRAFFIC_COLOUR = (100, 200, 255)  # blue - highway-env default for IDM traffic

AdvController = Callable[[Any], int]


def _try_set_colour(vehicle: Any, rgb: tuple) -> None:
    try:
        setattr(vehicle, "color", tuple(int(c) for c in rgb))
    except Exception:
        pass


def _policy_stack_depth(ego_policy: EgoPolicy) -> int:
    """Return expected frame-stack depth from ego policy's observation space.

    CNN policies on stacked grayscale frames have obs shape (C, H, W).
    Returns 0 for vector-obs policies.
    """
    space = getattr(ego_policy, "observation_space", None)
    s = getattr(space, "shape", ())
    return int(s[0]) if len(s) == 3 else 0


def _maybe_stack(obs: Any, stacker: Optional[deque], n: int):
    """Stack 2-D grayscale (H, W) to (C, H, W) when policy expects CHW."""
    if n <= 0 or not hasattr(obs, "ndim") or obs.ndim != 2:
        return obs, stacker
    if stacker is None:
        stacker = deque(maxlen=n)
    if len(stacker) == 0:
        for _ in range(n):
            stacker.append(obs.copy())
    else:
        stacker.append(obs.copy())
    return np.stack(list(stacker), axis=0), stacker


def _add_batch_dim(obs: Any) -> Any:
    """SB3 CNN predict() expects a batch dimension."""
    if isinstance(obs, np.ndarray) and obs.ndim == 3:
        return obs[None, ...]
    return obs


def _ego_obs_from_multiagent(obs: Any) -> Any:
    """Extract ego observation from a MultiAgentObservation tuple.

    MultiAgentObservation returns a tuple/list where index 0 is ego.
    Falls back to the full obs if it is not a sequence (single-agent env).
    """
    if isinstance(obs, (list, tuple)) and len(obs) >= 1:
        return obs[0]
    return obs


def run_episode(
    spec: RolloutSpec,
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    adv_controllers: List[AdvController],
    thresholds: IncidentThresholds,
    *,
    env_cfg: Any = None,
    render_mode: Optional[str] = None,
    stop_on_critical: bool = True,
    record_frames: bool = False,
    nuisance_window_steps: int = 20,
    post_reset_fn: Optional[Callable[[Any], None]] = None,
    adv_obs_extractor: Optional[Callable[[Any, int], Any]] = None,
) -> EpisodeResult:
    """Run a single episode and return the full result.

    adv_controllers must be stateless across episodes. For fixed action sequences,
    use action_seq_controllers().

    adv_obs_extractor: optional callable (obs_tuple, adv_index) -> obs_array.
        When provided, adversary i receives adv_obs_extractor(obs, i) instead of
        adapter.get_kinematic_obs().  Use for lidar-mode models trained on env obs:
            adv_obs_extractor=lambda o, i: np.asarray(o[i+1], dtype=np.float32).flatten()
    """
    if len(adv_controllers) != spec.n_adversaries:
        raise ValueError(
            f"Expected {spec.n_adversaries} adv_controllers, got {len(adv_controllers)}."
        )

    multi_agent = spec.n_adversaries >= 1

    env = adapter.make_env(env_cfg, n_adversaries=spec.n_adversaries, render_mode=render_mode)
    obs, _info = env.reset(seed=spec.env_seed)
    unwrapped = env.unwrapped

    controlled = adapter.get_controlled_vehicles(unwrapped)
    ego = controlled[0] if controlled else getattr(unwrapped, "vehicle", None)
    if ego is None:
        env.close()
        raise RuntimeError("Env reset produced no ego vehicle.")

    advs = controlled[1 : 1 + spec.n_adversaries]

    _try_set_colour(ego, _EGO_COLOUR)
    for adv in advs:
        _try_set_colour(adv, _ADV_COLOUR)

    adapter.apply_perturbation(unwrapped, spec.perturb)
    if post_reset_fn is not None:
        post_reset_fn(unwrapped)

    policy_freq = float(
        unwrapped.config.get("policy_frequency", 1)
        if hasattr(unwrapped, "config") else 1
    )
    dt = 1.0 / policy_freq if policy_freq > 0 else 1.0

    ego_stacker: Optional[deque] = None
    ego_n_stack = _policy_stack_depth(ego_policy)

    ego_x: List[float] = []
    ego_y: List[float] = []
    ego_psi: List[float] = []
    ego_v: List[float] = []

    adv_x: List[List[float]] = [[] for _ in advs]
    adv_y: List[List[float]] = [[] for _ in advs]
    adv_psi: List[List[float]] = [[] for _ in advs]
    adv_v: List[List[float]] = [[] for _ in advs]
    adv_prev_v: List[Optional[float]] = [None] * len(advs)
    adv_prev_a: List[Optional[float]] = [None] * len(advs)

    ego_action_log: List[int] = []
    adv_action_logs: List[List[int]] = [[] for _ in advs]

    frames: Optional[List[np.ndarray]] = [] if record_frames else None
    if record_frames:
        fr0 = env.render()
        if fr0 is not None:
            frames.append(fr0)  # type: ignore[union-attr]

    episode_return = 0.0
    prev_ego_v: Optional[float] = None

    ego_collision = False
    ego_offroad: Optional[bool] = None

    critical_any = False
    critical_adv = False
    near_collision_seen = False

    t_collision = spec.horizon_steps
    t_critical = spec.horizon_steps
    t_critical_adv = spec.horizon_steps

    min_d_any = float("inf")
    min_d_adv = float("inf")
    min_ttc_any = float("inf")
    min_ttc_adv = float("inf")

    spd_min = float("inf")
    spd_sum = 0.0
    acc_min = float("inf")
    hard_brake_count = 0
    near_stop_steps = 0
    stop_streak = 0

    adv_crashed_flags: List[bool] = [False] * len(advs)
    adv_crash_steps: List[Optional[int]] = [None] * len(advs)
    adv_crash_dists: List[Optional[float]] = [None] * len(advs)
    adv_adv_collision_count = 0
    max_adv_a = float("-inf")
    max_adv_jerk = float("-inf")

    termination = "other"
    th = thresholds

    for t in range(spec.horizon_steps):
        vehicles_all = list(unwrapped.road.vehicles)

        es = adapter.extract_state(ego)
        ego_x.append(es.x)
        ego_y.append(es.y)
        ego_psi.append(es.psi)
        ego_v.append(es.v)
        ev = es.v

        spd_min = min(spd_min, ev)
        spd_sum += ev

        a_ego = (ev - prev_ego_v) / dt if prev_ego_v is not None else 0.0
        acc_min = min(acc_min, a_ego)
        if float(a_ego) <= th.hard_brake_a:
            hard_brake_count += 1

        for i, adv in enumerate(advs):
            s = adapter.extract_state(adv)
            adv_x[i].append(s.x)
            adv_y[i].append(s.y)
            adv_psi[i].append(s.psi)
            adv_v[i].append(s.v)
            if adv_prev_v[i] is not None:
                a_adv = (s.v - adv_prev_v[i]) / dt  # type: ignore[operator]
                max_adv_a = max(max_adv_a, abs(a_adv))
                if adv_prev_a[i] is not None:
                    max_adv_jerk = max(max_adv_jerk, abs((a_adv - adv_prev_a[i]) / dt))  # type: ignore[operator]
                adv_prev_a[i] = a_adv
            adv_prev_v[i] = s.v

        d_any = min_dist_ego_any(ego, vehicles_all)
        if np.isfinite(d_any):
            min_d_any = min(min_d_any, d_any)

        # min distance and TTC across ALL adversaries (not just advs[0])
        d_adv = (
            min(dist_pair(ego, a) for a in advs)
            if advs else float("nan")
        )
        if np.isfinite(d_adv):
            min_d_adv = min(min_d_adv, d_adv)

        ttc_any = min_ttc_ego_any(ego, vehicles_all, dy_gate=th.dy_near_m)
        if np.isfinite(ttc_any):
            min_ttc_any = min(min_ttc_any, ttc_any)

        ttc_adv = (
            min(ttc_pair_longitudinal(ego, a, dy_gate=th.dy_near_m) for a in advs)
            if advs else float("nan")
        )
        if np.isfinite(ttc_adv):
            min_ttc_adv = min(min_ttc_adv, ttc_adv)

        ego_offroad = ego_offroad_best_effort(ego)
        prox = proximity_flag(ego, vehicles_all, min_ttc_any=ttc_any, th=th)

        ex, ey = es.x, es.y
        if not critical_any and not ego_collision:
            if ego_offroad is True:
                critical_any = True
                t_critical = t + 1

            if not critical_any and np.isfinite(ttc_any) and ttc_any < th.ttc_crit_s:
                near_collision_seen = True
                critical_any = True
                t_critical = min(t_critical, t + 1)

            if not critical_any:
                for v in vehicles_all:
                    if v is ego:
                        continue
                    dx = float(v.position[0]) - ex
                    dy = float(v.position[1]) - ey
                    if abs(dy) <= th.dy_near_m and abs(dx) <= th.dx_near_m:
                        near_collision_seen = True
                        critical_any = True
                        t_critical = min(t_critical, t + 1)
                        break

            if not critical_any:
                if ev <= th.v_stop_mps:
                    stop_streak += 1
                    near_stop_steps += 1
                else:
                    stop_streak = 0
                if stop_streak >= th.stop_streak_steps and prox and near_collision_seen:
                    critical_any = True
                    t_critical = t + 1

            if not critical_any and float(a_ego) <= th.hard_brake_a and prox and near_collision_seen:
                critical_any = True
                t_critical = t + 1

            if not critical_any and not near_collision_seen:
                if np.isfinite(d_any) and d_any <= th.critical_dist:
                    near_collision_seen = True
                    critical_any = True
                    t_critical = min(t_critical, t + 1)

        if advs and not critical_adv and not ego_collision:
            for av in advs:
                ax, ay = float(av.position[0]), float(av.position[1])
                d_a = dist_pair(ego, av)
                ttc_a = ttc_pair_longitudinal(ego, av, dy_gate=th.dy_near_m)
                if np.isfinite(ttc_a) and ttc_a < th.ttc_crit_s:
                    critical_adv = True
                    t_critical_adv = t + 1
                    break
                elif np.isfinite(d_a) and abs(ay - ey) <= th.dy_near_m and abs(ax - ex) <= th.dx_near_m:
                    critical_adv = True
                    t_critical_adv = t + 1
                    break
                elif np.isfinite(d_a) and d_a <= th.critical_dist:
                    critical_adv = True
                    t_critical_adv = t + 1
                    break

        ego_raw = _ego_obs_from_multiagent(obs) if multi_agent else obs
        ego_raw, ego_stacker = _maybe_stack(ego_raw, ego_stacker, ego_n_stack)
        ego_act = ego_policy.act(_add_batch_dim(ego_raw), deterministic=True)
        ego_action_log.append(ego_act)

        adv_acts: List[int] = []
        for i, ctrl in enumerate(adv_controllers):
            if adv_obs_extractor is not None:
                adv_obs_i = adv_obs_extractor(obs, i)
            else:
                adv_obs_i = adapter.get_kinematic_obs(unwrapped, advs[i], vehicles_all)
            adv_acts.append(int(ctrl(adv_obs_i)))
            adv_action_logs[i].append(adv_acts[-1])

        action = tuple([ego_act] + adv_acts) if multi_agent else ego_act  # type: ignore[assignment]

        obs, reward, terminated, truncated, _info = env.step(action)
        episode_return += float(reward)

        if record_frames:
            fr = env.render()
            if fr is not None:
                frames.append(fr)  # type: ignore[union-attr]

        if adapter.is_crashed(ego):
            ego_collision = True
            t_collision = min(t_collision, t + 1)
            if not critical_any:
                critical_any = True
                t_critical = min(t_critical, t + 1)
            # Ego crash is the definitive adversary-caused critical event: update
            # t_critical_adv so phi = (min_d_adv, t_collision) not (min_d_adv, 240).
            if not critical_adv and advs:
                critical_adv = True
                t_critical_adv = t + 1
            termination = "ego_collision"
            break

        for i, adv in enumerate(advs):
            if not adv_crashed_flags[i] and adapter.is_crashed(adv):
                adv_crashed_flags[i] = True
                adv_crash_steps[i] = t + 1
                adv_crash_dists[i] = float(dist_pair(ego, adv))

        if stop_on_critical and critical_any and termination == "other":
            termination = "critical_incident"
            break

        if bool(terminated or truncated):
            termination = "timeout" if (t + 1) >= spec.horizon_steps else "env_terminated"
            break

        prev_ego_v = ev

    any_adv_crashed = any(adv_crashed_flags)
    adv_nuisance = False
    if any_adv_crashed and not ego_collision:
        for i, crash_step in enumerate(adv_crash_steps):
            if adv_crashed_flags[i] and crash_step is not None:
                crash_dist = adv_crash_dists[i]
                if th.nuisance_dist is not None and crash_dist is not None:
                    if crash_dist > th.nuisance_dist:
                        adv_nuisance = True
                elif not critical_any:
                    adv_nuisance = True
                elif abs(t_critical - crash_step) > nuisance_window_steps:
                    adv_nuisance = True

    if adv_nuisance and termination in ("timeout", "env_terminated", "other"):
        termination = "adv_nuisance_crash"

    if termination in ("other", "timeout", "env_terminated"):
        if ego_collision:
            termination = "ego_collision"
        elif critical_any:
            termination = "critical_incident"

    steps = len(ego_x)

    def _fin(v: float, fallback: float = float("nan")) -> float:
        return float(v) if np.isfinite(v) else fallback

    phi = (
        (_fin(min_d_adv), float(t_critical_adv))
        if advs else None
    )

    metrics = EpisodeMetrics(
        ego_collision=bool(ego_collision),
        ego_offroad=ego_offroad,
        adv_ego_collision=bool(ego_collision and any(
            adv_crashed_flags[i] and adv_crash_dists[i] is not None and adv_crash_dists[i] <= th.nuisance_dist
            for i in range(len(advs))
        )),
        adv_adv_collision_count=int(adv_adv_collision_count),
        critical_incident=bool(critical_any),
        critical_incident_adv=bool(critical_adv),
        termination_reason=str(termination),
        time_to_collision_steps=int(t_collision),
        time_to_critical_incident_steps=int(t_critical),
        time_to_critical_incident_adv_steps=int(t_critical_adv),
        min_dist_ego_any=_fin(min_d_any),
        min_dist_ego_adv=_fin(min_d_adv),
        min_ttc_ego_any=_fin(min_ttc_any),
        min_ttc_ego_adv=_fin(min_ttc_adv),
        ego_speed_min=_fin(spd_min),
        ego_speed_mean=float(spd_sum / max(1, steps)) if steps > 0 else float("nan"),
        ego_accel_min=_fin(acc_min),
        ego_hard_brake_count=int(hard_brake_count),
        ego_near_stop_steps=int(near_stop_steps),
        adv_crashed=bool(any_adv_crashed),
        adv_nuisance_crash=bool(adv_nuisance),
        max_adv_accel=_fin(max_adv_a, fallback=0.0),
        max_adv_jerk=_fin(max_adv_jerk, fallback=0.0),
        episode_return=float(episode_return),
        steps=int(steps),
        phi=phi,
    )

    env.close()

    T = steps
    ego_trace = np.column_stack([
        np.asarray(ego_x, dtype=np.float32), np.asarray(ego_y, dtype=np.float32),
        np.asarray(ego_psi, dtype=np.float32), np.asarray(ego_v, dtype=np.float32),
    ]) if T > 0 else np.zeros((0, 4), dtype=np.float32)

    adv_traces_arr: List[np.ndarray] = []
    for i in range(len(advs)):
        if len(adv_x[i]) > 0:
            adv_traces_arr.append(np.column_stack([
                np.asarray(adv_x[i], dtype=np.float32), np.asarray(adv_y[i], dtype=np.float32),
                np.asarray(adv_psi[i], dtype=np.float32), np.asarray(adv_v[i], dtype=np.float32),
            ]))
        else:
            adv_traces_arr.append(np.zeros((0, 4), dtype=np.float32))

    ego_actions_arr = np.asarray(ego_action_log, dtype=np.int32)
    adv_actions_mat = (
        np.column_stack([np.asarray(adv_action_logs[i], dtype=np.int32) for i in range(len(advs))])
        if advs and len(adv_action_logs[0]) > 0
        else np.zeros((T, max(1, spec.n_adversaries)), dtype=np.int32)
    )

    return EpisodeResult(
        spec=spec,
        metrics=metrics,
        ego_trace=ego_trace,
        adv_traces=adv_traces_arr,
        ego_actions=ego_actions_arr,
        adv_actions=adv_actions_mat,
        frames=frames,
    )
