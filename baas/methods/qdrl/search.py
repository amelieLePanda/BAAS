"""QD-RL: MAP-Elites search over PPO reward shaping parameters.

Each archive cell holds a distinct adversary policy trained with a different
combination of reward shaping values. The solution space is:

    [r_critical, r_adv_crash_close, r_adv_crash_nuisance]

MAP-Elites evolves these values using CMA-ME (EvolutionStrategyEmitter from
pyribs). For each candidate, a PPO adversary is trained for burst_steps
timesteps inside AdversaryVsEgoEnv with the given reward params, then
evaluated on a subset of rollout specs to get fitness and
phi = (min_dist_ego_adv / dist_max, TTCI_adv / horizon)  in [0,1]^2.

The archive uses a 10x10 normalised grid identical to MAP-Elites, so
Table 3 (archive-level diversity) is a like-for-like comparison.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baas.core.env_adapter import EnvAdapter
from baas.core.ego_policy import EgoPolicy
from baas.core.metrics import IncidentThresholds
from baas.core.rollout import run_episode
from baas.core.types import EpisodeResult, RolloutSpec
from baas.evaluation.run_io import update_run_status, write_run_meta
from baas.methods.qdrl.config import QDRLConfig

logger = logging.getLogger(__name__)

_DIST_MAX_DEFAULT = 60.0  # metres — matches MAP-Elites and benchmark_v1.yaml diversity.dist_max


def _score(result: EpisodeResult) -> float:
    """Objective to maximise.

    Matches paper Eq. (6) and MAP-Elites _fitness exactly:
      +1.0 for critical incident, +0.5 for ego collision, -0.5 for nuisance crash.
    Uses adv_nuisance_crash flag (set by run_episode) rather than an inline
    distance check, keeping the definition identical to MAP-Elites.
    """
    m = result.metrics
    f = 0.0
    if m.critical_incident:
        f += 1.0
    if m.ego_collision:
        f += 0.5
    if m.adv_nuisance_crash:
        f -= 0.5
    return f


def _measures(
    result: EpisodeResult,
    *,
    horizon: int,
    dist_max: float = _DIST_MAX_DEFAULT,
) -> Tuple[float, float]:
    """Behaviour descriptor phi in [0,1]^2 — identical to MAP-Elites _behaviour_descriptor.

    Uses time_to_critical_incident_adv_steps (TTCI_adv), not TTCI_any, so the
    archive axes match MAP-Elites and Table 3 is a valid comparison.
    """
    m = result.metrics
    d = float(m.min_dist_ego_adv) if np.isfinite(m.min_dist_ego_adv) else dist_max
    ttci = float(m.time_to_critical_incident_adv_steps)
    d_norm = float(np.clip(d, 0.0, dist_max) / dist_max)
    ttci_norm = float(np.clip(ttci, 1.0, horizon) / horizon)
    return d_norm, ttci_norm


def _train_burst(
    *,
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    env_cfg: Any,
    cfg: QDRLConfig,
    r_critical: float,
    r_adv_close: float,
    r_adv_nuis: float,
    burst_steps: int,
    seed: int,
    out_path: Path,
) -> Path:
    """Train a PPO adversary for burst_steps and save to out_path."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    from baas.methods.ppo_adversary.config import PPOAdvConfig
    from baas.methods.ppo_adversary.train import AdversaryVsEgoEnv

    ppo_cfg = PPOAdvConfig(
        n_adversaries=cfg.n_adversaries,
        total_timesteps=burst_steps,
        lr=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        clip_range=cfg.clip_range,
        batch_size=cfg.ppo_batch_size,
        seed=seed,
    )

    critical_dist = float(getattr(env_cfg, "critical_dist", 6.0))
    nuisance_dist = float(getattr(env_cfg, "nuisance_dist", 12.0))

    def _make():
        return AdversaryVsEgoEnv(
            adapter=adapter,
            ego_policy=ego_policy,
            env_cfg=env_cfg,
            cfg=ppo_cfg,
            seed=seed,
            critical_dist=critical_dist,
            nuisance_dist=nuisance_dist,
            r_critical=r_critical,
            r_adv_crash_close=r_adv_close,
            r_adv_crash_nuisance=r_adv_nuis,
        )

    venv = make_vec_env(_make, n_envs=int(cfg.n_envs), seed=int(seed))

    horizon = int(getattr(env_cfg, "horizon_steps", 240))
    model = PPO(
        policy="MlpPolicy",
        env=venv,
        learning_rate=cfg.lr,
        n_steps=max(cfg.ppo_n_steps, horizon),
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        clip_range=cfg.clip_range,
        verbose=0,
        device=cfg.device,
        seed=int(seed),
    )
    model.learn(total_timesteps=int(burst_steps))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    venv.close()
    return out_path


def _eval_policy(
    *,
    policy_path: Path,
    specs: List[RolloutSpec],
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    thresholds: IncidentThresholds,
    env_cfg: Any,
    cfg: QDRLConfig,
    rng: np.random.Generator,
    dist_max: float = _DIST_MAX_DEFAULT,
) -> Dict[str, Any]:
    """Run a subset of rollout specs with the given PPO adversary.

    Returns objective, phi measures (normalised [0,1]^2), and a summary dict.
    """
    from stable_baselines3 import PPO

    ppo = PPO.load(str(policy_path), device=cfg.device)
    horizon = int(getattr(env_cfg, "horizon_steps", 240))

    def _adv_ctrl(obs: Any) -> int:
        act, _ = ppo.predict(np.asarray(obs, dtype=np.float32).flatten(), deterministic=True)
        return int(np.asarray(act).reshape(-1)[0])

    n = min(int(cfg.eval_rollouts), len(specs))
    idxs = rng.choice(len(specs), size=n, replace=False)

    scores: List[float] = []
    measures: List[Tuple[float, float]] = []
    p_critical = 0.0
    p_collision = 0.0

    for idx in idxs:
        spec = specs[int(idx)]
        result = run_episode(
            spec=spec,
            adapter=adapter,
            ego_policy=ego_policy,
            adv_controllers=[_adv_ctrl],
            thresholds=thresholds,
            env_cfg=env_cfg,
            stop_on_critical=False,
        )
        s = _score(result)
        d_norm, ttci_norm = _measures(result, horizon=horizon, dist_max=dist_max)
        scores.append(s)
        measures.append((d_norm, ttci_norm))
        p_critical += 1.0 if result.metrics.critical_incident else 0.0
        p_collision += 1.0 if result.metrics.ego_collision else 0.0

    n_actual = max(1, len(scores))
    mean_score = float(np.mean(scores)) if scores else float("-inf")
    mean_d = float(np.mean([m[0] for m in measures])) if measures else 1.0
    mean_t = float(np.mean([m[1] for m in measures])) if measures else 1.0

    return {
        "objective": mean_score,
        "measures": (mean_d, mean_t),
        "summary": {
            "n": n_actual,
            "mean_score": mean_score,
            "p_critical": p_critical / n_actual,
            "p_collision": p_collision / n_actual,
            "mean_d_norm": mean_d,
            "mean_ttci_adv_norm": mean_t,
        },
    }


def _export_archive(
    *,
    archive: Any,
    out_path: Path,
    meta: Dict[bytes, Dict[str, Any]],
    dims: List[int],
    specs: List[RolloutSpec],
) -> None:
    """Write archive to JSON. Ranges are normalised [0,1]^2 on both axes."""
    from dataclasses import asdict

    sols = archive.data("solution")
    objs = archive.data("objective")
    meas = archive.data("measures")
    n = int(sols.shape[0]) if hasattr(sols, "shape") else len(sols)

    cells = []
    for k in range(n):
        sol = np.asarray(sols[k], dtype=np.float32)
        m = meta.get(sol.tobytes(), {})
        cells.append({
            "objective": float(objs[k]),
            "measures": [float(meas[k, 0]), float(meas[k, 1])],
            "solution": sol.tolist(),
            "policy_path": m.get("policy_path"),
            "summary": m.get("summary"),
            "train_cfg": m.get("train_cfg"),
        })

    payload = {
        "method": "qdrl",
        "archive": {
            "dims": dims,
            # Both axes are normalised to [0,1]: d_norm = min_dist/60, ttci_adv_norm = TTCI_adv/240
            # Identical grid to MAP-Elites — Table 3 archive comparison is valid.
            "ranges": {
                "d_norm": [0.0, 1.0],
                "ttci_adv_norm": [0.0, 1.0],
            },
            "occupied": int(len(archive)),
            "cells": cells,
        },
        "rollout_specs": [asdict(s) for s in specs],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("QD-RL archive written to %s  (%d cells)", out_path, len(cells))


def run_qdrl(
    specs: List[RolloutSpec],
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    thresholds: IncidentThresholds,
    cfg: QDRLConfig,
    *,
    output_dir: Path,
    seed: int = 0,
    env_cfg: Any = None,
    dist_max: float = _DIST_MAX_DEFAULT,
) -> None:
    """Run QD-RL and write the final archive to output_dir.

    Each MAP-Elites iteration:
      1. Ask pyribs for a batch of reward shaping candidates.
      2. For each candidate, train a burst PPO adversary with those params.
      3. Evaluate the trained policy on a rollout-spec subset.
      4. Tell pyribs the fitness and phi measures (normalised [0,1]^2).

    The archive uses ranges [(0.0, 1.0), (0.0, 1.0)] — same as MAP-Elites —
    so archive-level diversity (Table 3) is directly comparable.
    Writes qdrl_archive.json and periodic snapshots.
    """
    try:
        from ribs.archives import GridArchive
        from ribs.emitters import EvolutionStrategyEmitter
        from ribs.schedulers import Scheduler
    except ImportError as exc:
        raise ImportError("pyribs is required: pip install ribs") from exc

    output_dir = Path(output_dir)
    policies_dir = output_dir / "policies"
    policies_dir.mkdir(parents=True, exist_ok=True)

    write_run_meta(
        output_dir,
        method="qdrl",
        config=cfg,
        seed=seed,
        n_adversaries=cfg.n_adversaries,
    )

    horizon = int(getattr(env_cfg, "horizon_steps", 240)) if env_cfg else 240

    dims = list(cfg.archive_grid_dims)

    rc_lo, rc_hi = float(cfg.r_critical_range[0]), float(cfg.r_critical_range[1])
    rcl_lo, rcl_hi = float(cfg.r_adv_close_range[0]), float(cfg.r_adv_close_range[1])
    rn_lo, rn_hi = float(cfg.r_adv_nuis_range[0]), float(cfg.r_adv_nuis_range[1])
    sol_lo = np.array([rc_lo, rcl_lo, rn_lo], dtype=np.float32)
    sol_hi = np.array([rc_hi, rcl_hi, rn_hi], dtype=np.float32)

    # Archive ranges are normalised [0,1]^2, identical to MAP-Elites.
    # dist_range in config is unused for archive indexing; dist_max controls normalisation.
    archive = GridArchive(
        solution_dim=3,
        dims=dims,
        ranges=[(0.0, 1.0), (0.0, 1.0)],
        qd_score_offset=0.0,
    )
    emitters = [
        EvolutionStrategyEmitter(
            archive=archive,
            x0=(sol_lo + sol_hi) / 2.0,
            sigma0=float(cfg.sigma0),
            batch_size=int(cfg.emitter_batch_size),
            lower_bounds=sol_lo,
            upper_bounds=sol_hi,
        )
        for _ in range(int(cfg.n_emitters))
    ]
    sched = Scheduler(archive, emitters)

    rng = np.random.default_rng(int(seed) + 12345)
    meta: Dict[bytes, Dict[str, Any]] = {}
    t0 = time.time()

    for it in range(int(cfg.n_iters)):
        sols = sched.ask()
        objs: List[float] = []
        meas: List[np.ndarray] = []

        for j, sol in enumerate(sols):
            sol = np.asarray(sol, dtype=np.float32)
            r_critical, r_adv_close, r_adv_nuis = float(sol[0]), float(sol[1]), float(sol[2])

            cand_seed = int(seed) + it * 10_000 + j * 100
            policy_path = policies_dir / f"it{it:04d}_j{j:02d}.zip"

            _train_burst(
                adapter=adapter,
                ego_policy=ego_policy,
                env_cfg=env_cfg,
                cfg=cfg,
                r_critical=r_critical,
                r_adv_close=r_adv_close,
                r_adv_nuis=r_adv_nuis,
                burst_steps=int(cfg.burst_steps),
                seed=cand_seed,
                out_path=policy_path,
            )

            ev = _eval_policy(
                policy_path=policy_path,
                specs=specs,
                adapter=adapter,
                ego_policy=ego_policy,
                thresholds=thresholds,
                env_cfg=env_cfg,
                cfg=cfg,
                rng=rng,
                dist_max=dist_max,
            )

            objs.append(float(ev["objective"]))
            meas.append(np.array(ev["measures"], dtype=np.float32))

            sol_key = sol.tobytes()
            prev = meta.get(sol_key)
            if prev is None or ev["objective"] > float(prev.get("objective", -1e18)):
                meta[sol_key] = {
                    "objective": ev["objective"],
                    "policy_path": str(policy_path),
                    "summary": ev["summary"],
                    "train_cfg": {
                        "r_critical": r_critical,
                        "r_adv_crash_close": r_adv_close,
                        "r_adv_crash_nuisance": r_adv_nuis,
                        "burst_steps": cfg.burst_steps,
                        "seed": cand_seed,
                    },
                }

        sched.tell(objs, meas)

        # Remove burst-trained policies that did not make it into the archive.
        # Only the policies referenced by current archive members are kept.
        live_policy_paths = {
            m.get("policy_path")
            for m in meta.values()
            if m.get("policy_path") is not None
        }
        for policy_file in list(policies_dir.glob("*.zip")):
            if str(policy_file) not in live_policy_paths:
                try:
                    policy_file.unlink()
                except OSError:
                    pass

        elapsed = time.time() - t0
        logger.info(
            "QD-RL it=%d/%d  occupied=%d  policies_kept=%d  elapsed_min=%.1f",
            it + 1, cfg.n_iters, len(archive), len(live_policy_paths), elapsed / 60,
        )

        if cfg.snapshot_every > 0 and ((it == 0) or ((it + 1) % cfg.snapshot_every == 0)):
            snap = output_dir / f"qdrl_archive_it{it:04d}.json"
            _export_archive(
                archive=archive, out_path=snap, meta=meta,
                dims=dims, specs=specs,
            )

    _export_archive(
        archive=archive,
        out_path=output_dir / "qdrl_archive.json",
        meta=meta,
        dims=dims,
        specs=specs,
    )
    update_run_status(
        output_dir,
        status="complete",
        extra={"archive_occupied": int(len(archive)), "n_iters": cfg.n_iters},
    )
