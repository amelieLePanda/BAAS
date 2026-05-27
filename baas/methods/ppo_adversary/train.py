"""PPO adversary training against a frozen ego policy.

Trains an adversary using PPO (stable-baselines3) inside a custom Gymnasium
wrapper that runs the frozen DQN ego internally on each step.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from baas.core.ego_policy import EgoPolicy
from baas.core.env_adapter import EnvAdapter
from baas.evaluation.run_io import update_run_status, write_run_meta
from baas.methods.ppo_adversary.config import PPOAdvConfig

logger = logging.getLogger(__name__)


def _distance(v1: Any, v2: Any) -> float:
    dx = float(v1.position[0] - v2.position[0])
    dy = float(v1.position[1] - v2.position[1])
    return float(np.hypot(dx, dy))


class AdversaryVsEgoEnv:
    """Custom Gymnasium env exposing only the adversary's obs/action space.

    Internally runs two controlled vehicles in highway-env:
      - vehicle 0: ego, driven by the frozen policy
      - vehicle 1: adversary, trained by PPO

    The env uses MultiAgentAction so both vehicles are controlled each step.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 15}

    def __init__(
        self,
        adapter: EnvAdapter,
        ego_policy: EgoPolicy,
        env_cfg: Any,
        *,
        cfg: PPOAdvConfig,
        seed: int = 0,
        render_mode: Optional[str] = None,
        critical_dist: float = 6.0,
        nuisance_dist: float = 12.0,
        r_critical: Optional[float] = None,
        r_adv_crash_close: Optional[float] = None,
        r_adv_crash_nuisance: Optional[float] = None,
    ) -> None:
        import gymnasium as gym

        self._adapter = adapter
        self._ego_policy = ego_policy
        self._cfg = cfg
        self.base_seed = int(seed)
        self.critical_dist = float(critical_dist)
        self.nuisance_dist = float(nuisance_dist)
        # Reward params: explicit kwargs override cfg (QDRL varies these across cells)
        self.r_critical = float(r_critical if r_critical is not None else cfg.r_critical)
        self.r_adv_crash_close = float(r_adv_crash_close if r_adv_crash_close is not None else cfg.r_adv_crash_close)
        self.r_adv_crash_nuisance = float(r_adv_crash_nuisance if r_adv_crash_nuisance is not None else cfg.r_adv_crash_nuisance)

        self._horizon = int(getattr(env_cfg, "horizon_steps", cfg.n_steps))

        self._env = adapter.make_env(env_cfg, n_adversaries=1, render_mode=render_mode)

        raw_act = self._env.action_space
        if hasattr(raw_act, "spaces") and len(raw_act.spaces) >= 2:
            self.action_space = raw_act.spaces[1]
        else:
            self.action_space = raw_act

        # Adversary obs is a flat kinematic vector: (n_obs_vehicles * 5,) = (25,).
        # This avoids giving the adversary grayscale frames (which its MlpPolicy cannot use).
        _N_OBS_VEHICLES = 5
        _N_FEATURES = 5
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(_N_OBS_VEHICLES * _N_FEATURES,),
            dtype=np.float32,
        )

        self._t = 0
        self._critical_triggered = False
        self._last_ego_obs: Optional[np.ndarray] = None

    def _get_adv_obs(self) -> np.ndarray:
        """Extract kinematic observation for the adversary vehicle."""
        unwrapped = self._env.unwrapped
        controlled = list(getattr(unwrapped, "controlled_vehicles", []) or [])
        adv_v = controlled[1] if len(controlled) > 1 else None
        vehicles_all = list(getattr(getattr(unwrapped, "road", None), "vehicles", []) or [])
        if adv_v is None:
            return np.zeros(25, dtype=np.float32)
        return self._adapter.get_kinematic_obs(unwrapped, adv_v, vehicles_all).flatten()

    def _ego_obs_from(self, raw_obs: Any) -> np.ndarray:
        """Extract ego grayscale obs from MultiAgentObservation tuple."""
        if isinstance(raw_obs, (tuple, list)) and len(raw_obs) >= 1:
            return np.array(raw_obs[0], dtype=np.float32)
        return np.array(raw_obs, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._t = 0
        self._critical_triggered = False
        s = self.base_seed if seed is None else int(seed)
        raw_obs, info = self._env.reset(seed=s)
        self._last_ego_obs = self._ego_obs_from(raw_obs)
        adv_obs = self._get_adv_obs()
        return adv_obs, info

    def step(self, action_adv: int):
        if self._last_ego_obs is None:
            raise RuntimeError("Call reset() before step().")

        ego_in = self._last_ego_obs
        if ego_in.ndim < 4:
            ego_in = ego_in[None, ...]
        ego_act = self._ego_policy.act(ego_in, deterministic=True)

        raw_obs, _env_reward, terminated, truncated, info = self._env.step(
            (int(ego_act), int(action_adv))
        )
        self._t += 1

        self._last_ego_obs = self._ego_obs_from(raw_obs)
        adv_obs = self._get_adv_obs()

        unwrapped = self._env.unwrapped
        controlled = list(getattr(unwrapped, "controlled_vehicles", []) or [])
        ego_v = controlled[0] if len(controlled) > 0 else getattr(unwrapped, "vehicle", None)
        adv_v = controlled[1] if len(controlled) > 1 else None

        ego_crashed = bool(getattr(ego_v, "crashed", False))
        adv_crashed = bool(getattr(adv_v, "crashed", False)) if adv_v is not None else False
        dist = _distance(ego_v, adv_v) if (ego_v is not None and adv_v is not None) else 9999.0
        proximity = 1.0 / (1.0 + dist)
        adv_nuisance = bool(adv_crashed and not ego_crashed and dist > self.nuisance_dist)

        r = self._cfg.w_proximity * proximity - self._cfg.c_time_step

        critical = bool(ego_crashed) or (dist <= self.critical_dist)
        if critical and not self._critical_triggered:
            r += self.r_critical
            self._critical_triggered = True

        done = bool(terminated or truncated)

        if ego_crashed:
            if self._t < self._cfg.min_ttf_steps:
                r += 1.0 - 5.0 * (1.0 - self._t / max(1, self._cfg.min_ttf_steps))
            else:
                r += 10.0
            done = True

        if adv_crashed and not ego_crashed:
            r += self.r_adv_crash_nuisance if adv_nuisance else self.r_adv_crash_close
            done = True

        if self._t >= self._horizon:
            done = True

        info = dict(info) if isinstance(info, dict) else {}
        info.update({
            "ego_crashed": ego_crashed,
            "adv_crashed": adv_crashed,
            "adv_nuisance_crash": adv_nuisance,
            "critical_incident": critical,
            "dist_ego_adv": float(dist),
            "t": self._t,
        })

        return adv_obs.astype(np.float32), float(r), bool(done), False, info

    def render(self):
        return self._env.render()

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


def train_ppo_adversary(
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    cfg: PPOAdvConfig,
    env_cfg: Any,
    *,
    output_dir: Path,
    seed: int = 0,
    n_adversaries: int = 1,
) -> Path:
    """Train a PPO adversary and save the checkpoint.

    The ego policy is frozen throughout. Returns path to the saved .zip checkpoint.

    Saves:
      output_dir/run.json              -- metadata written at start; updated on completion
      output_dir/ppo_adv_final.zip     -- final model
      output_dir/best/best_model.zip   -- checkpoint with highest mean reward (via EvalCallback)
      output_dir/checkpoints/          -- sparse periodic saves (~4 evenly spaced)
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_run_meta(
        output_dir,
        method="ppo_adversary",
        config=cfg,
        seed=seed,
        n_adversaries=n_adversaries,
    )

    critical_dist = float(getattr(env_cfg, "critical_dist", 6.0))
    nuisance_dist = float(getattr(env_cfg, "nuisance_dist", 12.0))
    horizon = int(getattr(env_cfg, "horizon_steps", cfg.n_steps))

    def _make_env():
        return AdversaryVsEgoEnv(
            adapter=adapter,
            ego_policy=ego_policy,
            env_cfg=env_cfg,
            cfg=cfg,
            seed=seed,
            critical_dist=critical_dist,
            nuisance_dist=nuisance_dist,
        )

    venv = make_vec_env(_make_env, n_envs=1, seed=int(seed))
    eval_venv = make_vec_env(_make_env, n_envs=1, seed=int(seed) + 999)

    # Save ~4 evenly spaced checkpoints rather than one every 10k steps.
    ckpt_freq = max(horizon, cfg.total_timesteps // 4)
    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_adv",
        save_replay_buffer=False,
        verbose=0,
    )

    # Track the checkpoint with the highest mean reward automatically.
    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "logs"),
        eval_freq=max(horizon * 5, cfg.total_timesteps // 10),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    model = PPO(
        policy="MlpPolicy",
        env=venv,
        learning_rate=cfg.lr,
        n_steps=horizon,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        clip_range=cfg.clip_range,
        verbose=1,
        seed=int(seed),
        device=cfg.device,
    )

    logger.info(
        "Training PPO adversary: %d steps  horizon=%d  n_adv=%d",
        cfg.total_timesteps, horizon, cfg.n_adversaries,
    )
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList([checkpoint_cb, eval_cb]),
    )

    final_path = output_dir / "ppo_adv_final.zip"
    model.save(str(final_path))
    venv.close()
    eval_venv.close()

    best_path = output_dir / "best" / "best_model.zip"
    update_run_status(
        output_dir,
        status="complete",
        extra={
            "final_model": str(final_path),
            "best_model": str(best_path) if best_path.exists() else None,
            "total_timesteps": cfg.total_timesteps,
        },
    )
    logger.info("PPO adversary saved to %s  (best: %s)", final_path, best_path)
    return final_path
