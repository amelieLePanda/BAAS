"""Train the ego DQN policy with a CNN on stacked grayscale frames.

The trained checkpoint is frozen and used as the ego policy in all adversarial
search experiments.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from baas.training.config import GRAYSCALE_OBS_CFG, EgoTrainConfig

logger = logging.getLogger(__name__)


def _make_reseed_wrapper(env: Any, base_seed: int) -> Any:
    """Wrap an env to reseed on every reset, used for the eval env."""
    import gymnasium as gym

    class _W(gym.Wrapper):
        def __init__(self, e: Any, seed: int) -> None:
            super().__init__(e)
            self.base_seed = int(seed)
            self.episode_idx = 0

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            s = self.base_seed + self.episode_idx
            self.episode_idx += 1
            return self.env.reset(seed=s, options=options)

    return _W(env, base_seed)


class _RewardComponentLogger:
    """SB3 callback that logs per-episode returns."""

    def __init__(self, verbose: int = 0) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Cb(BaseCallback):
            def __init__(self_cb) -> None:
                super().__init__(verbose=verbose)
                self_cb._ep_return: Dict[int, float] = {}
                self_cb._ep_len: Dict[int, int] = {}

            def _on_step(self_cb) -> bool:
                infos = self_cb.locals.get("infos")
                dones = self_cb.locals.get("dones")
                rewards = self_cb.locals.get("rewards")
                if infos is None or dones is None or rewards is None:
                    return True
                for i, info in enumerate(infos if isinstance(infos, list) else [infos]):
                    done = bool(dones[i]) if hasattr(dones, "__len__") else bool(dones)
                    r = rewards[i] if hasattr(rewards, "__len__") else float(rewards)
                    self_cb._ep_return.setdefault(i, 0.0)
                    self_cb._ep_len.setdefault(i, 0)
                    self_cb._ep_return[i] += float(r)
                    self_cb._ep_len[i] += 1
                    if done:
                        self_cb.logger.record("train/episode_return", self_cb._ep_return[i])
                        self_cb.logger.record("train/episode_length", self_cb._ep_len[i])
                        self_cb._ep_return[i] = 0.0
                        self_cb._ep_len[i] = 0
                return True

        self._cb_class = _Cb

    def build(self) -> Any:
        return self._cb_class()


def train_ego_dqn(cfg: EgoTrainConfig, output_dir: Path, *, seed: int = 0) -> Path:
    """Train a DQN ego policy and save the SB3 checkpoint.

    Uses CnnPolicy on 4-frame stacked grayscale observations (128x64).
    Returns path to the saved .zip checkpoint.
    """
    import gymnasium as gym
    import highway_env  # noqa: F401
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.utils import get_linear_fn
    from stable_baselines3.common.vec_env import DummyVecEnv

    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    tb_dir = output_dir / "tb"
    eval_log_dir = output_dir / "eval"
    for d in (ckpt_dir, tb_dir, eval_log_dir):
        d.mkdir(parents=True, exist_ok=True)

    duration_s = float(cfg.horizon_steps) / float(cfg.policy_frequency)

    env_cfg: Dict[str, Any] = {
        "observation": GRAYSCALE_OBS_CFG,
        "action": {"type": "DiscreteMetaAction"},
        "collision_reward": cfg.collision_reward,
        "lane_change_reward": cfg.lane_change_reward,
        "high_speed_reward": cfg.high_speed_reward,
        "right_lane_reward": cfg.right_lane_reward,
        "policy_frequency": cfg.policy_frequency,
        "simulation_frequency": cfg.simulation_frequency,
        "duration": duration_s,
        "controlled_vehicles": 1,
        "vehicles_count": 1 + cfg.background_traffic,
    }

    def _make_train_env(i: int):
        def _thunk():
            env = gym.make(cfg.env_id, config=env_cfg)
            env.reset(seed=int(seed) + i)
            return env
        return _thunk

    def _make_eval_env():
        def _thunk():
            env = gym.make(cfg.env_id, config=env_cfg)
            return _make_reseed_wrapper(env, base_seed=int(seed) + 10_000)
        return _thunk

    train_env = DummyVecEnv([_make_train_env(0)])
    eval_env = DummyVecEnv([_make_eval_env()])

    if cfg.lr_schedule == "linear":
        lr = get_linear_fn(start=float(cfg.learning_rate), end=0.0, end_fraction=1.0)
    else:
        lr = float(cfg.learning_rate)

    model = DQN(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=lr,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        target_update_interval=cfg.target_update_interval,
        exploration_fraction=cfg.exploration_fraction,
        exploration_initial_eps=cfg.exploration_initial_eps,
        exploration_final_eps=cfg.exploration_final_eps,
        verbose=1,
        tensorboard_log=str(tb_dir),
        seed=int(seed),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(ckpt_dir),
        name_prefix="ego_dqn",
        save_replay_buffer=False,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir),
        log_path=str(eval_log_dir),
        eval_freq=25_000,
        deterministic=True,
        render=False,
    )
    reward_cb = _RewardComponentLogger(verbose=0).build()

    logger.info(
        "Training DQN ego: %d steps  policy_freq=%d  horizon=%d",
        cfg.total_timesteps, cfg.policy_frequency, cfg.horizon_steps,
    )
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[checkpoint_cb, eval_cb, reward_cb],
    )

    final_path = ckpt_dir / "ego_dqn_final.zip"
    model.save(str(final_path))
    logger.info("Ego DQN saved to %s", final_path)
    return final_path
