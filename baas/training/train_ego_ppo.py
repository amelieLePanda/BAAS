"""Train the ego PPO policy: an alternative to DQN.

Same observation config (stacked grayscale CNN) and timing settings as the DQN
ego, so both checkpoints are evaluated on equal footing.
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


def train_ego_ppo(cfg: EgoTrainConfig, output_dir: Path, *, seed: int = 0) -> Path:
    """Train a PPO ego policy and save the SB3 checkpoint.

    Uses CnnPolicy on 4-frame stacked grayscale observations (128x64), identical
    to the DQN ego setup so both policies can be benchmarked side-by-side.
    Returns path to the saved .zip checkpoint.
    """
    import gymnasium as gym
    import highway_env  # noqa: F401
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
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

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(tb_dir),
        seed=int(seed),
        device=cfg.device,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(ckpt_dir),
        name_prefix="ego_ppo",
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

    logger.info(
        "Training PPO ego: %d steps  policy_freq=%d  horizon=%d",
        cfg.total_timesteps, cfg.policy_frequency, cfg.horizon_steps,
    )
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
    )

    final_path = ckpt_dir / "ego_ppo_final.zip"
    model.save(str(final_path))
    logger.info("Ego PPO saved to %s", final_path)
    return final_path
