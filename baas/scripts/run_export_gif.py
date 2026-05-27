"""Export a failure episode from the ego policy as a GIF.

Tries seeds sequentially until a crash is found, then saves the frames.
Also saves a slow version (half fps) for presentations.

Usage:
    python baas/scripts/run_export_gif.py --config configs/benchmark_v1.yaml --ego-policy checkpoints/ego_dqn_v3/checkpoints/ego_dqn_final.zip --output gifs/ego_dqn_v3_failure.gif

    # Save a clean (no-crash) episode too:
    python baas/scripts/run_export_gif.py --config configs/benchmark_v1.yaml --ego-policy checkpoints/ego_dqn_v3/checkpoints/ego_dqn_final.zip --output gifs/ego_dqn_v3_failure.gif --also-save-success
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _run_episode(model, env, seed: int, freeze_frames_after_crash: int = 8):
    """Run one episode, return (crashed, frames, steps).

    Stops recording immediately after a crash, then appends a few frozen
    frames of the crash moment so the impact is visible on screen.
    """
    obs, _ = env.reset(seed=seed)
    frames = []
    done = False
    crashed = False
    steps = 0

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action, _ = model.predict(obs[None, ...] if obs.ndim == 3 else obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(np.asarray(action).flat[0]))
        steps += 1
        done = terminated or truncated
        if info.get("crashed", False):
            crashed = True
            crash_frame = env.render()
            if crash_frame is not None:
                frames.extend([crash_frame] * freeze_frames_after_crash)
            break

    if not crashed:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    return crashed, frames, steps


def _save_gif(frames, path: Path, fps: int) -> None:
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps)
    logger.info("Saved %s  (%d frames @ %d fps)", path, len(frames), fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ego failure/success episode as GIF")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ego-policy", required=True, type=Path)
    parser.add_argument("--policy-type", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--output", required=True, type=Path, help="Path for the failure GIF")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--slow-fps", type=int, default=5, help="FPS for the slow version")
    parser.add_argument("--max-seeds", type=int, default=200)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--also-save-success", action="store_true")
    parser.add_argument("--background-traffic", type=int, default=None)
    args = parser.parse_args()

    import gymnasium as gym
    import highway_env  # noqa: F401

    from baas.evaluation.config_loader import load_config
    from baas.training.config import GRAYSCALE_OBS_CFG

    bundle = load_config(args.config)
    ego_cfg = bundle.ego_training
    duration_s = float(ego_cfg.horizon_steps) / float(ego_cfg.policy_frequency)

    env_config = {
        "observation": GRAYSCALE_OBS_CFG,
        "action": {"type": "DiscreteMetaAction"},
        "collision_reward": ego_cfg.collision_reward,
        "lane_change_reward": ego_cfg.lane_change_reward,
        "high_speed_reward": ego_cfg.high_speed_reward,
        "right_lane_reward": ego_cfg.right_lane_reward,
        "policy_frequency": ego_cfg.policy_frequency,
        "simulation_frequency": ego_cfg.simulation_frequency,
        "duration": duration_s,
        "controlled_vehicles": 1,
        "vehicles_count": 1 + (args.background_traffic if args.background_traffic is not None else ego_cfg.background_traffic),
    }

    if args.policy_type == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(str(args.ego_policy))
    else:
        from stable_baselines3 import PPO
        model = PPO.load(str(args.ego_policy))

    env = gym.make(ego_cfg.env_id, config=env_config, render_mode="rgb_array")

    failure_frames = None
    success_frames = None

    for seed in range(args.start_seed, args.start_seed + args.max_seeds):
        crashed, frames, steps = _run_episode(model, env, seed)

        if crashed and failure_frames is None:
            logger.info("Crash found at seed=%d  steps=%d", seed, steps)
            failure_frames = frames

        if not crashed and success_frames is None and args.also_save_success:
            logger.info("Clean episode found at seed=%d  steps=%d", seed, steps)
            success_frames = frames

        if failure_frames is not None and (not args.also_save_success or success_frames is not None):
            break
    else:
        logger.warning("No crash found in %d seeds.", args.max_seeds)

    env.close()

    if failure_frames:
        out = Path(args.output)
        _save_gif(failure_frames, out, args.fps)
        _save_gif(failure_frames, out.with_stem(out.stem + "_slow"), args.slow_fps)
    else:
        logger.error("Could not find a crash episode — try increasing --max-seeds")

    if success_frames:
        out = Path(args.output)
        success_out = out.with_stem(out.stem.replace("failure", "success") + "_success") \
            if "failure" in out.stem else out.with_stem(out.stem + "_success")
        _save_gif(success_frames, success_out, args.fps)
        _save_gif(success_frames, success_out.with_stem(success_out.stem + "_slow"), args.slow_fps)


if __name__ == "__main__":
    main()
