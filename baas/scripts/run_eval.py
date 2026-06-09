"""CLI entry point for evaluating a saved adversary artefact.

Loads an artefact from any search method, runs it against the benchmark
rollout specs, and writes a results JSON that run_benchmark.py can read.

Usage:
    python baas/scripts/run_eval.py \\
        --config configs/benchmark_v1.yaml \\
        --method map_elites \\
        --artefact runs/map_elites/run_000/archive.json \\
        --specs runs/map_elites/run_000/rollout_specs.json \\
        --ego-policy checkpoints/ego_dqn.zip \\
        --output runs/map_elites/run_000/results.json

    python baas/scripts/run_eval.py ... --feasibility-reruns 10
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ["map_elites", "ppo_adversary", "mappo_adversary", "mappo_adversary_lidar", "maddpg_adversary", "qdrl", "king_light", "cma_es", "novelty_search"]


def _make_controllers_map_elites(
    artefact_path: Path, cfg: dict, n_adversaries: int,
) -> Callable:
    from baas.evaluation.controllers import make_controllers_map_elites
    return make_controllers_map_elites(artefact_path, cfg, n_adversaries)


def _make_controllers_ppo(artefact_path: Path, device: str = "cpu") -> Callable:
    from baas.evaluation.controllers import make_controllers_ppo
    return make_controllers_ppo(artefact_path, device)


def _make_controllers_mappo(artefact_path: Path, n_adversaries: int, device: str = "cpu") -> Callable:
    """Load a MAPPO checkpoint and build N coordinated adversary controllers.

    The MAPPO model takes concatenated (N*25,) obs and outputs MultiDiscrete
    actions. Controllers are wired to share a joint obs buffer: the last
    controller (index N-1) triggers the model predict; earlier controllers
    return the action stored from the previous step (1-step lag, negligible).
    """
    from stable_baselines3 import PPO

    model = PPO.load(str(artefact_path), device=device)
    n = n_adversaries

    def make_controllers(spec: Any) -> List[Any]:
        shared: dict = {
            "obs": [np.zeros(25, dtype=np.float32)] * n,
            "acts": [1] * n,  # default IDLE
        }

        def make_ctrl(i: int) -> Any:
            def ctrl(obs: Any) -> int:
                shared["obs"][i] = np.asarray(obs, dtype=np.float32).flatten()[:25]
                if i == n - 1:
                    joint = np.concatenate(shared["obs"])
                    acts, _ = model.predict(joint, deterministic=True)
                    shared["acts"] = list(np.asarray(acts).flatten().astype(int))
                return int(shared["acts"][i])
            return ctrl

        return [make_ctrl(i) for i in range(n)]

    return make_controllers


def _make_controllers_mappo_lidar(artefact_path: Path, n_adversaries: int, device: str = "cpu") -> Callable:
    """Load a lidar MAPPO checkpoint and build N coordinated adversary controllers.

    Identical to _make_controllers_mappo but uses 128-dim lidar obs per adversary
    (shape N*128 joint obs) instead of 25-dim kinematic obs.  Must be used with
    adv_obs_extractor so run_episode passes lidar obs from the env tuple.
    """
    from stable_baselines3 import PPO

    model = PPO.load(str(artefact_path), device=device)
    n = n_adversaries
    _LIDAR_DIM = 128  # LIDAR_OBS_DIM = 64 cells * 2 features

    def make_controllers(spec: Any) -> List[Any]:
        shared: dict = {
            "obs": [np.zeros(_LIDAR_DIM, dtype=np.float32)] * n,
            "acts": [1] * n,
        }

        def make_ctrl(i: int) -> Any:
            def ctrl(obs: Any) -> int:
                shared["obs"][i] = np.asarray(obs, dtype=np.float32).flatten()[:_LIDAR_DIM]
                if i == n - 1:
                    joint = np.concatenate(shared["obs"])
                    acts, _ = model.predict(joint, deterministic=True)
                    shared["acts"] = list(np.asarray(acts).flatten().astype(int))
                return int(shared["acts"][i])
            return ctrl

        return [make_ctrl(i) for i in range(n)]

    return make_controllers


def _make_controllers_maddpg(artefact_path: Path, n_adversaries: int, device: str = "cpu") -> Callable:
    """Load a MADDPG .pt checkpoint and build N decentralised controllers.

    Each Actor_i acts on its local obs independently - no shared joint buffer needed.
    """
    import torch
    from baas.methods.maddpg_adversary.networks import Actor

    ckpt = torch.load(str(artefact_path), map_location=device)
    n = n_adversaries
    obs_dim    = ckpt.get("obs_dim", 25)
    n_actions  = ckpt.get("n_actions", 5)
    hidden_dim = ckpt.get("hidden_dim", 64)

    actors = []
    for i in range(n):
        a = Actor(obs_dim, n_actions, hidden_dim)
        a.load_state_dict(ckpt["actors"][i])
        a.eval()
        actors.append(a)

    def make_controllers(spec: Any) -> List[Any]:
        def make_ctrl(i: int) -> Any:
            def ctrl(obs: Any) -> int:
                obs_t = torch.tensor(
                    np.asarray(obs, dtype=np.float32).flatten()[:obs_dim],
                    dtype=torch.float32,
                ).unsqueeze(0)
                with torch.no_grad():
                    return int(actors[i](obs_t).argmax(dim=-1).item())
            return ctrl
        return [make_ctrl(i) for i in range(n)]

    return make_controllers


def _make_controllers_action_seq(artefact_path: Path, n_adversaries: int) -> Callable:
    from baas.evaluation.controllers import make_controllers_action_seq
    return make_controllers_action_seq(artefact_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved adversary artefact")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=SUPPORTED_METHODS)
    parser.add_argument("--artefact", required=True, type=Path)
    parser.add_argument("--specs", required=True, type=Path, help="rollout_specs.json")
    parser.add_argument("--ego-policy", required=True, type=Path)
    parser.add_argument("--n-adversaries", type=int, default=1)
    parser.add_argument("--feasibility-reruns", type=int, default=0,
                        help="reruns per rollout to estimate feasibility (0 = skip)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", type=str, default="",
                        help="Human-readable label embedded in results JSON, e.g. 'run_003_n2_seed42'")
    args = parser.parse_args()

    import yaml
    from baas.adapters.highway_env.adapter import HighwayEnvAdapter
    from baas.adapters.highway_env.config import HighwayEnvBenchmarkConfig
    from baas.core.ego_policy import DQNEgoPolicy, PPOEgoPolicy
    from baas.core.metrics import IncidentThresholds
    from baas.evaluation.benchmark import load_rollout_specs, rollout_specs_sha1
    from baas.evaluation.runner import evaluate_artefact, save_results

    cfg = yaml.safe_load(args.config.read_text())
    config_sha1 = __import__("hashlib").sha1(
        args.config.read_bytes()
    ).hexdigest()

    inc = cfg["incident"]
    thresholds = IncidentThresholds(
        critical_dist=inc["critical_dist"],
        nuisance_dist=inc["nuisance_dist"],
        ttc_crit_s=inc["ttc_crit_s"],
        dx_near_m=inc["dx_near_m"],
        dy_near_m=inc["dy_near_m"],
    )
    obs_mode = cfg.get("env", {}).get("obs_mode", "grayscale")
    env_cfg = HighwayEnvBenchmarkConfig(
        env_id=cfg["env"]["env_id"],
        policy_frequency=cfg["env"]["policy_frequency"],
        simulation_frequency=cfg["env"]["simulation_frequency"],
        horizon_steps=cfg["env"]["horizon_steps"],
        obs_mode=obs_mode,
    )

    specs = load_rollout_specs(args.specs)
    adapter = HighwayEnvAdapter()
    # Lidar ablation uses PPOEgoPolicy. All other methods use DQNEgoPolicy.
    if args.method == "mappo_adversary_lidar":
        ego = PPOEgoPolicy(str(args.ego_policy))
    else:
        ego = DQNEgoPolicy(str(args.ego_policy))

    adv_obs_extractor = None

    if args.method == "map_elites":
        make_controllers = _make_controllers_map_elites(args.artefact, cfg, args.n_adversaries)
    elif args.method in ("ppo_adversary", "qdrl"):
        make_controllers = _make_controllers_ppo(args.artefact, device=args.device)
    elif args.method == "mappo_adversary":
        make_controllers = _make_controllers_mappo(args.artefact, args.n_adversaries, device=args.device)
    elif args.method == "mappo_adversary_lidar":
        make_controllers = _make_controllers_mappo_lidar(args.artefact, args.n_adversaries, device=args.device)
        # Lidar adversaries receive obs from the env's MultiAgentObservation tuple
        adv_obs_extractor = lambda obs_tuple, i: (
            np.asarray(obs_tuple[i + 1], dtype=np.float32).flatten()
            if isinstance(obs_tuple, (tuple, list)) and len(obs_tuple) > i + 1
            else np.zeros(128, dtype=np.float32)
        )
    elif args.method == "maddpg_adversary":
        make_controllers = _make_controllers_maddpg(args.artefact, args.n_adversaries, device=args.device)
    elif args.method in ("king_light", "cma_es", "novelty_search"):
        make_controllers = _make_controllers_action_seq(args.artefact, args.n_adversaries)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    results = evaluate_artefact(
        specs=specs,
        adapter=adapter,
        ego_policy=ego,
        thresholds=thresholds,
        make_controllers=make_controllers,
        env_cfg=env_cfg,
        n_feasibility_reruns=args.feasibility_reruns,
        adv_obs_extractor=adv_obs_extractor,
    )

    save_results(
        results,
        args.output,
        method=args.method,
        config=cfg,
        config_sha1=config_sha1,
        specs=specs,
        run_id=args.run_id,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
