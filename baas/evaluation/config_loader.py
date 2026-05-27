"""Central YAML-to-config loader.

Load a benchmark_vN.yaml once and get all config objects in one call.
Every script should use load_config() instead of manually extracting keys.

Usage:
    bundle = load_config(Path("configs/benchmark_v1.yaml"))
    # or if you already have the parsed dict:
    bundle = config_from_dict(raw_dict)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from baas.adapters.highway_env.config import HighwayEnvBenchmarkConfig
from baas.core.metrics import IncidentThresholds
from baas.methods.cma_es.config import CMAESConfig
from baas.methods.king_light.config import KingLightConfig
from baas.methods.map_elites.search import MapElitesConfig
from baas.methods.novelty_search.config import NoveltySearchConfig
from baas.methods.parameter_sweep.config import ParameterSweepConfig
from baas.methods.ppo_adversary.config import PPOAdvConfig
from baas.methods.qdrl.config import QDRLConfig
from baas.training.config import EgoTrainConfig


@dataclass
class BenchmarkConfigBundle:
    """All config objects unpacked from a single benchmark YAML file."""
    env: HighwayEnvBenchmarkConfig
    thresholds: IncidentThresholds
    ego_training: EgoTrainConfig
    map_elites: MapElitesConfig
    ppo_adversary: PPOAdvConfig
    cma_es: CMAESConfig
    novelty_search: NoveltySearchConfig
    qdrl: QDRLConfig
    king_light: KingLightConfig
    parameter_sweep: ParameterSweepConfig
    raw: Dict[str, Any]  # original dict kept for provenance embedding


def load_config(path: Path) -> BenchmarkConfigBundle:
    """Parse a YAML file and return a BenchmarkConfigBundle."""
    raw = yaml.safe_load(Path(path).read_text())
    return config_from_dict(raw)


def config_from_dict(raw: Dict[str, Any]) -> BenchmarkConfigBundle:
    """Build a BenchmarkConfigBundle from a parsed YAML dict."""
    env_raw = raw.get("env", {})
    inc_raw = raw.get("incident", {})
    roll_raw = raw.get("rollouts", {})
    pert_raw = raw.get("perturbation", {})
    div_raw = raw.get("diversity", {})
    me_raw = raw.get("map_elites", {})
    ppo_raw = raw.get("ppo_adversary", {})
    cma_raw = raw.get("cma_es", {})
    ns_raw = raw.get("novelty_search", {})
    qdrl_raw = raw.get("qdrl", {})
    king_raw = raw.get("king_light", {})
    ps_raw = raw.get("parameter_sweep", {})

    ego_raw = raw.get("ego_training", {})

    env = HighwayEnvBenchmarkConfig(
        env_id=env_raw.get("env_id", "highway-v0"),
        policy_frequency=env_raw.get("policy_frequency", 2),
        simulation_frequency=env_raw.get("simulation_frequency", 15),
        horizon_steps=env_raw.get("horizon_steps", 240),
        background_traffic=env_raw.get("background_traffic", 6),
        vehicles_density=env_raw.get("vehicles_density", 1.0),
        obs_mode=env_raw.get("obs_mode", "grayscale"),
        k=roll_raw.get("k", 30),
        base_seed=roll_raw.get("base_seed", 0),
        env_seed_mode=roll_raw.get("env_seed_mode", "jitter"),
        salt=roll_raw.get("salt", 0),
        ego_dy_eps=pert_raw.get("ego_dy_eps", 0.5),
        ego_dv_eps=pert_raw.get("ego_dv_eps", 1.0),
        adv_dy_eps=pert_raw.get("adv_dy_eps", 0.5),
        adv_dv_eps=pert_raw.get("adv_dv_eps", 1.0),
        critical_dist=inc_raw.get("critical_dist", 6.0),
        nuisance_dist=inc_raw.get("nuisance_dist", 12.0),
        ttc_crit_s=inc_raw.get("ttc_crit_s", 1.5),
        dx_near_m=inc_raw.get("dx_near_m", 5.0),
        dy_near_m=inc_raw.get("dy_near_m", 1.0),
    )

    thresholds = IncidentThresholds(
        critical_dist=inc_raw.get("critical_dist", 6.0),
        nuisance_dist=inc_raw.get("nuisance_dist", 12.0),
        ttc_crit_s=inc_raw.get("ttc_crit_s", 1.5),
        dx_near_m=inc_raw.get("dx_near_m", 5.0),
        dy_near_m=inc_raw.get("dy_near_m", 1.0),
        v_stop_mps=inc_raw.get("v_stop_mps", 1.0),
        stop_streak_steps=inc_raw.get("stop_streak_steps", 5),
        hard_brake_a=inc_raw.get("hard_brake_a", -4.0),
    )

    map_elites = MapElitesConfig(
        n_iterations=me_raw.get("n_iterations", 5000),
        batch_size=me_raw.get("batch_size", 30),
        initial_population=me_raw.get("initial_population", 100),
        sigma0=me_raw.get("sigma0", 1.0),
        dist_bins=div_raw.get("dist_bins", 10),
        ttci_bins=div_raw.get("ttci_bins", 10),
        dist_max=div_raw.get("dist_max", 60.0),
        snapshot_every=me_raw.get("snapshot_every", 500),
    )

    ppo_adversary = PPOAdvConfig(
        n_adversaries=ppo_raw.get("n_adversaries", 1),
        total_timesteps=ppo_raw.get("total_timesteps", 500_000),
        lr=ppo_raw.get("lr", 3e-4),
        gamma=ppo_raw.get("gamma", 0.99),
        gae_lambda=ppo_raw.get("gae_lambda", 0.95),
        n_steps=ppo_raw.get("n_steps", 2048),
        batch_size=ppo_raw.get("batch_size", 64),
        ent_coef=ppo_raw.get("ent_coef", 0.01),
        clip_range=ppo_raw.get("clip_range", 0.2),
        min_ttf_steps=ppo_raw.get("min_ttf_steps", 10),
        w_proximity=ppo_raw.get("w_proximity", 0.05),
        c_time_step=ppo_raw.get("c_time_step", 0.001),
        r_critical=ppo_raw.get("r_critical", 1.5),
        r_adv_crash_close=ppo_raw.get("r_adv_crash_close", -1.0),
        r_adv_crash_nuisance=ppo_raw.get("r_adv_crash_nuisance", -6.0),
    )

    cma_es = CMAESConfig(
        n_adversaries=cma_raw.get("n_adversaries", 1),
        popsize=cma_raw.get("popsize", 20),
        sigma0=cma_raw.get("sigma0", 0.5),
        n_generations=cma_raw.get("n_generations", 200),
    )

    novelty_search = NoveltySearchConfig(
        n_adversaries=ns_raw.get("n_adversaries", 1),
        pop_size=ns_raw.get("pop_size", 50),
        n_generations=ns_raw.get("n_generations", 200),
        k_nearest=ns_raw.get("k_nearest", 10),
        archive_prob=ns_raw.get("archive_prob", 0.2),
    )

    qdrl_grid: List[int] = qdrl_raw.get("archive_grid_dims", [25, 25])
    qdrl = QDRLConfig(
        n_adversaries=qdrl_raw.get("n_adversaries", 1),
        n_iters=qdrl_raw.get("n_iters", 60),
        n_emitters=qdrl_raw.get("n_emitters", 4),
        emitter_batch_size=qdrl_raw.get("emitter_batch_size", 2),
        sigma0=qdrl_raw.get("sigma0", 0.8),
        snapshot_every=qdrl_raw.get("snapshot_every", 5),
        archive_grid_dims=qdrl_grid,
        dist_range=qdrl_raw.get("dist_range", [0.0, 50.0]),
        burst_steps=qdrl_raw.get("burst_steps", 50_000),
        n_envs=qdrl_raw.get("n_envs", 1),
        eval_rollouts=qdrl_raw.get("eval_rollouts", 6),
        r_critical_range=qdrl_raw.get("r_critical_range", [1.0, 2.0]),
        r_adv_close_range=qdrl_raw.get("r_adv_close_range", [-2.0, -0.2]),
        r_adv_nuis_range=qdrl_raw.get("r_adv_nuis_range", [-10.0, -2.0]),
        lr=qdrl_raw.get("lr", 3e-4),
        gamma=qdrl_raw.get("gamma", 0.99),
        gae_lambda=qdrl_raw.get("gae_lambda", 0.95),
        ppo_n_steps=qdrl_raw.get("ppo_n_steps", 1024),
        ppo_batch_size=qdrl_raw.get("ppo_batch_size", 256),
        ent_coef=qdrl_raw.get("ent_coef", 0.0),
        clip_range=qdrl_raw.get("clip_range", 0.2),
        device=qdrl_raw.get("device", "auto"),
    )

    king_light = KingLightConfig(
        n_adversaries=king_raw.get("n_adversaries", 1),
        n_opt_steps=king_raw.get("n_opt_steps", 400),
        lr=king_raw.get("lr", 0.05),
        device=king_raw.get("device", "cpu"),
        attack_horizon=king_raw.get("attack_horizon", 60),
        gumbel_tau=king_raw.get("gumbel_tau", 1.0),
        max_steer=king_raw.get("max_steer", 0.35),
        acc_lo=king_raw.get("acc_lo", -6.0),
        acc_hi=king_raw.get("acc_hi", 6.0),
        collision_radius=king_raw.get("collision_radius", 2.0),
        warmup_steps=king_raw.get("warmup_steps", 5),
        warmup_penalty=king_raw.get("warmup_penalty", 1.0),
        progress_weight=king_raw.get("progress_weight", 0.05),
    )

    parameter_sweep = ParameterSweepConfig(
        delta_x_grid=ps_raw.get("delta_x_grid", [-40.0, -20.0, 0.0, 20.0]),
        delta_y_grid=ps_raw.get("delta_y_grid", [-3.5, 0.0, 3.5]),
        delta_v_grid=ps_raw.get("delta_v_grid", [-5.0, 0.0, 5.0]),
    )

    ego_training = EgoTrainConfig(
        env_id=env_raw.get("env_id", "highway-v0"),
        policy_frequency=env_raw.get("policy_frequency", 2),
        simulation_frequency=env_raw.get("simulation_frequency", 15),
        horizon_steps=env_raw.get("horizon_steps", 240),
        background_traffic=env_raw.get("background_traffic", 6),
        learning_rate=ego_raw.get("learning_rate", 5e-4),
        lr_schedule=ego_raw.get("lr_schedule", "constant"),
        buffer_size=ego_raw.get("buffer_size", 15_000),
        learning_starts=ego_raw.get("learning_starts", 200),
        batch_size=ego_raw.get("batch_size", 32),
        gamma=ego_raw.get("gamma", 0.8),
        train_freq=ego_raw.get("train_freq", 1),
        gradient_steps=ego_raw.get("gradient_steps", 1),
        target_update_interval=ego_raw.get("target_update_interval", 50),
        exploration_fraction=ego_raw.get("exploration_fraction", 0.7),
        exploration_initial_eps=ego_raw.get("exploration_initial_eps", 1.0),
        exploration_final_eps=ego_raw.get("exploration_final_eps", 0.05),
        collision_reward=ego_raw.get("collision_reward", -3.0),
        lane_change_reward=ego_raw.get("lane_change_reward", -0.1),
        high_speed_reward=ego_raw.get("high_speed_reward", 0.1),
        right_lane_reward=ego_raw.get("right_lane_reward", 0.1),
    )

    return BenchmarkConfigBundle(
        env=env,
        thresholds=thresholds,
        ego_training=ego_training,
        map_elites=map_elites,
        ppo_adversary=ppo_adversary,
        cma_es=cma_es,
        novelty_search=novelty_search,
        qdrl=qdrl,
        king_light=king_light,
        parameter_sweep=parameter_sweep,
        raw=raw,
    )
