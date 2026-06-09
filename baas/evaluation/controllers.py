"""Adversary controller factories shared by run_eval.py and the replay loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List

import numpy as np


def make_controllers_ppo(artefact_path: Path, device: str = "cpu") -> Callable:
    from stable_baselines3 import PPO
    ppo = PPO.load(str(artefact_path), device=device)

    def _ctrl(obs: Any) -> int:
        act, _ = ppo.predict(np.asarray(obs, dtype=np.float32).flatten(), deterministic=True)
        return int(np.asarray(act).reshape(-1)[0])

    return lambda spec: [_ctrl]


def make_controllers_map_elites(
    artefact_path: Path, cfg: dict, n_adversaries: int
) -> Callable:
    from baas.evaluation.runner import action_seq_controllers
    from baas.methods.map_elites.genome import ActionSeqGenomeSpec

    data = json.loads(artefact_path.read_text(encoding="utf-8"))
    records = data.get("archive", [])
    if not records:
        raise RuntimeError(f"No elites in archive: {artefact_path}")

    best = max(records, key=lambda r: float(r.get("objective", float("-inf"))))
    sol_keys = sorted(k for k in best if k.startswith("solution_"))
    solution = np.array([float(best[k]) for k in sol_keys], dtype=np.float32)

    me_raw = cfg.get("map_elites", {})
    genome_spec = ActionSeqGenomeSpec(
        horizon_steps=cfg["env"]["horizon_steps"],
        n_adversaries=n_adversaries,
        n_blocks=me_raw.get("n_blocks", 20),
        block_size=me_raw.get("block_size", 3),
    )
    seqs = genome_spec.decode(genome_spec.from_continuous(solution))
    horizon = cfg["env"]["horizon_steps"]
    return lambda spec: action_seq_controllers(seqs, horizon)


def make_controllers_action_seq(artefact_path: Path) -> Callable:
    from baas.evaluation.runner import action_seq_controllers

    data = json.loads(artefact_path.read_text(encoding="utf-8"))
    adv_actions_idx: List[List[int]] = data.get("adv_actions_idx", [])
    if not adv_actions_idx:
        raise RuntimeError(f"No adv_actions_idx in: {artefact_path}")
    if isinstance(adv_actions_idx[0], int):
        adv_actions_idx = [adv_actions_idx]
    horizon = len(adv_actions_idx[0])
    return lambda spec: action_seq_controllers(adv_actions_idx, horizon)
