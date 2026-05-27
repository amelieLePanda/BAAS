"""Rollout-spec generation, serialisation, and SHA-1 provenance.

The single place that creates RolloutSpec lists.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from baas.core.types import Perturb, RolloutSpec

_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _hash64(base_seed: int, rollout_index: int, salt: int = 0) -> int:
    x = (
        (int(base_seed) & _MASK64)
        ^ ((int(rollout_index) * 0xD2B74407B1CE6E93) & _MASK64)
        ^ (int(salt) & _MASK64)
    )
    return _splitmix64(x)


def rollout_rng(base_seed: int, rollout_index: int, salt: int = 0) -> np.random.Generator:
    """Deterministic RNG for a given rollout, independent of evaluation order."""
    return np.random.default_rng(_hash64(base_seed, rollout_index, salt=salt))


def make_rollout_specs(
    *,
    env_id: str,
    horizon_steps: int,
    n_adversaries: int,
    background_traffic: int,
    k: int,
    base_seed: int,
    env_seed_mode: str = "jitter",
    ego_dy_eps: float = 0.5,
    ego_dv_eps: float = 1.0,
    adv_dy_eps: float = 0.5,
    adv_dv_eps: float = 1.0,
    salt: int = 0,
    tag: Optional[str] = None,
) -> List[RolloutSpec]:
    """Generate the benchmark rollout list.

    env_seed_mode="jitter" increments the env seed per rollout.
    env_seed_mode="fixed" reuses base_seed for every rollout.
    Perturbations are sampled deterministically per rollout from rollout_rng.
    """
    if env_seed_mode not in ("jitter", "fixed"):
        raise ValueError("env_seed_mode must be 'jitter' or 'fixed'")

    specs: List[RolloutSpec] = []
    for i in range(int(k)):
        env_seed_i = int(base_seed + i) if env_seed_mode == "jitter" else int(base_seed)
        rng = rollout_rng(base_seed, i, salt=salt)

        specs.append(RolloutSpec(
            rollout_index=i,
            env_seed=env_seed_i,
            env_id=env_id,
            horizon_steps=horizon_steps,
            n_adversaries=n_adversaries,
            background_traffic=background_traffic,
            perturb=Perturb.from_flat(
                float(rng.uniform(-ego_dy_eps, ego_dy_eps)),
                float(rng.uniform(-ego_dv_eps, ego_dv_eps)),
                float(rng.uniform(-adv_dy_eps, adv_dy_eps)),
                float(rng.uniform(-adv_dv_eps, adv_dv_eps)),
                n_adversaries,
            ),
            tag=tag,
        ))
    return specs


def make_rollout_specs_from_config(cfg: Dict[str, Any], n_adversaries: int = 1) -> List[RolloutSpec]:
    """Convenience wrapper that builds specs directly from a loaded YAML config dict."""
    env = cfg["env"]
    rollouts = cfg["rollouts"]
    perturb = cfg["perturbation"]
    return make_rollout_specs(
        env_id=env["env_id"],
        horizon_steps=env["horizon_steps"],
        n_adversaries=n_adversaries,
        background_traffic=env["background_traffic"],
        k=rollouts["k"],
        base_seed=rollouts["base_seed"],
        env_seed_mode=rollouts["env_seed_mode"],
        ego_dy_eps=perturb["ego_dy_eps"],
        ego_dv_eps=perturb["ego_dv_eps"],
        adv_dy_eps=perturb["adv_dy_eps"],
        adv_dv_eps=perturb["adv_dv_eps"],
        salt=rollouts["salt"],
    )


def _spec_to_dict(s: RolloutSpec) -> Dict[str, Any]:
    return {
        "rollout_index": s.rollout_index,
        "env_seed": s.env_seed,
        "env_id": s.env_id,
        "horizon_steps": s.horizon_steps,
        "n_adversaries": s.n_adversaries,
        "background_traffic": s.background_traffic,
        "perturb": {"agents": list(s.perturb.agents)},
        "tag": s.tag,
    }


def rollout_specs_sha1(specs: List[RolloutSpec]) -> str:
    """Stable SHA-1 of the spec list, embedded in every output JSON."""
    payload = json.dumps(
        [_spec_to_dict(s) for s in specs], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def save_rollout_specs(specs: List[RolloutSpec], path: Path) -> str:
    """Write specs to path and return the SHA-1."""
    sha = rollout_specs_sha1(specs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"sha1": sha, "rollout_specs": [_spec_to_dict(s) for s in specs]}, indent=2),
        encoding="utf-8",
    )
    return sha


def load_rollout_specs(path: Path) -> List[RolloutSpec]:
    """Load specs from a previously saved JSON file."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    raw = obj.get("rollout_specs", obj)
    if not isinstance(raw, list):
        raise ValueError("Expected a list under 'rollout_specs'")

    specs: List[RolloutSpec] = []
    for r in raw:
        p = r.get("perturb", {})
        agents_raw = p.get("agents", [])
        if agents_raw:
            agents = tuple(tuple(a) for a in agents_raw)
        else:
            # Legacy flat format
            n_adv = int(r.get("n_adversaries", 1))
            agents = tuple(
                (float(p.get("dy_ego", 0)), float(p.get("dv_ego", 0)))
                if i == 0
                else (float(p.get("dy_adv", 0)), float(p.get("dv_adv", 0)))
                for i in range(1 + n_adv)
            )
        specs.append(RolloutSpec(
            rollout_index=int(r["rollout_index"]),
            env_seed=int(r["env_seed"]),
            env_id=str(r["env_id"]),
            horizon_steps=int(r["horizon_steps"]),
            n_adversaries=int(r.get("n_adversaries", 1)),
            background_traffic=int(r["background_traffic"]),
            perturb=Perturb(agents=agents),  # type: ignore[arg-type]
            tag=r.get("tag"),
        ))
    return specs
