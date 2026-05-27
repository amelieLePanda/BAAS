"""KING-light: gradient-based adversary optimisation via BicycleProxy.

Optimises a discrete adversary action sequence over DiscreteMetaAction indices
using a differentiable BicycleProxy and Gumbel-Softmax, then exports the
sequence for replay in highway-env.

Single adversary only. The policy_frequency mismatch from the original
king_light.py is fixed here: proxy dt = 1 / policy_frequency, taken from
the experiment config.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from baas.core.env_adapter import EnvAdapter
from baas.core.ego_policy import EgoPolicy
from baas.core.metrics import IncidentThresholds
from baas.core.types import RolloutSpec
from baas.methods.king_light.config import KingLightConfig
from baas.methods.king_light.proxy import BicycleProxy

logger = logging.getLogger(__name__)

N_META_ACTIONS = 5  # DiscreteMetaAction: 0 LEFT  1 IDLE  2 RIGHT  3 FASTER  4 SLOWER


def _build_meta_to_proxy(adapter: EnvAdapter, device: torch.device) -> torch.Tensor:
    """Convert adapter.action_map() to (A, 2) tensor for matrix multiply in rollout."""
    amap = adapter.action_map()
    rows = [amap[i] for i in range(len(amap))]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _ego_controller(state: torch.Tensor, ref_y: float, ref_v: float) -> torch.Tensor:
    """Simple lane-keeping and speed controller for the proxy ego."""
    y, psi, v = state[1], state[2], state[3]
    steer_raw = (-2.0 * (y - ref_y)) + (-1.0 * psi)
    accel_raw = -1.5 * (v - ref_v)
    return torch.stack([steer_raw, accel_raw])


def rollout_proxy(
    ego0: torch.Tensor,
    adv0: torch.Tensor,
    adv_onehot: torch.Tensor,
    proxy: BicycleProxy,
    meta_to_proxy: torch.Tensor,
) -> Dict[str, Any]:
    """Differentiable rollout under the bicycle proxy.

    ego0: (4,), adv0: (N, 4), adv_onehot: (T, N, A)
    """
    device = ego0.device
    T, N, A = adv_onehot.shape

    ego_states = [ego0]
    adv_states = [adv0]

    ref_y = float(ego0[1].item())
    ref_v = float(ego0[3].item())

    # Pre-compute continuous controls for all steps: (T, N, 2)
    adv_u = torch.einsum("tna,ac->tnc", adv_onehot, meta_to_proxy)

    for t in range(T):
        ego_u = _ego_controller(ego_states[-1], ref_y=ref_y, ref_v=ref_v)
        ego_states.append(proxy.step(ego_states[-1], ego_u))
        adv_states.append(proxy.step(adv_states[-1], adv_u[t]))

    ego_traj = torch.stack(ego_states, dim=0)   # (T+1, 4)
    adv_traj = torch.stack(adv_states, dim=0)   # (T+1, N, 4)

    ego_xy = ego_traj[:, :2]                    # (T+1, 2)
    adv_xy = adv_traj[:, :, :2]                 # (T+1, N, 2)
    d = torch.cdist(ego_xy.unsqueeze(1), adv_xy).squeeze(1)  # (T+1, N)
    d_t, _ = d.min(dim=1)                       # (T+1,)

    ego_progress = (ego_traj[-1, 0] - ego_traj[0, 0]).clamp(min=0.0)

    return {"ego_traj": ego_traj, "adv_traj": adv_traj, "d_t": d_t, "ego_progress": ego_progress}


def _objective(roll: Dict[str, Any], cfg: KingLightConfig) -> torch.Tensor:
    """Differentiable objective: pull adversary close to ego, weighted early."""
    d_t: torch.Tensor = roll["d_t"]
    ego_progress: torch.Tensor = roll["ego_progress"]

    d = d_t[1:]  # skip t=0
    T = d.shape[0]
    r = float(cfg.collision_radius)

    # Exponentially decaying weights so getting close early is worth more
    decay = 0.97
    w = decay ** torch.arange(T, device=d.device, dtype=d.dtype)
    w = w / (w.sum() + 1e-8)

    hinge = torch.relu(d - r)
    collide_term = (w * hinge).sum()

    # Softmin distance term pulls the whole trajectory toward small distances
    softmin_term = 0.25 * (-torch.logsumexp(-d, dim=0))

    progress_term = -cfg.progress_weight * ego_progress

    # Penalise very early collisions to discourage trivial stationary attacks
    warm = max(0, min(int(cfg.warmup_steps), T))
    if warm > 0:
        early_inside = (d[:warm] <= r).float().mean()
        warm_term = cfg.warmup_penalty * early_inside
    else:
        warm_term = torch.tensor(0.0, device=d.device)

    return collide_term + softmin_term + warm_term + progress_term


def _hard_metrics(d_t: torch.Tensor, cfg: KingLightConfig) -> Tuple[bool, Optional[int], float]:
    """Non-differentiable: check collision and extract proxy metrics."""
    r = float(cfg.collision_radius)
    d_np = d_t.detach().cpu().numpy()
    min_dist = float(d_np.min())
    collided = bool((d_np <= r).any())
    ttf_step = int(np.argmax(d_np <= r)) if collided else None
    return collided, ttf_step, min_dist


def run_king_light(
    spec: RolloutSpec,
    adapter: EnvAdapter,
    ego_policy: EgoPolicy,
    thresholds: IncidentThresholds,
    cfg: KingLightConfig,
    *,
    env_cfg: Any = None,
) -> Dict[str, Any]:
    """Run KING-light proxy optimisation and return the best action sequence.

    Uses the spec's env_seed and perturb to place ego and adversary, then
    optimises the adversary's action sequence via gradient descent through
    the BicycleProxy.

    Returns a dict with 'adv_actions_idx' (list of int, length horizon_steps)
    and proxy-side metrics.
    """
    from baas.adapters.highway_env.seeder import sample_initial_states

    if cfg.n_adversaries != 1:
        raise ValueError("KING-light supports single adversary only (n_adversaries=1).")

    device = torch.device(cfg.device)

    env = adapter.make_env(env_cfg, n_adversaries=1)
    try:
        ego_np, adv_np = sample_initial_states(env, n_adversaries=1, seed=spec.env_seed)
    finally:
        env.close()

    ego0 = torch.tensor(ego_np, dtype=torch.float32, device=device)
    adv0 = torch.tensor(adv_np, dtype=torch.float32, device=device)

    if len(spec.perturb.agents) >= 2:
        _, dv_ego = spec.perturb.agents[0]
        dy_adv, dv_adv = spec.perturb.agents[1]
        ego0 = ego0.clone()
        ego0[3] = (ego0[3] + float(dv_ego)).clamp(min=0.0)
        adv0 = adv0.clone()
        adv0[0, 1] = adv0[0, 1] + float(dy_adv)
        adv0[0, 3] = (adv0[0, 3] + float(dv_adv)).clamp(min=0.0)

    policy_frequency = float(
        getattr(env_cfg, "policy_frequency", None) or 2
    )
    proxy = BicycleProxy(
        dt=1.0 / policy_frequency,
        max_steer=cfg.max_steer,
        acc_lo=cfg.acc_lo,
        acc_hi=cfg.acc_hi,
    ).to(device)

    meta_to_proxy = _build_meta_to_proxy(adapter, device)

    T_attack = cfg.attack_horizon
    horizon = spec.horizon_steps
    IDLE = 1

    logits = torch.zeros(
        (T_attack, cfg.n_adversaries, N_META_ACTIONS), device=device, requires_grad=True
    )
    opt = torch.optim.Adam([logits], lr=cfg.lr)

    best_loss = float("inf")
    best_pack: Optional[Dict[str, Any]] = None
    best_collide_pack: Optional[Dict[str, Any]] = None
    best_collide_ttf: int = 10 ** 9

    for it in range(cfg.n_opt_steps):
        opt.zero_grad(set_to_none=True)

        adv_onehot = F.gumbel_softmax(logits, tau=cfg.gumbel_tau, hard=True, dim=-1)
        roll = rollout_proxy(ego0, adv0, adv_onehot, proxy, meta_to_proxy)
        loss = _objective(roll, cfg)
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().cpu().item())
        collided, ttf_step, min_dist = _hard_metrics(roll["d_t"].detach(), cfg)

        def _pack_actions() -> list:
            idx = adv_onehot.detach().argmax(dim=-1).cpu().tolist()  # (T_attack, N)
            if T_attack < horizon:
                idx += [[IDLE] * cfg.n_adversaries for _ in range(horizon - T_attack)]
            return idx

        if loss_val < best_loss:
            best_loss = loss_val
            best_pack = {
                "loss": best_loss,
                "adv_actions_idx": _pack_actions(),
                "metrics": {
                    "collided": collided,
                    "ttf_step": ttf_step,
                    "min_dist": min_dist,
                    "ego_progress": float(roll["ego_progress"].detach().cpu().item()),
                },
            }

        if collided and ttf_step is not None:
            better = (best_collide_pack is None) or (ttf_step < best_collide_ttf)
            if better:
                best_collide_pack = {
                    "loss": loss_val,
                    "adv_actions_idx": _pack_actions(),
                    "metrics": {
                        "collided": True,
                        "ttf_step": int(ttf_step),
                        "min_dist": float(min_dist),
                        "ego_progress": float(roll["ego_progress"].detach().cpu().item()),
                    },
                }
                best_collide_ttf = int(ttf_step)

        if (it + 1) % 50 == 0:
            logger.info(
                "KING-light step %d/%d  loss=%.4f  collided=%s  min_dist=%.3f",
                it + 1, cfg.n_opt_steps, loss_val, collided, min_dist,
            )

    result = best_collide_pack if best_collide_pack is not None else best_pack
    assert result is not None
    return result
