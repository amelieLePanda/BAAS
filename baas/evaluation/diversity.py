"""Diversity metrics for the MADS evaluation pipeline.

Three complementary metrics: phi-space coverage and Shannon entropy,
trajectory dispersion, and discrete Frechet distance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PhiGridConfig:
    """Grid parameters for discretising phi = (min_dist, TTCI)."""
    dist_bins: int = 10
    ttci_bins: int = 10
    dist_max: float = 60.0
    horizon_steps: int = 240


def bin_phi(phi: Tuple[float, float], cfg: PhiGridConfig) -> Optional[Tuple[int, int]]:
    """Map a phi value to a (dist_bin, ttci_bin) cell. Returns None if non-finite."""
    d, ttci = phi
    if not (np.isfinite(d) and np.isfinite(ttci)):
        return None

    d = max(0.0, min(float(d), cfg.dist_max))
    ttci = max(1.0, min(float(ttci), float(cfg.horizon_steps)))

    di = int(math.floor((d / cfg.dist_max) * cfg.dist_bins)) if cfg.dist_max > 0 else 0
    di = min(max(di, 0), cfg.dist_bins - 1)

    H = float(max(1, cfg.horizon_steps))
    ti_frac = (ttci - 1.0) / max(1e-9, H - 1.0) if H > 1 else 0.0
    ti = min(max(int(math.floor(ti_frac * cfg.ttci_bins)), 0), cfg.ttci_bins - 1)

    return (di, ti)


def phi_coverage_entropy(phis: List[Tuple[float, float]], cfg: PhiGridConfig) -> Dict[str, float]:
    """Compute phi-space coverage, normalised entropy, and diversity score."""
    K = cfg.dist_bins * cfg.ttci_bins
    counts: Dict[Tuple[int, int], int] = {}
    for phi in phis:
        cell = bin_phi(phi, cfg)
        if cell is not None:
            counts[cell] = counts.get(cell, 0) + 1

    if not counts:
        nan = float("nan")
        return dict(coverage=nan, entropy_norm=nan, diversity_score=nan,
                    bins_occupied=0, total_bins=K)

    occ = len(counts)
    n = float(sum(counts.values()))
    ps = np.array([c / n for c in counts.values()], dtype=np.float64)
    ent = float(-np.sum(ps * np.log(ps + 1e-12)))
    ent_norm = float(ent / math.log(K)) if K > 1 else 0.0

    return dict(
        coverage=float(occ / K),
        entropy_norm=ent_norm,
        diversity_score=0.5 * (occ / K + ent_norm),
        bins_occupied=occ,
        total_bins=K,
    )


def trajectory_dispersion(trajectories: List[np.ndarray]) -> float:
    """Mean pairwise L2 dispersion of flattened ego-relative trajectories.

    Each trajectory should be (T, 2). Returns nan for fewer than 2 inputs.
    """
    if len(trajectories) < 2:
        return float("nan")

    flat = [t.reshape(-1).astype(np.float64) for t in trajectories]
    ds: List[float] = []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            a, b = flat[i], flat[j]
            L = min(len(a), len(b))
            ds.append(float(np.linalg.norm(a[:L] - b[:L])))

    return float(np.mean(ds)) if ds else float("nan")


def _frechet_dp(P: np.ndarray, Q: np.ndarray) -> float:
    """Discrete Frechet distance via DP. O(T1*T2) time, O(T2) space."""
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return float("nan")

    prev = np.full(m, np.inf, dtype=np.float64)
    curr = np.full(m, np.inf, dtype=np.float64)

    for i in range(n):
        for j in range(m):
            d = float(np.linalg.norm(P[i] - Q[j]))
            if i == 0 and j == 0:
                curr[j] = d
            elif i == 0:
                curr[j] = max(curr[j - 1], d)
            elif j == 0:
                curr[j] = max(prev[j], d)
            else:
                curr[j] = max(min(prev[j], prev[j - 1], curr[j - 1]), d)
        prev, curr = curr, prev
        curr.fill(np.inf)

    return float(prev[m - 1])


def _downsample(traj: np.ndarray, n: int) -> np.ndarray:
    """Uniformly downsample (T, D) to exactly n rows."""
    T = len(traj)
    if T <= n:
        return traj
    idx = np.round(np.linspace(0, T - 1, n)).astype(int)
    return traj[idx]


def ego_relative_trajectory(ego_trace: np.ndarray, adv_trace: np.ndarray) -> np.ndarray:
    """Compute ego-relative adversary trajectory, zero-centred at t=0.

    ego_trace and adv_trace should be (T, 4), columns [x, y, psi, v].
    Returns (T, 2) float32 as [adv_x - ego_x, adv_y - ego_y] relative to t=0.
    """
    T = min(len(ego_trace), len(adv_trace))
    rel = adv_trace[:T, :2] - ego_trace[:T, :2]
    return (rel - rel[0:1, :]).astype(np.float32)


def compute_frechet_diversity(
    trajectories: List[np.ndarray],
    downsample_to: int = 30,
) -> Dict[str, float]:
    """Mean and median pairwise discrete Frechet distance.

    Trajectories are downsampled to downsample_to steps before computation.
    Only upper-triangular pairs are computed (N*(N-1)/2 total).
    """
    if len(trajectories) < 2:
        return dict(frechet_mean=float("nan"), frechet_median=float("nan"), n_pairs=0)

    ds_trajs = [_downsample(t, downsample_to) for t in trajectories]
    dists: List[float] = []
    for i in range(len(ds_trajs)):
        for j in range(i + 1, len(ds_trajs)):
            d = _frechet_dp(ds_trajs[i], ds_trajs[j])
            if np.isfinite(d):
                dists.append(float(d))

    if not dists:
        return dict(frechet_mean=float("nan"), frechet_median=float("nan"), n_pairs=0)

    return dict(
        frechet_mean=float(np.mean(dists)),
        frechet_median=float(np.median(dists)),
        n_pairs=len(dists),
    )


def compute_diversity_report(
    phis: List[Tuple[float, float]],
    trajectories: List[np.ndarray],
    phi_cfg: PhiGridConfig,
    frechet_downsample_to: int = 30,
) -> Dict[str, float]:
    """Run all three diversity metrics and return a combined flat dict.

    trajectories should be ego-relative, from ego_relative_trajectory().
    """
    report: Dict[str, float] = {}
    report.update(phi_coverage_entropy(phis, phi_cfg))
    report["dispersion"] = trajectory_dispersion(trajectories)
    report.update(compute_frechet_diversity(trajectories, downsample_to=frechet_downsample_to))
    return report
