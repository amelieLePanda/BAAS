"""Aggregate evaluation results into summary tables and LaTeX output."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baas.evaluation.diversity import (
    PhiGridConfig,
    compute_frechet_diversity,
    ego_relative_trajectory,
    phi_coverage_entropy,
    trajectory_dispersion,
)

logger = logging.getLogger(__name__)

# Default phi grid config - matches benchmark_v1.yaml diversity section.
_DEFAULT_PHI_CFG = PhiGridConfig(dist_bins=10, ttci_bins=10, dist_max=60.0, horizon_steps=240)
_FRECHET_DOWNSAMPLE = 30  # steps used for Fréchet / dispersion (matches runner.py default)


def load_result_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def load_results_dir(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all result JSON files found recursively under results_dir."""
    runs = []
    for p in sorted(results_dir.rglob("*.json")):
        data = load_result_json(p)
        # Must have both "method" and "rollouts" to be a proper result file.
        # Excludes rollout spec files, archive snapshots, and other JSON artifacts.
        if data and "method" in data and "rollouts" in data:
            runs.append(data)
    logger.info("Loaded %d result files from %s", len(runs), results_dir)
    return runs


def _safe(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def _extract_rel_trajectories(rollouts: List[Dict[str, Any]]) -> List[np.ndarray]:
    """Extract ego-relative adversary trajectories from rollout dicts.

    Uses the downsampled traces stored by save_results (ego_trace_ds / adv_traces_ds).
    Returns a flat list of (T, 2) ego-relative trajectories - one per adversary per rollout.
    """
    trajs: List[np.ndarray] = []
    for r in rollouts:
        ego_ds = r.get("ego_trace_ds")
        adv_ds_list = r.get("adv_traces_ds")
        if not ego_ds or not adv_ds_list:
            continue
        ego_arr = np.array(ego_ds, dtype=np.float32)  # (T_ds, 4)
        for adv_ds in adv_ds_list:
            if not adv_ds:
                continue
            adv_arr = np.array(adv_ds, dtype=np.float32)  # (T_ds, 4)
            rel = ego_relative_trajectory(ego_arr, adv_arr)  # (T, 2)
            trajs.append(rel)
    return trajs


def aggregate_method(
    rollouts: List[Dict[str, Any]],
    phi_cfg: PhiGridConfig = _DEFAULT_PHI_CFG,
    frechet_downsample_to: int = _FRECHET_DOWNSAMPLE,
) -> Dict[str, Any]:
    """Compute aggregate statistics over a list of rollout dicts."""
    if not rollouts:
        return {}

    # Guard: skip entries that pre-date the nested "metrics" dict format
    rollouts = [r for r in rollouts if isinstance(r.get("metrics"), dict)]
    if not rollouts:
        return {}

    def _col(key: str) -> List[float]:
        return [v for r in rollouts if (v := _safe(r.get("metrics", {}).get(key))) is not None]

    n = len(rollouts)

    # --- Effectiveness ---
    p_collision    = float(np.mean([r["metrics"].get("ego_collision", False) for r in rollouts]))
    p_critical     = float(np.mean([r["metrics"].get("critical_incident", False) for r in rollouts]))
    # Adversary-caused critical incidents (excludes background traffic interactions)
    p_critical_adv = float(np.mean([r["metrics"].get("critical_incident_adv", False) for r in rollouts]))
    ttci_vals      = _col("time_to_critical_incident_steps")
    ttci_adv_vals  = _col("time_to_critical_incident_adv_steps")
    dist_vals      = _col("min_dist_ego_adv")
    speed_vals   = _col("ego_speed_mean")
    accel_vals   = _col("ego_accel_min")
    adv_acc_vals = _col("max_adv_accel")
    nuisance_rate = float(np.mean([r["metrics"].get("adv_nuisance_crash", False) for r in rollouts]))

    # --- Feasibility ---
    feas_vals = _col("feasibility")
    mean_feasibility = round(float(np.mean(feas_vals)), 4) if feas_vals else None

    # Difficulty label distribution (fraction of rollouts in each tier)
    label_counts: Dict[str, int] = {}
    for r in rollouts:
        lbl = r.get("metrics", {}).get("difficulty_label")
        if lbl:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
    difficulty_dist = {k: round(v / n, 4) for k, v in sorted(label_counts.items())}

    # --- Multi-adversary coordination ---
    adv_adv_vals = _col("adv_adv_collision_count")
    mean_adv_adv_collisions = round(float(np.mean(adv_adv_vals)), 4) if adv_adv_vals else None

    # --- Diversity: phi-space coverage + entropy ---
    phi_list: List[Tuple[float, float]] = []
    for r in rollouts:
        phi = r.get("metrics", {}).get("phi")
        if phi and len(phi) == 2:
            d, t = phi
            if d is not None and t is not None:
                phi_list.append((float(d), float(t)))

    phi_div = phi_coverage_entropy(phi_list, phi_cfg) if phi_list else {}

    # --- Diversity: trajectory dispersion + Fréchet ---
    rel_trajs = _extract_rel_trajectories(rollouts)
    dispersion = trajectory_dispersion(rel_trajs) if len(rel_trajs) >= 2 else float("nan")
    frechet = compute_frechet_diversity(rel_trajs, downsample_to=frechet_downsample_to)

    result: Dict[str, Any] = {
        # Effectiveness - adversary-caused (primary paper metrics)
        "n_rollouts":           n,
        "p_collision":          round(p_collision, 4),
        "p_critical_adv":       round(p_critical_adv, 4),
        "mean_ttci_adv_steps":  round(float(np.mean(ttci_adv_vals)), 2) if ttci_adv_vals else None,
        "mean_min_dist_adv":    round(float(np.mean(dist_vals)), 3) if dist_vals else None,
        # Effectiveness - any vehicle (includes background traffic, context only)
        "p_critical_any":       round(p_critical, 4),
        "mean_ttci_any_steps":  round(float(np.mean(ttci_vals)), 2) if ttci_vals else None,
        # Realism
        "ego_speed_mean":      round(float(np.mean(speed_vals)), 3) if speed_vals else None,
        "ego_accel_min_mean":  round(float(np.mean(accel_vals)), 3) if accel_vals else None,
        "max_adv_accel_mean":  round(float(np.mean(adv_acc_vals)), 3) if adv_acc_vals else None,
        "adv_nuisance_rate":   round(nuisance_rate, 4),
        # Feasibility
        "mean_feasibility":    mean_feasibility,
        "difficulty_dist":     difficulty_dist,
        # Multi-adversary coordination
        "mean_adv_adv_collisions": mean_adv_adv_collisions,
        # Diversity: phi-space
        "phi_coverage":        round(phi_div.get("coverage", float("nan")), 4) if phi_div else None,
        "phi_entropy_norm":    round(phi_div.get("entropy_norm", float("nan")), 4) if phi_div else None,
        "diversity_score":     round(phi_div.get("diversity_score", float("nan")), 4) if phi_div else None,
        "phi_bins_occupied":   phi_div.get("bins_occupied"),
        # Diversity: trajectory
        "dispersion":          round(dispersion, 3) if np.isfinite(dispersion) else None,
        "frechet_mean":        round(frechet.get("frechet_mean", float("nan")), 3)
                               if np.isfinite(frechet.get("frechet_mean", float("nan"))) else None,
        "frechet_median":      round(frechet.get("frechet_median", float("nan")), 3)
                               if np.isfinite(frechet.get("frechet_median", float("nan"))) else None,
    }
    return result


def summarise_runs(
    runs: List[Dict[str, Any]],
    phi_cfg: PhiGridConfig = _DEFAULT_PHI_CFG,
    frechet_downsample_to: int = _FRECHET_DOWNSAMPLE,
) -> Dict[str, Dict[str, Any]]:
    """Group runs by (method, n_adversaries) and aggregate each group.

    The group key is formatted as "<method> (n=<N>)" so different adversary
    counts appear as distinct rows in the summary tables.
    """
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        method = run.get("method", "unknown")
        n_adv = run.get("n_adversaries", "?")
        key = f"{method} (n={n_adv})"
        by_group.setdefault(key, []).extend(run.get("rollouts", []))
    return {
        key: aggregate_method(rollouts, phi_cfg=phi_cfg, frechet_downsample_to=frechet_downsample_to)
        for key, rollouts in by_group.items()
    }


# Columns for the main effectiveness + realism table (flat numeric fields).
# Trajectory diversity (dispersion, frechet) go in a separate table since they
# require trace data and may be absent for older result files.
_MAIN_METRICS: List[Tuple[str, str]] = [
    ("p_collision",         "p_coll"),
    ("p_critical_adv",      "p_crit_adv"),   # adversary-caused - primary metric
    ("mean_ttci_adv_steps", "TTCI_adv"),     # adversary-caused TTCI - primary metric
    ("p_critical_any",      "p_crit_any"),   # includes background traffic (context)
    ("mean_ttci_any_steps", "TTCI_any"),
    ("mean_min_dist_adv",   "min_d_adv"),
    ("adv_nuisance_rate",   "nuisance"),
    ("mean_feasibility",    "feasibility"),
    ("mean_adv_adv_collisions", "adv_adv_coll"),
    ("ego_speed_mean",      "ego_spd"),
    ("ego_accel_min_mean",  "ego_acc_min"),
    ("max_adv_accel_mean",  "adv_acc_max"),
]

_DIVERSITY_METRICS: List[Tuple[str, str]] = [
    ("phi_coverage",       "phi_cov"),
    ("phi_entropy_norm",   "phi_ent"),
    ("diversity_score",    "div_score"),
    ("phi_bins_occupied",  "phi_bins"),
    ("dispersion",         "disp"),
    ("frechet_mean",       "frechet"),
    ("frechet_median",     "frechet_med"),
]

# Combined order used for CSV export (all columns)
_ALL_METRICS: List[Tuple[str, str]] = _MAIN_METRICS + _DIVERSITY_METRICS


def _fmt(v: Any) -> str:
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.3f}" if np.isfinite(v) else "--"
    if isinstance(v, int):
        return str(v)
    return str(v)


def to_latex(summary: Dict[str, Dict[str, Any]]) -> str:
    """Render effectiveness + realism table as LaTeX booktabs."""
    col_headers = " & ".join(["Method"] + [label for _, label in _MAIN_METRICS])
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(_MAIN_METRICS) + "}",
        r"\toprule",
        col_headers + r" \\",
        r"\midrule",
    ]
    for method, stats in sorted(summary.items()):
        vals = [method] + [_fmt(stats.get(key)) for key, _ in _MAIN_METRICS]
        lines.append(" & ".join(vals) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{BAAS benchmark: effectiveness and realism}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def to_latex_diversity(summary: Dict[str, Dict[str, Any]]) -> str:
    """Render diversity metrics table as LaTeX booktabs."""
    col_headers = " & ".join(["Method"] + [label for _, label in _DIVERSITY_METRICS])
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(_DIVERSITY_METRICS) + "}",
        r"\toprule",
        col_headers + r" \\",
        r"\midrule",
    ]
    for method, stats in sorted(summary.items()):
        vals = [method] + [_fmt(stats.get(key)) for key, _ in _DIVERSITY_METRICS]
        lines.append(" & ".join(vals) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{BAAS benchmark: diversity metrics}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def to_csv(summary: Dict[str, Dict[str, Any]]) -> str:
    """Render summary as CSV (all metrics)."""
    headers = ["method"] + [key for key, _ in _ALL_METRICS]
    rows = [",".join(headers)]
    for method, stats in sorted(summary.items()):
        vals = [method] + [str(stats.get(key, "")) for key, _ in _ALL_METRICS]
        rows.append(",".join(vals))
    return "\n".join(rows)


def summarise_multiseed(
    per_seed_summaries: List[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-seed summaries into mean ± std statistics.

    Each element of per_seed_summaries is the output of summarise_runs() for one
    seed. Returns a dict keyed by method with "_mean" and "_std" variants for
    every numeric metric. Methods absent in a given seed are skipped for that seed.

    Usage:
        seed_summaries = [summarise_runs(load_results_dir(d)) for d in seed_dirs]
        multi = summarise_multiseed(seed_summaries)
        save_summary(multi, output_dir, tag="_multiseed")
    """
    all_methods: set = set()
    for s in per_seed_summaries:
        all_methods.update(s.keys())

    result: Dict[str, Dict[str, Any]] = {}
    for method in sorted(all_methods):
        # Collect per-seed stats dicts for this method (skip seeds where absent)
        seed_stats: List[Dict[str, Any]] = [
            s[method] for s in per_seed_summaries if method in s and s[method]
        ]
        if not seed_stats:
            continue

        numeric_keys = [
            k for k in seed_stats[0]
            if isinstance(seed_stats[0][k], (int, float)) and seed_stats[0][k] is not None
        ]

        combined: Dict[str, Any] = {"n_seeds": len(seed_stats)}
        for key in numeric_keys:
            vals = [s[key] for s in seed_stats if s.get(key) is not None and np.isfinite(float(s[key]))]
            if not vals:
                combined[f"{key}_mean"] = None
                combined[f"{key}_std"] = None
            else:
                combined[f"{key}_mean"] = round(float(np.mean(vals)), 4)
                combined[f"{key}_std"] = round(float(np.std(vals, ddof=0)), 4)

        # Pass through non-numeric fields from the first seed (e.g. difficulty_dist)
        for key, val in seed_stats[0].items():
            if key not in numeric_keys:
                combined.setdefault(key, val)

        result[method] = combined

    return result


def save_summary(
    summary: Dict[str, Dict[str, Any]],
    output_dir: Path,
    tag: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"summary{tag}"
    (output_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / f"{stem}.csv").write_text(to_csv(summary), encoding="utf-8")
    (output_dir / f"{stem}.tex").write_text(to_latex(summary), encoding="utf-8")
    (output_dir / f"{stem}_diversity.tex").write_text(to_latex_diversity(summary), encoding="utf-8")
    logger.info("Summary saved to %s  (stem=%s)", output_dir, stem)
