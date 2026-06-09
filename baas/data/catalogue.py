"""Build and extend the scenario catalogue (one row per evaluated rollout).

Each row is a flat dict suitable for JSON serialisation. The catalogue is a
JSON array that can be appended to incrementally as new runs finish.

Row schema
----------
scenario_id         str    e.g. "map_elites_s0_007"
method              str
opt_seed            int    seed index used during optimisation (0 for single-run methods)
rollout_index       int
env_seed            int
critical_incident   bool
ego_collision       bool
feasibility         float | null
difficulty_label    str   | null
phi_dist            float | null   min_dist_ego_adv in metres (raw, not normalised)
phi_ttci            float | null   TTCI_adv in steps (raw, not normalised)
min_dist_ego_adv    float | null
ttci_adv_steps      int   | null
max_adv_jerk        float | null
results_path        str    relative path from catalogue dir to results.json
artefact_reference  str    relative path from catalogue dir to artefact file, or ""
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _scenario_id(method: str, opt_seed: int, rollout_index: int) -> str:
    return f"{method}_s{opt_seed}_{rollout_index:03d}"


def _rows_for_run(
    results_path: Path,
    opt_seed: int,
    artefact_path: Optional[Path],
    catalogue_dir: Path,
) -> List[Dict[str, Any]]:
    """Parse one results.json and return one catalogue row per rollout."""
    data = json.loads(results_path.read_text(encoding="utf-8"))
    method = data["method"]
    artefact_ref = (
        str(artefact_path.relative_to(catalogue_dir)) if artefact_path else ""
    )
    results_ref = str(results_path.relative_to(catalogue_dir))

    rows = []
    for r in data["rollouts"]:
        m = r["metrics"]
        phi = m.get("phi")
        rollout_index = int(r["rollout_index"])
        row: Dict[str, Any] = {
            "scenario_id": _scenario_id(method, opt_seed, rollout_index),
            "method": method,
            "opt_seed": opt_seed,
            "rollout_index": rollout_index,
            "env_seed": int(r["env_seed"]),
            "critical_incident": bool(m["critical_incident"]),
            "ego_collision": bool(m["ego_collision"]),
            "feasibility": m.get("feasibility"),
            "difficulty_label": m.get("difficulty_label"),
            "phi_dist": phi[0] if phi else None,
            "phi_ttci": phi[1] if phi else None,
            "min_dist_ego_adv": m.get("min_dist_ego_adv"),
            "ttci_adv_steps": m.get("time_to_critical_incident_adv_steps"),
            "max_adv_jerk": m.get("max_adv_jerk"),
            "results_path": results_ref,
            "artefact_reference": artefact_ref,
        }
        rows.append(row)
    return rows


def append_method(
    catalogue_path: Path,
    run_dirs: List[Path],
    artefact_name: str,
    *,
    opt_seeds: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Append rows for one method (possibly multiple seed runs) to the catalogue.

    Parameters
    ----------
    catalogue_path:
        Target catalogue JSON file. Created if it does not exist.
    run_dirs:
        List of run directories to process (one per seed for multi-seed methods).
    artefact_name:
        Filename of the artefact inside each run dir (e.g. "archive.json").
        Pass "" if the method has no artefact file.
    opt_seeds:
        Seed indices aligned with run_dirs. Defaults to [0, 1, ...].
    """
    if opt_seeds is None:
        opt_seeds = list(range(len(run_dirs)))

    catalogue_dir = catalogue_path.parent
    catalogue_dir.mkdir(parents=True, exist_ok=True)

    existing: List[Dict[str, Any]] = []
    if catalogue_path.exists():
        existing = json.loads(catalogue_path.read_text(encoding="utf-8"))

    existing_ids = {row["scenario_id"] for row in existing}
    new_rows: List[Dict[str, Any]] = []

    for run_dir, seed in zip(run_dirs, opt_seeds):
        results_path = run_dir / "results.json"
        if not results_path.exists():
            logger.warning("results.json not found: %s", results_path)
            continue
        artefact_path = (run_dir / artefact_name) if artefact_name else None
        if artefact_path and not artefact_path.exists():
            logger.warning("Artefact not found, leaving reference empty: %s", artefact_path)
            artefact_path = None

        rows = _rows_for_run(results_path, seed, artefact_path, catalogue_dir)
        added = [r for r in rows if r["scenario_id"] not in existing_ids]
        new_rows.extend(added)
        existing_ids.update(r["scenario_id"] for r in added)
        logger.info("  seed %d: %d rows (%d new)", seed, len(rows), len(added))

    all_rows = existing + new_rows
    catalogue_path.write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Catalogue: %d total rows -> %s", len(all_rows), catalogue_path)
    return new_rows


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Append a method's runs to the scenario catalogue")
    parser.add_argument("--catalogue", required=True, type=Path, help="catalogue JSON output path")
    parser.add_argument("--run-dirs", required=True, nargs="+", type=Path)
    parser.add_argument("--artefact-name", default="", help="artefact filename inside each run dir")
    parser.add_argument("--opt-seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    new = append_method(
        args.catalogue,
        args.run_dirs,
        args.artefact_name,
        opt_seeds=args.opt_seeds,
    )
    print(f"Added {len(new)} new rows.")
    if new:
        print("Sample row:")
        print(json.dumps(new[0], indent=2))
