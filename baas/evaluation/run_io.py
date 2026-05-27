"""Shared utilities for writing run metadata and results.

Every search method calls write_run_meta() at the start of a run and
update_run_status() at the end. This gives every run directory a consistent
run.json that is easy to scan or load for comparison.

MLflow integration is optional. Set MLFLOW_TRACKING_URI (e.g. "http://localhost:5000"
or a local path like "file:./mlruns") and install mlflow to enable it. If mlflow
is not installed or the env var is not set, all logging is silently skipped.
Each call to write_run_meta() starts an MLflow run; update_run_status() logs
final metrics and ends it. The run.json files remain the ground truth regardless.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_MADS_VERSION = "0.1.0"

# Module-level MLflow run ID so update_run_status can end the right run.
# Keyed by str(output_dir) to support multiple concurrent runs in tests.
_mlflow_run_ids: Dict[str, str] = {}


def _git_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _serialise(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _mlflow_available() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI")) and _try_import_mlflow()


def _try_import_mlflow() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def _flatten_params(d: Any, prefix: str = "") -> Dict[str, str]:
    """Flatten a nested dict to dot-separated string params for MLflow."""
    out: Dict[str, str] = {}
    if not isinstance(d, dict):
        return {prefix: str(d)} if prefix else {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_params(v, key))
        else:
            out[key] = str(v)
    return out


def write_run_meta(
    output_dir: Path,
    *,
    method: str,
    config: Any,
    seed: int,
    n_adversaries: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write run.json to output_dir at the start of a run.

    Call this before the search loop so a crash mid-run still leaves a
    traceable metadata file. If MLflow is configured, also starts an MLflow run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = _serialise(config) if not isinstance(config, dict) else config

    meta: Dict[str, Any] = {
        "baas_version": _MADS_VERSION,
        "method": method,
        "timestamp_start": datetime.now(timezone.utc).isoformat(),
        "timestamp_end": None,
        "status": "running",
        "seed": seed,
        "n_adversaries": n_adversaries,
        "git_hash": _git_hash(),
        "config": cfg_dict,
    }
    if extra:
        meta.update(extra)

    path = output_dir / "run.json"
    path.write_text(json.dumps(meta, indent=2, default=_serialise), encoding="utf-8")
    logger.info("Run metadata written to %s", path)

    # MLflow: start run and log params
    if _mlflow_available():
        try:
            import mlflow
            run_name = f"{method}_s{seed}_n{n_adversaries}"
            mlflow.set_experiment(f"MADS/{method}")
            mlf_run = mlflow.start_run(run_name=run_name)
            _mlflow_run_ids[str(output_dir)] = mlf_run.info.run_id
            mlflow.set_tags({
                "method": method,
                "n_adversaries": str(n_adversaries),
                "seed": str(seed),
                "git_hash": meta.get("git_hash") or "",
                "output_dir": str(output_dir),
            })
            params = _flatten_params(cfg_dict)
            # MLflow param values are capped at 500 chars; truncate safely
            mlflow.log_params({k: v[:500] for k, v in params.items()})
        except Exception as exc:
            logger.debug("MLflow logging skipped: %s", exc)

    return path


def update_run_status(
    output_dir: Path,
    *,
    status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Update run.json status and end timestamp in-place.

    If MLflow is configured, logs any numeric values in extra as metrics
    and ends the active run.
    """
    path = Path(output_dir) / "run.json"
    if not path.exists():
        return
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["status"] = status
        meta["timestamp_end"] = datetime.now(timezone.utc).isoformat()
        if extra:
            meta.update(extra)
        path.write_text(json.dumps(meta, indent=2, default=_serialise), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not update run.json: %s", exc)

    # MLflow: log final metrics and end run
    if _mlflow_available() and str(output_dir) in _mlflow_run_ids:
        try:
            import mlflow
            run_id = _mlflow_run_ids.pop(str(output_dir))
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tag("status", status)
                if extra:
                    metrics = {k: float(v) for k, v in extra.items()
                               if isinstance(v, (int, float)) and not isinstance(v, bool)}
                    if metrics:
                        mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(path), artifact_path="run_meta")
        except Exception as exc:
            logger.debug("MLflow end-run logging skipped: %s", exc)


def save_snapshot(
    output_dir: Path,
    data: Any,
    *,
    name: str,
) -> Path:
    """Write a JSON snapshot to output_dir/snapshots/name.json."""
    snap_dir = Path(output_dir) / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = snap_dir / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=_serialise), encoding="utf-8")
    return path


def sha1_file(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()
