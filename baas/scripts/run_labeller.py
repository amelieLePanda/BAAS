"""CLI entry point for human feedback labelling on adversarial scenarios.

Loads a scenario dataset JSON, shows metadata for each scenario, and
prompts for a label. Saves the labelled dataset back to disk.

Usage:
    python baas/scripts/run_labeller.py --dataset runs/scenarios.json
    python baas/scripts/run_labeller.py --dataset runs/scenarios.json --output runs/scenarios_labelled.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_records(path: Path) -> list:
    """Load ScenarioRecords from a JSON file as plain dicts."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("scenarios", [])
    return raw


def _save_records(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scenarios": records}, indent=2), encoding="utf-8")
    logger.info("Saved %d labelled records to %s", len(records), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive scenario labeller")
    parser.add_argument("--dataset", required=True, type=Path,
                        help="scenarios.json produced by data/collector.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="output path (defaults to overwriting --dataset)")
    args = parser.parse_args()

    records = _load_records(args.dataset)
    if not records:
        logger.warning("No records found in %s", args.dataset)
        return

    logger.info("Loaded %d scenarios from %s", len(records), args.dataset)
    print(f"\nStarting labelling session: {len(records)} scenarios.\n"
          "  Labels: good | bad | interesting | skip\n")

    for i, rec in enumerate(records):
        method = rec.get("method", "?")
        phi = rec.get("phi")
        collision = rec.get("outcome", {}).get("ego_collision", "?")
        critical = rec.get("outcome", {}).get("critical_incident", "?")
        difficulty = rec.get("outcome", {}).get("difficulty_label", "?")
        existing = rec.get("label")

        label_hint = f"  current={existing}" if existing else ""
        print(
            f"[{i+1}/{len(records)}]  method={method}  phi={phi}  "
            f"collision={collision}  critical={critical}  difficulty={difficulty}{label_hint}"
        )

        while True:
            choice = input("  Label [good/bad/interesting/skip]: ").strip().lower()
            if choice in {"good", "bad", "interesting", "skip"}:
                break
            print("  Invalid. Options: good, bad, interesting, skip")

        if choice != "skip":
            rec["label"] = choice

    out_path = args.output or args.dataset
    _save_records(records, out_path)
    print(f"\nDone. Labelled dataset saved to: {out_path}")


if __name__ == "__main__":
    main()
