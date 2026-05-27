"""Minimal CLI for human feedback on rendered scenario replays.

Shows scenario metadata and asks the operator to label each one:
  good / bad / interesting / skip

Labels are written back to ScenarioRecord.label.
Used for RLHF-style selection of high-quality adversarial scenarios for
ego retraining.

Usage:
    python -m baas.scripts.run_labeller --dataset runs/scenarios.json
"""
from __future__ import annotations

from typing import List

from baas.data.schema import ScenarioRecord

VALID_LABELS = {"good", "bad", "interesting", "skip"}


def label_scenarios_cli(records: List[ScenarioRecord]) -> List[ScenarioRecord]:
    """Interactive CLI labelling loop. Modifies records in-place."""
    for i, rec in enumerate(records):
        print(
            f"\n[{i+1}/{len(records)}]  method={rec.method}  "
            f"phi={rec.phi}  collision={rec.outcome.ego_collision}"
        )
        while True:
            choice = input("Label [good/bad/interesting/skip]: ").strip().lower()
            if choice in VALID_LABELS:
                break
            print(f"  Invalid.  Options: {sorted(VALID_LABELS)}")
        if choice != "skip":
            rec.label = choice
    return records
