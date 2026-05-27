"""Generate and save the benchmark rollout specs JSON.

Rollout specs are the frozen set of (seed, perturbation) pairs used to
evaluate ALL methods on equal footing.  Generate once per experiment config
and n_adversaries value, then reuse for every run_eval.py call.

The SHA-1 of the specs is embedded in every results JSON so evaluations can
be traced back to the exact spec set that produced them.

Usage:
    python baas/scripts/run_generate_specs.py \\
        --config configs/benchmark_v1.yaml \\
        --n-adversaries 2 \\
        --output runs/rollout_specs_n2.json

    python baas/scripts/run_generate_specs.py \\
        --config configs/benchmark_v1.yaml \\
        --n-adversaries 4 \\
        --output runs/rollout_specs_n4.json
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark rollout specs")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--n-adversaries", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path,
                        help="Output path, e.g. runs/rollout_specs_n2.json")
    args = parser.parse_args()

    import yaml
    from baas.evaluation.benchmark import make_rollout_specs_from_config, save_rollout_specs

    cfg = yaml.safe_load(args.config.read_text())
    specs = make_rollout_specs_from_config(cfg, n_adversaries=args.n_adversaries)
    sha = save_rollout_specs(specs, args.output)

    logger.info(
        "Saved %d rollout specs (n_adv=%d) → %s  sha1=%s",
        len(specs), args.n_adversaries, args.output, sha[:12],
    )


if __name__ == "__main__":
    main()
