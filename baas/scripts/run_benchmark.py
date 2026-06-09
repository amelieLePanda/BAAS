"""CLI entry point for benchmark evaluation and diversity reporting.

Loads a result directory produced by any search method, computes diversity
metrics, and writes summary tables (JSON / CSV / LaTeX).

Usage:
    # Single-run summary (one result file per method):
    python baas/scripts/run_benchmark.py summarise runs/ --output runs/summary

    # Multi-seed summary: aggregate named method across per-seed subdirs,
    # treat all other methods as single runs:
    python baas/scripts/run_benchmark.py summarise runs/ --output runs/summary --multiseed-method king_light --seed-dirs runs/king_light/run_king_light_seed0 runs/king_light/run_king_light_seed1 runs/king_light/run_king_light_seed2 runs/king_light/run_king_light_seed3 runs/king_light/run_king_light_seed4
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _cmd_summarise(args: argparse.Namespace) -> None:
    from baas.evaluation.summarise import (
        load_results_dir, load_result_json, save_summary,
        summarise_runs, summarise_multiseed,
    )

    multiseed_method: str = getattr(args, "multiseed_method", None) or ""
    seed_dirs = [Path(d) for d in (getattr(args, "seed_dirs", None) or [])]

    # Load all runs, filtering out the multiseed method's results
    all_runs = load_results_dir(args.results_dir)
    if multiseed_method:
        single_runs = [r for r in all_runs if r.get("method") != multiseed_method]
    else:
        single_runs = all_runs

    summary = summarise_runs(single_runs)

    # Aggregate the multiseed method separately
    if multiseed_method and seed_dirs:
        per_seed = []
        for d in seed_dirs:
            p = d / "results.json"
            if not p.exists():
                logger.warning("Seed dir missing results.json: %s", d)
                continue
            data = load_result_json(p)
            if data:
                per_seed.append(summarise_runs([data]))
        if per_seed:
            ms = summarise_multiseed(per_seed)
            summary.update(ms)  # keys match summarise_runs format, e.g. "king_light (n=1)"
            logger.info("Multi-seed summary for %s: %d seeds", multiseed_method, len(per_seed))
        else:
            logger.warning("No seed dirs found for multiseed method %s", multiseed_method)

    tag = f"_{args.tag}" if getattr(args, "tag", "") else ""
    save_summary(summary, args.output, tag=tag)
    logger.info("Summary written to %s", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="BAAS benchmark evaluation and summary")
    sub = parser.add_subparsers(dest="cmd")

    s = sub.add_parser("summarise", help="Aggregate results from a directory")
    s.add_argument("results_dir", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--tag", type=str, default="")
    s.add_argument("--multiseed-method", type=str, default="",
                   help="Method name to aggregate via summarise_multiseed (mean±std)")
    s.add_argument("--seed-dirs", nargs="+", default=[],
                   help="Per-seed result directories for the multiseed method")

    args = parser.parse_args()
    if args.cmd == "summarise":
        _cmd_summarise(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
