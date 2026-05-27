"""CLI entry point for benchmark evaluation and diversity reporting.

Loads a result directory produced by any search method, computes diversity
metrics, and writes summary tables (JSON / CSV / LaTeX).

Usage:
    # Summarise multiple methods at once:
    python baas/scripts/run_benchmark.py summarise runs/ --output runs/summary
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _cmd_summarise(args: argparse.Namespace) -> None:
    from baas.evaluation.summarise import load_results_dir, save_summary, summarise_runs

    runs = load_results_dir(args.results_dir)
    summary = summarise_runs(runs)
    tag = f"_{args.tag}" if getattr(args, "tag", "") else ""
    save_summary(summary, args.output, tag=tag)
    logger.info("Summary written to %s", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="MADS benchmark evaluation and summary")
    sub = parser.add_subparsers(dest="cmd")

    s = sub.add_parser("summarise", help="Aggregate results from a directory")
    s.add_argument("results_dir", type=Path)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--tag", type=str, default="",
                   help="Optional label appended to output filenames, e.g. 'mappo_n2_vs_maddpg_n2'")

    args = parser.parse_args()
    if args.cmd == "summarise":
        _cmd_summarise(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
