"""Command-line entry points for the HRP reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hrp_lab.config import load_config
from hrp_lab.pipeline import (
    load_research_data,
    make_output_dir,
    run_backtest_phase,
    run_benchmark_phase,
    run_monte_carlo_phase,
    run_reproduction,
    save_backtests,
    summarize_backtests,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hrp-lab")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-data", "monte-carlo", "benchmark-tree", "backtest", "reproduce"):
        command = subcommands.add_parser(name)
        command.add_argument("--config", default="configs/crsp_reproduction.yaml")
        command.add_argument("--output")
        if name in ("monte-carlo", "reproduce"):
            command.add_argument("--trials", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "reproduce":
        output = run_reproduction(
            args.config,
            output_dir=args.output,
            monte_carlo_trials=args.trials,
        )
        print(output)
        return 0

    config = load_config(args.config)
    output = make_output_dir(args.output)
    loaded = load_research_data(config)
    if args.command == "validate-data":
        write_json(loaded.manifest, output / "data_manifest.json")
        print(json.dumps(loaded.manifest, indent=2, default=str))
        return 0
    if args.command == "monte-carlo":
        if args.trials is not None:
            config["monte_carlo"]["replications"] = args.trials
        result = run_monte_carlo_phase(loaded.panel.returns, config)
        result.trials.to_csv(output / "monte_carlo_trials.csv.gz", index=False, compression="gzip")
        result.summary.to_csv(output / "monte_carlo_summary.csv")
        print(result.summary.to_string())
        return 0
    if args.command == "benchmark-tree":
        benchmark = run_benchmark_phase(loaded.panel.returns, config)
        write_json(benchmark, output / "tree_benchmark.json")
        print(json.dumps(benchmark["actual_crsp"], indent=2, default=str))
        return 0
    if args.command == "backtest":
        strategies = run_backtest_phase(loaded.panel.returns, config)
        save_backtests(strategies, output)
        summary = summarize_backtests(strategies, config)
        write_json(summary, output / "metrics.json")
        print(json.dumps(summary, indent=2, default=str))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
