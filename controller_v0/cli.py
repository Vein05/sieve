"""CLI for the rule_v0 controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from controller_v0.logic import run_example
from controller_v0.reporting import (
    default_output_path,
    evaluate,
    load_jsonl,
    write_jsonl,
    write_summary_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rule_v0 controller on a JSONL slice.")
    parser.add_argument(
        "--slice",
        default="dataset-slices/kill_test_v1.jsonl",
        help="Path to the JSONL slice (default: dataset-slices/kill_test_v1.jsonl)",
    )
    parser.add_argument(
        "--config",
        default="configs/controller_v0.json",
        help="Path to the controller config (default: configs/controller_v0.json)",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Where to write per-example controller outputs (default: results/controller_runs/<timestamp>-<slice>-rule_v0.jsonl)",
    )
    parser.add_argument(
        "--summary-md",
        default=None,
        help="Optional path for a markdown summary",
    )
    parser.add_argument(
        "--print-json-summary",
        action="store_true",
        help="Print the evaluation summary as JSON instead of human-readable text",
    )
    args = parser.parse_args()

    slice_path = Path(args.slice)
    config_path = Path(args.config)
    if not slice_path.exists():
        raise SystemExit(f"ERROR: slice file not found: {slice_path}")
    if not config_path.exists():
        raise SystemExit(f"ERROR: config file not found: {config_path}")

    rows = load_jsonl(slice_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_path = Path(args.output_jsonl) if args.output_jsonl else default_output_path(slice_path)

    results = [run_example(row, config) for row in rows]
    summary = evaluate(results)

    write_jsonl(output_path, results)
    if args.summary_md:
        write_summary_markdown(
            Path(args.summary_md),
            slice_path=slice_path,
            config_path=config_path,
            output_path=output_path,
            summary=summary,
        )

    if args.print_json_summary:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Wrote per-example output to: {output_path}")
        print(f"Rows: {summary['rows']}")
        print(f"Action accuracy: {summary['action_accuracy']}")
        print(f"Exact selected-memory match rate: {summary['exact_selected_match_rate']}")
        print(
            "False suppressions on positive controls: "
            f"{summary['false_suppression_positive_control']}"
        )
        print(f"Average selected count: {summary['avg_selected_count']}")
        print(f"Average selected token count: {summary['avg_selected_token_count']}")
        print("By harm type:")
        for harm_type, stats in summary["by_harm_type"].items():
            print(
                f"  {harm_type}: action_accuracy={stats['action_accuracy']}, "
                f"exact_selected_match_rate={stats['exact_selected_match_rate']}, rows={stats['rows']}"
            )
