"""I/O and reporting helpers for the rule_v0 controller."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def iso_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"ERROR: {path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"ERROR: {path}:{lineno}: row is not a JSON object")
            rows.append(row)
    return rows


def evaluate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    action_matches = 0
    exact_selected_matches = 0
    false_suppression_positive_control = 0
    avg_selected_count = 0.0
    avg_selected_token_count = 0.0
    by_harm_type: dict[str, Counter[str]] = defaultdict(Counter)
    mismatches: list[dict[str, Any]] = []

    for result in results:
        harm_type = str(result["harm_type"])
        gold = result.get("gold_decision", {})
        gold_action = gold.get("action")
        gold_selected = list(gold.get("selected_memory_ids", []))
        predicted_action = result["decision_summary"]["predicted_action"]
        predicted_selected = result["decision_summary"]["predicted_selected_memory_ids"]

        avg_selected_count += result["decision_summary"]["selected_count"]
        avg_selected_token_count += result["decision_summary"]["selected_token_count"]
        by_harm_type[harm_type]["rows"] += 1

        if predicted_action == gold_action:
            action_matches += 1
            by_harm_type[harm_type]["action_matches"] += 1

        if predicted_selected == gold_selected:
            exact_selected_matches += 1
            by_harm_type[harm_type]["exact_selected_matches"] += 1
        else:
            mismatches.append(
                {
                    "example_id": result["example_id"],
                    "harm_type": harm_type,
                    "gold_selected": gold_selected,
                    "predicted_selected": predicted_selected,
                    "predicted_action": predicted_action,
                    "gold_action": gold_action,
                }
            )

        if harm_type == "positive_control" and not predicted_selected:
            false_suppression_positive_control += 1

    return {
        "rows": total,
        "action_accuracy": round(action_matches / total, 3) if total else 0.0,
        "exact_selected_match_rate": round(exact_selected_matches / total, 3) if total else 0.0,
        "false_suppression_positive_control": false_suppression_positive_control,
        "avg_selected_count": round(avg_selected_count / total, 3) if total else 0.0,
        "avg_selected_token_count": round(avg_selected_token_count / total, 3) if total else 0.0,
        "by_harm_type": {
            harm_type: {
                "rows": counts["rows"],
                "action_accuracy": round(counts["action_matches"] / counts["rows"], 3),
                "exact_selected_match_rate": round(
                    counts["exact_selected_matches"] / counts["rows"],
                    3,
                ),
            }
            for harm_type, counts in sorted(by_harm_type.items())
        },
        "mismatches": mismatches,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def write_summary_markdown(
    path: Path,
    *,
    slice_path: Path,
    config_path: Path,
    output_path: Path,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Controller V0 Slice Run",
        "",
        f"- date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "- objective: execute the first rule-based eligibility controller on the curated kill-test slice",
        f"- command: `scripts/run_controller_v0.py --slice {slice_path} --config {config_path} --output-jsonl {output_path}`",
        f"- dataset or slice: `{slice_path}`",
        f"- controller config: `{config_path}`",
        f"- raw output: `{output_path}`",
        f"- rows: {summary['rows']}",
        f"- action accuracy: {summary['action_accuracy']}",
        f"- exact selected-memory match rate: {summary['exact_selected_match_rate']}",
        f"- false suppressions on positive controls: {summary['false_suppression_positive_control']}",
        f"- average selected count: {summary['avg_selected_count']}",
        f"- average selected token count: {summary['avg_selected_token_count']}",
        "",
        "## By Harm Type",
        "",
    ]
    for harm_type, stats in summary["by_harm_type"].items():
        lines.append(
            f"- `{harm_type}`: action_accuracy={stats['action_accuracy']}, "
            f"exact_selected_match_rate={stats['exact_selected_match_rate']}, rows={stats['rows']}"
        )

    lines.extend(["", "## Mismatches", ""])
    if summary["mismatches"]:
        for mismatch in summary["mismatches"]:
            lines.append(
                f"- `{mismatch['example_id']}` ({mismatch['harm_type']}): "
                f"gold_selected={mismatch['gold_selected']}, "
                f"predicted_selected={mismatch['predicted_selected']}, "
                f"gold_action={mismatch['gold_action']}, "
                f"predicted_action={mismatch['predicted_action']}"
            )
    else:
        lines.append("- none")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_path(slice_path: Path) -> Path:
    stem = slice_path.stem
    return Path("results/controller_runs") / f"{iso_timestamp()}-{stem}-rule_v0.jsonl"
