"""Format-lever probe runner: reader x format, first-party providers locked.

Does evidence format alone move readers, and is the effect universal (same best
format across models) or a model trait (crossover)? --dry-run costs nothing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from analysis.format_lever import FORMATS, FormatRow, build_prompt, is_correct, load_rows
from reader.client import DEFAULT_OPENROUTER_BASE_URL, generate_answer

DEFAULT_SLICE = Path("dataset-slices/longmemeval_bm25_top20_v1.jsonl")
# model -> first-party OpenRouter provider slug (allow_fallbacks disabled).
PROVIDER_LOCK = {
    "xiaomi/mimo-v2.5": "xiaomi",
    "xiaomi/mimo-v2.5-pro": "xiaomi",
    "deepseek/deepseek-v4-flash": "deepseek",
    "minimax/minimax-m3": "minimax",
}
DEFAULT_READERS = tuple(PROVIDER_LOCK)
READER_MAX_TOKENS = 512  # headroom if a model ignores the reasoning-disable flag
READER_TIMEOUT_S = 120.0
DISABLE_REASONING = {"enabled": False}  # probe direct-answer format sensitivity
DEFAULT_PARALLELISM = 32
CHARS_PER_TOKEN = 4
USD_PER_MILLION_TOKENS = 0.30


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--slice", type=Path, default=DEFAULT_SLICE)
    parser.add_argument("--readers", nargs="+", default=list(DEFAULT_READERS))
    parser.add_argument("--limit", type=int, default=100, help="0 = all rows")
    parser.add_argument("--parallelism", type=int, default=DEFAULT_PARALLELISM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")


def _provider_routing(reader: str) -> dict | None:
    slug = PROVIDER_LOCK.get(reader)
    return {"order": [slug], "allow_fallbacks": False} if slug else None


def _plan(rows: list[FormatRow], readers: list[str]) -> list[dict]:
    return [
        {
            "example_id": row.example_id,
            "question_type": row.question_type,
            "format": fmt,
            "reader": reader,
            "prompt": build_prompt(row, fmt),
        }
        for row in rows
        for fmt in FORMATS
        for reader in readers
    ]


def _dry_run(plan: list[dict]) -> None:
    prompt_tokens = sum(len(item["prompt"]) // CHARS_PER_TOKEN for item in plan)
    est = prompt_tokens + len(plan) * READER_MAX_TOKENS
    print(f"[dry-run] {len(plan)} calls, ~{est:,} tokens, ~${est/1_000_000*USD_PER_MILLION_TOKENS:.3f}")


def _execute_one(item: dict) -> dict:
    result = generate_answer(
        provider="openrouter",
        base_url=DEFAULT_OPENROUTER_BASE_URL,
        model=item["reader"],
        prompt=item["prompt"],
        temperature=0.0,
        timeout_s=READER_TIMEOUT_S,
        max_tokens=READER_MAX_TOKENS,
        provider_routing=_provider_routing(item["reader"]),
        reasoning=DISABLE_REASONING,
    )
    return {**item, "raw_answer": result["raw_answer"], "success": result["success"]}


def _run_live(plan: list[dict], rows: list[FormatRow], parallelism: int) -> list[dict]:
    row_by_id = {row.example_id: row for row in rows}
    answers: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = [pool.submit(_execute_one, item) for item in plan]
        for done in concurrent.futures.as_completed(futures):
            record = done.result()
            correct = is_correct(row_by_id[record["example_id"]], record["raw_answer"])
            answers.append({**{k: v for k, v in record.items() if k != "prompt"}, "correct": correct})
    return answers


def _cells(answers: list[dict]) -> dict[str, dict[str, dict]]:
    cells: dict[str, dict[str, dict]] = {}
    for record in answers:
        reader = cells.setdefault(record["reader"], {})
        cell = reader.setdefault(record["format"], {"n": 0, "correct": 0})
        cell["n"] += 1
        cell["correct"] += int(record["correct"])
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Format-lever probe")
    add_common_args(parser)
    args = parser.parse_args()

    rows = load_rows(args.slice)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[data] {len(rows)} rows x {len(FORMATS)} formats x {len(args.readers)} readers")
    plan = _plan(rows, args.readers)

    if args.dry_run:
        _dry_run(plan)
        return

    answers = _run_live(plan, rows, args.parallelism)
    cells = _cells(answers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "answers.json").write_text(json.dumps(answers, indent=2))
    (args.output_dir / "cells.json").write_text(json.dumps(cells, indent=2))
    fails = sum(1 for a in answers if not a["success"])
    print(f"[done] {len(answers)} calls, {fails} failed -> {args.output_dir}/cells.json")


if __name__ == "__main__":
    main()
