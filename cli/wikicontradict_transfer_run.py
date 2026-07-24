"""WikiContradict transfer runner (Gate 1): 2x2 render x prompt, cheap readers.

Decides whether conflict-marked read-time rendering beats conflict-aware
prompting off BEAM. --dry-run estimates call count + cost with zero LLM calls;
the live run fans out reader calls and scores them deterministically.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from dataclasses import asdict
from pathlib import Path

from analysis.wikicontradict import (
    ContradictionRow,
    build_prompt,
    condition_cells,
    filter_valid,
    load_rows,
    score_response,
)
from reader.client import DEFAULT_OPENROUTER_BASE_URL, generate_answer

DEFAULT_CSV = Path("data/wikicontradict/WikiContradict_dataset_v1_rag_qa.csv")
DEFAULT_READERS = (
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-7b-instruct",
)
READER_MAX_TOKENS = 512
READER_TIMEOUT_S = 180.0
DEFAULT_PARALLELISM = 16
CHARS_PER_TOKEN = 4  # coarse estimate; only used by --dry-run
USD_PER_MILLION_TOKENS = 0.05  # ballpark for cheap open readers on OpenRouter


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--readers", nargs="+", default=list(DEFAULT_READERS))
    parser.add_argument("--limit", type=int, default=0, help="0 = all valid rows")
    parser.add_argument("--parallelism", type=int, default=DEFAULT_PARALLELISM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")


def _select_rows(csv_path: Path, limit: int) -> list[ContradictionRow]:
    valid, dropped = filter_valid(load_rows(csv_path))
    print(f"[data] {len(valid)} valid rows, {len(dropped)} dropped by validity gate")
    return valid[:limit] if limit > 0 else valid


def _plan(rows: list[ContradictionRow], readers: list[str]) -> list[dict]:
    plan = []
    for row in rows:
        for render_style, prompt_style in condition_cells():
            prompt = build_prompt(row, render_style, prompt_style)
            for reader in readers:
                plan.append(
                    {
                        "question_id": row.question_id,
                        "contradict_type": row.contradict_type,
                        "render": render_style,
                        "prompt_style": prompt_style,
                        "reader": reader,
                        "prompt": prompt,
                    }
                )
    return plan


def _dry_run(plan: list[dict]) -> None:
    prompt_tokens = sum(len(item["prompt"]) // CHARS_PER_TOKEN for item in plan)
    est_tokens = prompt_tokens + len(plan) * READER_MAX_TOKENS
    cost = est_tokens / 1_000_000 * USD_PER_MILLION_TOKENS
    print(f"[dry-run] {len(plan)} reader calls")
    print(f"[dry-run] ~{est_tokens:,} tokens (prompt+completion)")
    print(f"[dry-run] ~${cost:.3f} at ${USD_PER_MILLION_TOKENS}/M tokens")


def _execute_one(item: dict) -> dict:
    result = generate_answer(
        provider="openrouter",
        base_url=DEFAULT_OPENROUTER_BASE_URL,
        model=item["reader"],
        prompt=item["prompt"],
        temperature=0.0,
        timeout_s=READER_TIMEOUT_S,
        max_tokens=READER_MAX_TOKENS,
    )
    return {**item, "raw_answer": result["raw_answer"], "success": result["success"]}


def _run_live(plan: list[dict], rows: list[ContradictionRow], parallelism: int) -> list[dict]:
    row_by_id = {row.question_id: row for row in rows}
    answers: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = {pool.submit(_execute_one, item): item for item in plan}
        for done in concurrent.futures.as_completed(futures):
            record = done.result()
            row = row_by_id[record["question_id"]]
            score = score_response(row, record["raw_answer"])
            answers.append({**{k: v for k, v in record.items() if k != "prompt"}, **asdict(score)})
    return answers


def _aggregate(answers: list[dict]) -> dict:
    cells: dict[str, dict] = {}
    for record in answers:
        key = f"{record['render']}+{record['prompt_style']}|{record['reader']}"
        cell = cells.setdefault(key, {"n": 0, "det_correct": 0, "both_present": 0, "judge_candidate": 0})
        cell["n"] += 1
        cell["det_correct"] += int(record["det_correct"])
        cell["both_present"] += int(record["both_present"])
        cell["judge_candidate"] += int(record["judge_candidate"])
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="WikiContradict transfer test")
    add_common_args(parser)
    args = parser.parse_args()

    rows = _select_rows(args.csv, args.limit)
    plan = _plan(rows, args.readers)

    if args.dry_run:
        _dry_run(plan)
        return

    answers = _run_live(plan, rows, args.parallelism)
    cells = _aggregate(answers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "answers.json").write_text(json.dumps(answers, indent=2))
    (args.output_dir / "cells.json").write_text(json.dumps(cells, indent=2))
    for key, cell in sorted(cells.items()):
        n = cell["n"]
        print(
            f"{key}: det={cell['det_correct']}/{n} "
            f"both={cell['both_present']}/{n} judge_queue={cell['judge_candidate']}"
        )


if __name__ == "__main__":
    main()
