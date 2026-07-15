"""Reader driver for the interrogator v0 conversion experiment.

Builds the four evidence conditions per BEAM retrieval-bound row, calls the
reader panel via reader.client, and writes one judge-compatible run.json per
(condition, reader). See research/interrogator-v0-conversion-2026-07.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.interrogator_conversion import (
    CONDITIONS,
    ConditionEvidence,
    assemble_evidence,
    cohort_of,
    load_rb_rows,
    load_stores,
    store_for,
)
from reader.client import generate_answer
from v2.replay_io import build_reader_prompt

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RUN_ROOT = Path("results/answer_generation_runs/interrogator_v0_conversion")
DATASET_NAME = "BEAM"
NO_FALLBACK_ROUTING: dict[str, Any] = {"allow_fallbacks": False}
TEMPERATURE = 0.0
MAX_TOKENS = 128
TIMEOUT_S = 120.0
SMALL_READER = "meta-llama/llama-3.1-8b-instruct"
STRONG_READER = "qwen/qwen-2.5-72b-instruct"


@dataclass(frozen=True)
class ReaderCall:
    """One (row, condition) unit of work for one reader."""

    row: dict
    evidence: ConditionEvidence


def _prompt_for(row: dict, evidence: ConditionEvidence) -> str:
    return build_reader_prompt(
        question=row["query"],
        question_date=row.get("active_context"),
        evidence_text=evidence.evidence_text,
    )


def _record(row: dict, evidence: ConditionEvidence, reader: str, gen: dict[str, Any]) -> dict[str, Any]:
    cohort = cohort_of(row)
    return {
        "system": f"interrogator_v0::{evidence.condition}::{reader}",
        "base_system": "interrogator_v0",
        "condition": evidence.condition,
        "example_id": row["example_id"],
        "question": row["query"],
        "reference_answer": row["reference_answer"],
        "generated_answer": gen.get("clean_answer") or gen.get("raw_answer") or "Unknown",
        "question_type": row["ability"],
        "dataset_name": DATASET_NAME,
        "harm_type": row.get("harm_type", ""),
        "ability": row["ability"],
        "is_missing_gold": cohort.is_missing_gold,
        "reader_model": reader,
        "evidence_token_count": evidence.evidence_token_estimate,
        "n_acquired": len(evidence.acquired_ids),
        "n_acquired_survived": evidence.n_acquired_survived,
        "source_memory_ids": list(evidence.source_memory_ids),
        "prompt_tokens": gen.get("prompt_tokens"),
        "completion_tokens": gen.get("completion_tokens"),
        "latency_ms": gen.get("latency_ms"),
        "serving_provider": gen.get("serving_provider"),
        "served_model": gen.get("served_model"),
        "success": bool(gen.get("success")),
        "error": gen.get("error"),
    }


def _build_calls(rows: list[dict], stores: dict[str, dict[str, str]], condition: str) -> list[ReaderCall]:
    calls: list[ReaderCall] = []
    for row in rows:
        evidence = assemble_evidence(row, store_for(row, stores), condition)
        calls.append(ReaderCall(row=row, evidence=evidence))
    return calls


def _run_one(call: ReaderCall, reader: str) -> dict[str, Any]:
    gen = generate_answer(
        provider="openrouter",
        base_url=OPENROUTER_BASE_URL,
        model=reader,
        prompt=_prompt_for(call.row, call.evidence),
        temperature=TEMPERATURE,
        timeout_s=TIMEOUT_S,
        max_tokens=MAX_TOKENS,
        provider_routing=dict(NO_FALLBACK_ROUTING),
    )
    return _record(call.row, call.evidence, reader, gen)


def _execute(calls: list[ReaderCall], reader: str, parallelism: int) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = {pool.submit(_run_one, c, reader): c for c in calls}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            outputs.append(fut.result())
            done += 1
            if done % 40 == 0:
                print(f"    {reader} {done}/{len(calls)}")
    outputs.sort(key=lambda o: o["example_id"])
    return outputs


def _write_run(run_dir: Path, condition: str, reader: str, outputs: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    n_fail = sum(1 for o in outputs if not o["success"])
    payload = {
        "meta": {
            "experiment": "interrogator_v0_conversion",
            "condition": condition,
            "reader_model": reader,
            "dataset": DATASET_NAME,
            "n_outputs": len(outputs),
            "n_failed": n_fail,
        },
        "systems": [f"interrogator_v0::{condition}::{reader}"],
        "outputs": outputs,
    }
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"  wrote {run_dir/'run.json'} ({len(outputs)} outputs, {n_fail} failed)")


def _dir_slug(condition: str, reader: str) -> str:
    return f"{condition}__{reader.replace('/', '__')}"


def run(readers: list[str], parallelism: int, dry_run: bool) -> None:
    rows = load_rb_rows()
    cohorts = [cohort_of(r) for r in rows]
    n_mg = sum(1 for c in cohorts if c.is_missing_gold)
    print(f"RB rows: {len(rows)} (missing-gold {n_mg}, do-no-harm {len(rows)-n_mg})")
    stores = load_stores()

    total_calls = len(rows) * len(CONDITIONS) * len(readers)
    print(f"Planned reader calls: {total_calls} ({len(CONDITIONS)} conditions x {len(readers)} readers)")
    if dry_run:
        for condition in CONDITIONS:
            sample = assemble_evidence(rows[0], store_for(rows[0], stores), condition)
            print(
                f"  [{condition}] eid={sample.example_id} acquired={len(sample.acquired_ids)} "
                f"survived={sample.n_acquired_survived} evid_tok={sample.evidence_token_estimate}"
            )
        print("Dry run only. Re-run with --yes to execute reader calls.")
        return

    for condition in CONDITIONS:
        calls = _build_calls(rows, stores, condition)
        for reader in readers:
            print(f"[{condition}] {reader}: {len(calls)} calls")
            outputs = _execute(calls, reader, parallelism)
            _write_run(RUN_ROOT / _dir_slug(condition, reader), condition, reader, outputs)


def resume(readers: list[str], parallelism: int) -> None:
    """Re-run only failed outputs (e.g. transient 429s) and patch run.json in place."""
    rows = {r["example_id"]: r for r in load_rb_rows()}
    stores = load_stores()
    for condition in CONDITIONS:
        for reader in readers:
            run_dir = RUN_ROOT / _dir_slug(condition, reader)
            run_path = run_dir / "run.json"
            if not run_path.exists():
                continue
            payload = json.loads(run_path.read_text())
            outputs = payload["outputs"]
            failed_idx = [i for i, o in enumerate(outputs) if not o["success"]]
            if not failed_idx:
                continue
            print(f"[{condition}] {reader}: retrying {len(failed_idx)} failed calls")
            calls = [
                ReaderCall(
                    row=rows[outputs[i]["example_id"]],
                    evidence=assemble_evidence(
                        rows[outputs[i]["example_id"]],
                        store_for(rows[outputs[i]["example_id"]], stores),
                        condition,
                    ),
                )
                for i in failed_idx
            ]
            redone = _execute(calls, reader, parallelism)
            by_id = {o["example_id"]: o for o in redone}
            for i in failed_idx:
                outputs[i] = by_id[outputs[i]["example_id"]]
            _write_run(run_dir, condition, reader, outputs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interrogator v0 conversion reader driver.")
    parser.add_argument("--resume", action="store_true", help="Re-run only failed outputs.")
    parser.add_argument("--readers", default=f"{SMALL_READER},{STRONG_READER}")
    parser.add_argument("--parallelism", type=int, default=32)
    parser.add_argument("--yes", action="store_true", help="Execute reader calls (default dry-run).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    readers = [r.strip() for r in args.readers.split(",") if r.strip()]
    if args.resume:
        resume(readers, args.parallelism)
        return
    run(readers, args.parallelism, dry_run=not args.yes)


if __name__ == "__main__":
    main()
