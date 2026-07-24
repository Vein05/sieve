"""Run the reference-free BEAM event-order execution pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from analysis.event_order_executor import (
    ABILITY_EVENT_ORDERING,
    EvidenceCondition,
    EventRecord,
    extraction_prompt,
    gold_ids,
    parse_event_records,
    render_gold_evidence,
    render_records,
)
from analysis.interrogator_conversion import load_rb_rows, load_stores, store_for
from reader.client import generate_answer

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OUTPUT_DIR = Path("results/v2_runs/event_order_executor_pilot_v0")
COMPILER_MODEL = "qwen/qwen-2.5-72b-instruct"
READER_MODEL = COMPILER_MODEL
JUDGE_MODEL = "deepseek/deepseek-chat-v3-0324"
DATASET_NAME = "BEAM"
ROW_LIMIT = 20
MIN_GOLD_UNITS = 2
COMPILER_PARALLELISM = 1
READER_PARALLELISM = 8
JUDGE_PARALLELISM = 12
TIMEOUT_S = 180.0
COMPILER_MAX_TOKENS = 1_024
READER_MAX_TOKENS = 512
JUDGE_MAX_TOKENS = 256
MAX_PARSE_ATTEMPTS = 3
RETRY_DELAY_S = 2.0
LIVE_ACCURACY = 0.30
LIVE_GAIN = 0.15
DEAD_ACCURACY = 0.15
DEAD_GAIN = 0.05
NO_FALLBACK_ROUTING: dict[str, Any] = {"allow_fallbacks": False}
CONDITIONS = tuple(EvidenceCondition)
T = TypeVar("T")
READER_TEMPLATE = """Answer a question about a long-running conversation using the evidence below.
The question asks for events or aspects in chronological order. Follow any requested item count.
Return a numbered list in complete, concise sentences. Do not continue the conversation.

Question: {question}

Evidence:
{evidence}

Answer:"""
JUDGE_TEMPLATE = """Judge an ordered-list answer against the reference.
Mark correct only if it contains the requested number of semantic items, every item matches the
reference, and the items appear in the same chronological order. Paraphrases are allowed.
Return JSON only: {{"correct":true,"reason":"brief reason"}}

Question: {question}
Reference: {reference}
Candidate: {candidate}
"""
@dataclass(frozen=True)
class CompiledRow:
    """Successful query-focused extraction for one row."""

    example_id: str
    records: tuple[EventRecord, ...]
    raw_output: str
    prompt_tokens: int
    completion_tokens: int
@dataclass(frozen=True)
class AnswerRecord:
    """One reader response under one evidence condition."""

    example_id: str
    condition: str
    question: str
    reference_answer: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    success: bool
@dataclass(frozen=True)
class JudgeRecord:
    """One ability-specific correctness judgment."""

    example_id: str
    condition: str
    correct: bool
    reason: str
    success: bool
def _call(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return generate_answer(
        provider="openrouter",
        base_url=OPENROUTER_BASE_URL,
        model=model,
        prompt=prompt,
        temperature=0.0,
        timeout_s=TIMEOUT_S,
        max_tokens=max_tokens,
        provider_routing=dict(NO_FALLBACK_ROUTING),
    )
def _select_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    stores = load_stores()
    candidates = [row for row in load_rb_rows() if row["ability"] == ABILITY_EVENT_ORDERING]
    rows = []
    for row in candidates:
        store = store_for(row, stores)
        ids = gold_ids(row)
        if len(ids) >= MIN_GOLD_UNITS and all(store.get(memory_id) for memory_id in ids):
            rows.append(row)
        if len(rows) == ROW_LIMIT:
            break
    return rows, stores
def _compile_one(row: dict[str, Any], store: dict[str, str]) -> CompiledRow | None:
    prompt = extraction_prompt(row, store)
    for attempt in range(MAX_PARSE_ATTEMPTS):
        generation = _call(COMPILER_MODEL, prompt, COMPILER_MAX_TOKENS)
        raw = str(generation.get("raw_answer") or "")
        try:
            records = parse_event_records(raw, gold_ids(row))
            return CompiledRow(
                example_id=row["example_id"],
                records=records,
                raw_output=raw,
                prompt_tokens=int(generation.get("prompt_tokens") or 0),
                completion_tokens=int(generation.get("completion_tokens") or 0),
            )
        except (ValueError, json.JSONDecodeError) as exc:
            expected = ", ".join(gold_ids(row))
            prompt += f"\nInvalid output ({exc}). Include exactly these source ids: {expected}. Return JSON only."
            if attempt < MAX_PARSE_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY_S)
    return None
def _parallel_map(function: Callable[[T], Any], items: list[T], parallelism: int) -> list[Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
        return list(pool.map(function, items))
def _compile_rows(
    rows: list[dict[str, Any]], stores: dict[str, dict[str, str]]
) -> dict[str, CompiledRow]:
    cached = _load_compiled()
    def compile_item(row: dict[str, Any]) -> CompiledRow | None:
        return _compile_one(row, store_for(row, stores))
    missing = [row for row in rows if row["example_id"] not in cached]
    outputs = _parallel_map(compile_item, missing, COMPILER_PARALLELISM)
    cached.update({item.example_id: item for item in outputs if item is not None})
    return cached

def _load_compiled() -> dict[str, CompiledRow]:
    path = OUTPUT_DIR / "compiled.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        item["example_id"]: CompiledRow(
            item["example_id"], tuple(EventRecord(**record) for record in item["records"]),
            item["raw_output"], item["prompt_tokens"], item["completion_tokens"],
        )
        for item in payload
    }
def _evidence_for(
    row: dict[str, Any], compiled: CompiledRow, store: dict[str, str], condition: EvidenceCondition
) -> str:
    if condition == EvidenceCondition.GOLD_CHRONO:
        return render_gold_evidence(row, store)
    return render_records(compiled.records, condition)
def _answer_one(job: tuple[dict[str, Any], CompiledRow, dict[str, str], EvidenceCondition]) -> AnswerRecord:
    row, compiled, store, condition = job
    evidence = _evidence_for(row, compiled, store, condition)
    prompt = READER_TEMPLATE.format(question=row["query"], evidence=evidence)
    generation = _call(READER_MODEL, prompt, READER_MAX_TOKENS)
    return AnswerRecord(
        example_id=row["example_id"],
        condition=condition.value,
        question=row["query"],
        reference_answer=row["reference_answer"],
        answer=str(generation.get("raw_answer") or ""),
        prompt_tokens=int(generation.get("prompt_tokens") or 0),
        completion_tokens=int(generation.get("completion_tokens") or 0),
        success=bool(generation.get("success")),
    )


def _answer_rows(
    rows: list[dict[str, Any]], stores: dict[str, dict[str, str]], compiled: dict[str, CompiledRow]
) -> list[AnswerRecord]:
    jobs = []
    for row in rows:
        item = compiled.get(row["example_id"])
        if item is None:
            continue
        store = store_for(row, stores)
        jobs.extend((row, item, store, condition) for condition in CONDITIONS)
    return _parallel_map(_answer_one, jobs, READER_PARALLELISM)
def _parse_judgment(raw: str) -> tuple[bool, str]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge returned no JSON")
    payload = json.loads(raw[start : end + 1])
    return bool(payload["correct"]), str(payload.get("reason", ""))
def _judge_one(answer: AnswerRecord) -> JudgeRecord:
    prompt = JUDGE_TEMPLATE.format(
        question=answer.question,
        reference=answer.reference_answer,
        candidate=answer.answer,
    )
    generation = _call(JUDGE_MODEL, prompt, JUDGE_MAX_TOKENS)
    try:
        correct, reason = _parse_judgment(str(generation.get("raw_answer") or ""))
        return JudgeRecord(answer.example_id, answer.condition, correct, reason, True)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        return JudgeRecord(answer.example_id, answer.condition, False, str(exc), False)
def _gold_sanity(rows: list[dict[str, Any]]) -> list[JudgeRecord]:
    answers = [
        AnswerRecord(
            row["example_id"], "gold_sanity", row["query"], row["reference_answer"],
            row["reference_answer"], 0, 0, True,
        )
        for row in rows
    ]
    return _parallel_map(_judge_one, answers, JUDGE_PARALLELISM)
def _paired_counts(judgments: list[JudgeRecord], first: str, second: str) -> tuple[int, int]:
    labels = {(item.example_id, item.condition): item.correct for item in judgments}
    ids = {item.example_id for item in judgments}
    rescues = sum(not labels[(eid, first)] and labels[(eid, second)] for eid in ids)
    damages = sum(labels[(eid, first)] and not labels[(eid, second)] for eid in ids)
    return rescues, damages
def _summary(judgments: list[JudgeRecord]) -> dict[str, Any]:
    by_condition: dict[str, float] = {}
    for condition in CONDITIONS:
        cells = [item.correct for item in judgments if item.condition == condition.value]
        by_condition[condition.value] = sum(cells) / len(cells)
    baseline = max(
        by_condition[EvidenceCondition.GOLD_CHRONO.value],
        by_condition[EvidenceCondition.EXTRACTED_UNSORTED.value],
    )
    executed = by_condition[EvidenceCondition.EXECUTED_SORTED.value]
    gain = executed - baseline
    if executed >= LIVE_ACCURACY and gain >= LIVE_GAIN:
        verdict = "live"
    elif executed <= DEAD_ACCURACY or gain < DEAD_GAIN:
        verdict = "dead"
    else:
        verdict = "inconclusive"
    rescues, damages = _paired_counts(
        judgments, EvidenceCondition.EXTRACTED_UNSORTED.value,
        EvidenceCondition.EXECUTED_SORTED.value,
    )
    return {"accuracy": by_condition, "gain_over_best_control": gain, "verdict": verdict,
            "sorted_vs_unsorted_rescues": rescues, "sorted_vs_unsorted_damages": damages}
def _write_outputs(
    compiled: dict[str, CompiledRow], answers: list[AnswerRecord], judgments: list[JudgeRecord],
    sanity: list[JudgeRecord], summary: dict[str, Any],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    compiled_payload = [asdict(item) for item in compiled.values()]
    (OUTPUT_DIR / "compiled.json").write_text(json.dumps(compiled_payload, indent=2))
    (OUTPUT_DIR / "answers.json").write_text(json.dumps([asdict(item) for item in answers], indent=2))
    (OUTPUT_DIR / "judgments.json").write_text(json.dumps([asdict(item) for item in judgments], indent=2))
    payload = {"judge_gold_sanity": sum(item.correct for item in sanity) / len(sanity), **summary}
    (OUTPUT_DIR / "analysis.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
def run(execute: bool) -> None:
    rows, stores = _select_rows()
    calls_per_row = len(CONDITIONS) * 2 + 2
    print(f"selected rows: {len(rows)}; planned calls: {len(rows) * calls_per_row}")
    if not execute:
        print("Dry run only. Re-run with --yes to execute paid calls.")
        return
    compiled = _compile_rows(rows, stores)
    balanced_rows = [row for row in rows if row["example_id"] in compiled]
    print(f"valid compilations: {len(compiled)}/{len(rows)}")
    answers = _answer_rows(balanced_rows, stores, compiled)
    judgments = _parallel_map(_judge_one, answers, JUDGE_PARALLELISM)
    sanity = _gold_sanity(balanced_rows)
    _write_outputs(compiled, answers, judgments, sanity, _summary(judgments))
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the BEAM event-order executor pilot.")
    parser.add_argument("--yes", action="store_true", help="Execute paid OpenRouter calls.")
    args = parser.parse_args(argv)
    run(execute=args.yes)


if __name__ == "__main__":
    main()
