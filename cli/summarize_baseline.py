#!/usr/bin/env python3
"""LLM-Summarize baseline: BM25 top-20 → generic LLM summary → reader.

Same compiler model (Qwen3-8B) and same input as SIEVE, but no typed
routing or structured extraction — just "summarize what's relevant."
Isolates whether gains come from ANY compilation or from SIEVE's
specific typed pipeline.

Supports compile-once / replay-many via --summary-cache:
  # First run: compiles summaries and saves cache
  python run_llm_summarize_baseline.py --slice ... --reader-model ... \\
      --summary-cache data/compilation_cache/llm_sum_qwen3_8b.json

  # Subsequent runs: loads cached summaries, skips compilation
  python run_llm_summarize_baseline.py --slice ... --reader-model ... \\
      --summary-cache data/compilation_cache/llm_sum_qwen3_8b.json

Output format is identical to run_answer_generation_phase1.py so
scoring_judge.py works unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reader.client import (
    DEFAULT_OPENROUTER_BASE_URL,
    generate_answer,
)
from reader.prompts import build_answer_generation_prompt
from reader.scoring import quality_metrics_for_row
from reader.pali_shims import is_unknown_answer

SYSTEM_NAME = "llm_summarize"

_SUMMARIZE_PROMPT = """\
You are a precise evidence extractor. Given a user question and retrieved conversation memories, extract ONLY the information relevant to answering the question. Output a concise summary of the relevant evidence.

Rules:
- Include dates, names, numbers, and specific details that help answer the question.
- Omit greetings, small talk, and unrelated conversation turns.
- If multiple memories contain relevant information, combine them.
- If no memory is relevant, say "No relevant evidence found."
- Be concise — aim for 2-5 sentences maximum.

Question: {question}

Retrieved memories:
{memories}

Relevant evidence summary:"""


def _format_memories(candidate_memories: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, mem in enumerate(candidate_memories, 1):
        content = str(mem.get("content", "")).strip()
        if content:
            parts.append(f"[Memory {i}]\n{content}")
    return "\n\n".join(parts)


# ── Summary cache ─────────────────────────────────────────────────────

def _cache_fingerprint(slice_path: str, compiler_model: str) -> str:
    """Deterministic hash so cache only matches same slice + compiler."""
    parts = [slice_path, compiler_model]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _load_summary_cache(path: Path) -> dict[str, Any] | None:
    """Load existing cache. Returns None if file doesn't exist."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def _save_summary_cache(
    path: Path,
    entries: dict[str, dict[str, Any]],
    *,
    slice_path: str,
    compiler_model: str,
) -> None:
    """Write summary cache to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "fingerprint": _cache_fingerprint(slice_path, compiler_model),
        "created": datetime.now(timezone.utc).isoformat(),
        "compiler_model": compiler_model,
        "slice_path": slice_path,
        "row_count": len(entries),
        "entries": entries,
    }
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Compilation ───────────────────────────────────────────────────────

def _summarize_evidence(
    *,
    question: str,
    candidate_memories: list[dict[str, Any]],
    compiler_model: str,
    provider: str,
    base_url: str,
    api_key_env: str,
) -> dict[str, Any]:
    """Call the compiler model to produce a generic summary."""
    memories_text = _format_memories(candidate_memories)
    prompt = _SUMMARIZE_PROMPT.format(question=question, memories=memories_text)

    result = generate_answer(
        provider=provider,
        base_url=base_url,
        model=compiler_model,
        prompt=prompt,
        temperature=0.0,
        timeout_s=60.0,
        max_tokens=256,
        api_key_env=api_key_env,
    )
    return {
        "success": result.get("success", False),
        "summary": result.get("clean_answer", ""),
        "raw_summary": result.get("raw_answer", ""),
        "compiler_input_tokens": result.get("prompt_tokens", 0),
        "compiler_output_tokens": result.get("completion_tokens", 0),
        "compiler_latency_ms": result.get("latency_ms", 0.0),
    }


def _compile_row(
    row: dict[str, Any],
    *,
    compiler_model: str,
    provider: str,
    base_url: str,
    api_key_env: str,
) -> dict[str, Any]:
    """Compile a single row: produce summary + build reader prompt."""
    question = str(row["query"])
    candidates = row.get("candidate_memories", [])
    active_context = [str(line) for line in row.get("active_context", [])]

    summary_result = _summarize_evidence(
        question=question,
        candidate_memories=candidates,
        compiler_model=compiler_model,
        provider=provider,
        base_url=base_url,
        api_key_env=api_key_env,
    )

    summary_text = summary_result.get("summary", "").strip()
    if not summary_text or is_unknown_answer(summary_text):
        retrieved_memories = [
            str(mem.get("content", "")).strip()
            for mem in candidates
            if str(mem.get("content", "")).strip()
        ]
        compiler_fell_back = True
    else:
        retrieved_memories = [summary_text]
        compiler_fell_back = False

    prompt = build_answer_generation_prompt(
        question=question,
        active_context=active_context,
        retrieved_memories=retrieved_memories,
        variant="baseline",
        insufficient_answer_mode="unknown",
        question_type=str(row.get("question_type", "")),
    )

    return {
        "example_id": row.get("example_id", ""),
        "question": question,
        "question_type": row.get("question_type", ""),
        "dataset_name": row.get("dataset_name", ""),
        "reference_answer": str(row.get("reference_answer", "")),
        "active_context": active_context,
        "prompt": prompt,
        "summary_text": summary_text,
        "compiler_fell_back": compiler_fell_back,
        "compiler_model": compiler_model,
        "compiler_input_tokens": summary_result.get("compiler_input_tokens", 0),
        "compiler_output_tokens": summary_result.get("compiler_output_tokens", 0),
        "compiler_latency_ms": summary_result.get("compiler_latency_ms", 0.0),
        "selected_memory_tokens": sum(len(str(t).split()) for t in retrieved_memories),
    }


# ── Reading (replay) ──────────────────────────────────────────────────

def _read_row(
    entry: dict[str, Any],
    *,
    reader_model: str,
    provider: str,
    base_url: str,
    api_key_env: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    """Send cached prompt to a reader model and score."""
    generation = generate_answer(
        provider=provider,
        base_url=base_url,
        model=reader_model,
        prompt=entry["prompt"],
        temperature=temperature,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        api_key_env=api_key_env,
    )

    if generation.get("status_code") == 402:
        raise RuntimeError(f"API credit limit reached (402) for {entry.get('example_id')}")

    row_for_scoring = {
        "dataset_name": entry.get("dataset_name"),
        "question_type": entry.get("question_type"),
        "query": entry["question"],
        "reference_answer": entry.get("reference_answer"),
    }
    quality = quality_metrics_for_row(
        row_for_scoring,
        entry["question"],
        generation["clean_answer"],
        entry.get("reference_answer", ""),
    )

    return {
        "example_id": entry.get("example_id", ""),
        "dataset_name": entry.get("dataset_name", ""),
        "question": entry["question"],
        "question_type": entry.get("question_type", ""),
        "reference_answer": entry.get("reference_answer", ""),
        "generated_answer": generation.get("clean_answer", "Unknown"),
        "raw_answer": generation.get("raw_answer", ""),
        "system": SYSTEM_NAME,
        "success": generation.get("success", False),
        "quality_score": quality.get("quality_score", 0),
        "quality_norm": quality.get("quality_norm", 0.0),
        "exact_correct": quality.get("exact_correct", 0),
        "token_f1": quality.get("token_f1", 0.0),
        "prompt_tokens": generation.get("prompt_tokens", 0),
        "completion_tokens": generation.get("completion_tokens", 0),
        "latency_ms": generation.get("latency_ms", 0.0),
        "selected_memory_tokens": entry.get("selected_memory_tokens", 0),
        "compiler_model": entry.get("compiler_model", ""),
        "compiler_input_tokens": entry.get("compiler_input_tokens", 0),
        "compiler_output_tokens": entry.get("compiler_output_tokens", 0),
        "compiler_latency_ms": entry.get("compiler_latency_ms", 0.0),
        "compiler_fell_back": entry.get("compiler_fell_back", False),
        "summary_text": entry.get("summary_text", ""),
    }


# ── Summary markdown ──────────────────────────────────────────────────

def _write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    outputs = payload["outputs"]
    n = len(outputs)
    if n == 0:
        path.write_text("# LLM-Summarize Baseline\n\nNo outputs.\n")
        return

    successes = sum(1 for o in outputs if o.get("success"))
    correct = sum(1 for o in outputs if o.get("exact_correct"))
    avg_quality = sum(o.get("quality_norm", 0.0) for o in outputs) / n
    avg_mem_tok = sum(o.get("selected_memory_tokens", 0) for o in outputs) / n
    avg_prompt_tok = sum(o.get("prompt_tokens", 0) for o in outputs) / n
    avg_latency = sum(o.get("latency_ms", 0.0) for o in outputs) / n
    avg_compiler_in = sum(o.get("compiler_input_tokens", 0) for o in outputs) / n
    avg_compiler_out = sum(o.get("compiler_output_tokens", 0) for o in outputs) / n
    fallback_count = sum(1 for o in outputs if o.get("compiler_fell_back"))

    by_qtype: dict[str, list[dict]] = {}
    for o in outputs:
        qt = o.get("question_type", "unknown")
        by_qtype.setdefault(qt, []).append(o)

    lines = [
        "# LLM-Summarize Baseline",
        "",
        f"- **Reader model**: `{payload['meta']['reader_model']}`",
        f"- **Compiler model**: `{payload['meta']['compiler_model']}`",
        f"- **Rows**: {n}",
        f"- **Cache**: `{payload['meta'].get('summary_cache', 'none')}`",
        f"- **Success rate**: {successes}/{n} ({100*successes/n:.1f}%)",
        f"- **Exact correct (heuristic)**: {correct}/{n} ({100*correct/n:.1f}%)",
        f"- **Avg quality_norm**: {avg_quality:.3f}",
        f"- **Avg evidence tokens**: {avg_mem_tok:.0f}",
        f"- **Avg prompt tokens**: {avg_prompt_tok:.0f}",
        f"- **Avg latency**: {avg_latency:.0f}ms",
        f"- **Avg compiler input tokens**: {avg_compiler_in:.0f}",
        f"- **Avg compiler output tokens**: {avg_compiler_out:.0f}",
        f"- **Compiler fallbacks**: {fallback_count}/{n} ({100*fallback_count/n:.1f}%)",
        "",
        "## Per Question-Type",
        "",
        "| Question Type | n | Exact Correct | Avg Quality |",
        "| --- | ---: | ---: | ---: |",
    ]
    for qt in sorted(by_qtype.keys()):
        qt_rows = by_qtype[qt]
        qn = len(qt_rows)
        qcorrect = sum(1 for o in qt_rows if o.get("exact_correct"))
        qavg = sum(o.get("quality_norm", 0.0) for o in qt_rows) / qn
        lines.append(f"| {qt} | {qn} | {qcorrect}/{qn} ({100*qcorrect/qn:.1f}%) | {qavg:.3f} |")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-Summarize baseline: generic LLM summary → reader.",
    )
    parser.add_argument("--slice", required=True, help="Path to evaluation slice JSONL")
    parser.add_argument("--reader-model", required=True, help="Reader model (e.g. meta-llama/llama-3.1-8b-instruct)")
    parser.add_argument("--compiler-model", default="qwen/qwen3-8b", help="Compiler/summarizer model (default: qwen/qwen3-8b)")
    parser.add_argument("--provider", default="openrouter", choices=["openrouter", "ollama"])
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--summary-cache",
        default=None,
        help=(
            "Path to summary compilation cache JSON. "
            "If the file exists and fingerprint matches, summaries are loaded "
            "from cache (no compiler calls). If the file does not exist, "
            "summaries are compiled and saved to this path for future reuse."
        ),
    )
    args = parser.parse_args()

    slice_path = Path(args.slice)
    rows = [json.loads(line) for line in slice_path.read_text().strip().splitlines()]
    if args.max_examples:
        rows = rows[:args.max_examples]

    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        reader_slug = args.reader_model.split("/")[-1].replace(":", "-").lower()
        run_dir = Path(__file__).resolve().parent.parent / "results" / "answer_generation_runs" / f"paper_{reader_slug}_llm_summarize"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_url = DEFAULT_OPENROUTER_BASE_URL if args.provider == "openrouter" else "http://127.0.0.1:11434"

    # ── Step 1: Get compiled summaries (from cache or fresh) ──────────
    cache_path = Path(args.summary_cache) if args.summary_cache else None
    cache_entries: dict[str, dict[str, Any]] | None = None
    cache_hit = False

    if cache_path:
        existing = _load_summary_cache(cache_path)
        if existing is not None:
            expected_fp = _cache_fingerprint(str(slice_path), args.compiler_model)
            if existing.get("fingerprint") == expected_fp:
                cache_entries = existing["entries"]
                cache_hit = True
                print(f"Summary cache HIT: {cache_path} ({len(cache_entries)} entries)")
            else:
                print(f"Summary cache FINGERPRINT MISMATCH: {cache_path}")
                print(f"  Expected: {expected_fp}")
                print(f"  Got:      {existing.get('fingerprint')}")
                print(f"  Recompiling...")

    if not cache_hit:
        # Compile all rows
        print(f"Compiling summaries: {len(rows)} rows with {args.compiler_model}")
        compiled: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            futures = {
                executor.submit(
                    _compile_row,
                    row,
                    compiler_model=args.compiler_model,
                    provider=args.provider,
                    base_url=base_url,
                    api_key_env=args.api_key_env,
                ): row.get("example_id", str(i))
                for i, row in enumerate(rows)
            }
            for future in as_completed(futures):
                entry = future.result()
                compiled.append(entry)
                if len(compiled) % 50 == 0:
                    print(f"  Compiled {len(compiled)}/{len(rows)}...")

        cache_entries = {e["example_id"]: e for e in compiled}

        # Save cache if path was given
        if cache_path:
            _save_summary_cache(
                cache_path,
                cache_entries,
                slice_path=str(slice_path),
                compiler_model=args.compiler_model,
            )
            print(f"Summary cache SAVED: {cache_path} ({len(cache_entries)} entries)")

    # Build ordered list of entries matching slice order
    entries = []
    missing = 0
    for row in rows:
        eid = row.get("example_id", "")
        if eid in cache_entries:
            entries.append(cache_entries[eid])
        else:
            missing += 1
    if missing:
        print(f"WARNING: {missing} rows not found in cache, skipped")

    # ── Step 2: Replay with reader model ──────────────────────────────
    print(f"\nLLM-Summarize {'replay' if cache_hit else 'run'}: {len(entries)} rows")
    print(f"  Reader: {args.reader_model}")
    print(f"  Compiler: {args.compiler_model} ({'cached' if cache_hit else 'fresh'})")
    print(f"  Parallelism: {args.parallelism}")
    print(f"  Output: {run_dir}")

    outputs: list[dict[str, Any]] = []
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = {
            executor.submit(
                _read_row,
                entry,
                reader_model=args.reader_model,
                provider=args.provider,
                base_url=base_url,
                api_key_env=args.api_key_env,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout,
            ): entry["example_id"]
            for entry in entries
        }
        for future in as_completed(futures):
            result = future.result()
            outputs.append(result)
            if len(outputs) % 50 == 0:
                print(f"  Read {len(outputs)}/{len(entries)}...")

    elapsed = time.perf_counter() - started

    # Sort by original slice order
    eid_order = {row.get("example_id"): i for i, row in enumerate(rows)}
    outputs.sort(key=lambda o: eid_order.get(o.get("example_id"), 0))

    payload = {
        "meta": {
            "reader_model": args.reader_model,
            "compiler_model": args.compiler_model,
            "provider": args.provider,
            "slice": str(slice_path),
            "system": SYSTEM_NAME,
            "parallelism": args.parallelism,
            "rows": len(outputs),
            "wall_seconds": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
            "summary_cache": str(cache_path) if cache_path else None,
            "cache_hit": cache_hit,
        },
        "systems": [SYSTEM_NAME],
        "summaries": {
            SYSTEM_NAME: {
                "avg_quality_norm": sum(o.get("quality_norm", 0) for o in outputs) / max(len(outputs), 1),
                "exact_correct_rate": sum(o.get("exact_correct", 0) for o in outputs) / max(len(outputs), 1),
                "avg_selected_memory_tokens": sum(o.get("selected_memory_tokens", 0) for o in outputs) / max(len(outputs), 1),
                "avg_prompt_tokens": sum(o.get("prompt_tokens", 0) for o in outputs) / max(len(outputs), 1),
                "avg_latency_ms": sum(o.get("latency_ms", 0) for o in outputs) / max(len(outputs), 1),
            },
        },
        "outputs": outputs,
    }

    run_json = run_dir / "run.json"
    run_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_md = run_dir / "summary.md"
    _write_summary_markdown(summary_md, payload)

    n = len(outputs)
    correct = sum(1 for o in outputs if o.get("exact_correct"))
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Heuristic exact correct: {correct}/{n} ({100*correct/n:.1f}%)")
    print(f"  Wrote: {run_json}")
    print(f"  Wrote: {summary_md}")

    s = payload["summaries"][SYSTEM_NAME]
    print(
        f"  {SYSTEM_NAME}: avg_quality_norm={s['avg_quality_norm']:.3f}, "
        f"exact_correct={s['exact_correct_rate']:.3f}, "
        f"avg_memory_tokens={s['avg_selected_memory_tokens']:.0f}, "
        f"avg_prompt_tokens={s['avg_prompt_tokens']:.0f}, "
        f"avg_latency_ms={s['avg_latency_ms']:.0f}"
    )


if __name__ == "__main__":
    main()
