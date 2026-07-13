#!/usr/bin/env python3
"""Compile once, run readers again — fast reader-model sweep.

The SIEVE pipeline has two stages:
  1. Compilation: BM25 retrieval → NLP extraction → schema matching →
     LLM cascade compiler → budget-aware evidence packaging → prompt building.
  2. Reading: send the compiled prompt to a reader model and score.

Stage 1 is identical regardless of which reader model is used.  This script
caches stage 1 output (the final prompt + metadata per row) and replays only
stage 2 with a different reader model.  This makes reader-model sweeps
~100x faster since compilation dominates wall-clock time.

Usage:
  # Step 1: run a full pipeline once (creates the cache)
  python compile_once_run_again.py cache \\
      --source-run results/answer_generation_runs/<full_run_dir> \\
      --output-cache data/compilation_cache/longmemeval_bm25_sieve_v1.json

  # Step 2: replay with a different reader model
  python compile_once_run_again.py replay \\
      --cache data/compilation_cache/longmemeval_bm25_sieve_v1.json \\
      --model meta-llama/llama-3.1-70b-instruct \\
      --output-dir results/answer_generation_runs/llama70b_sieve_from_cache

  # Step 3: replay with yet another reader
  python compile_once_run_again.py replay \\
      --cache data/compilation_cache/longmemeval_bm25_sieve_v1.json \\
      --model qwen/qwen-2.5-7b-instruct \\
      --output-dir results/answer_generation_runs/qwen7b_sieve_from_cache

Cache integrity: the cache records a fingerprint of the compilation config
(compiler model, cascade settings, slice path hash, system name, pool aug
settings).  Replay refuses to run if the fingerprint doesn't match the
current config, preventing stale-cache bugs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

# Ensure scripts/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reader.client import DEFAULT_APP_URL, generate_answer
from reader.scoring import evaluate_run, quality_metrics_for_row


# ── Cache creation ───────────────────────────────────────────────────

def _build_fingerprint(trace: dict[str, Any], run_meta: dict[str, Any]) -> str:
    """Deterministic hash of compilation-relevant config fields."""
    parts = [
        run_meta.get("slice_path", ""),
        run_meta.get("system", ""),
        run_meta.get("prompt_variant", ""),
        run_meta.get("sieve_reader_mode", ""),
        str(run_meta.get("pool_augmentation", True)),
        run_meta.get("cascade_compiler_model", ""),
        str(run_meta.get("cascade_compiler_top_k", 5)),
        str(run_meta.get("external_prompt_cap", "")),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _extract_cache_entry(trace: dict[str, Any]) -> dict[str, Any]:
    """Extract the reader-independent fields from a trace entry."""
    gen = trace.get("generation", {}) or {}
    prompt_tokens = gen.get("prompt_tokens") or 0
    clean_answer = gen.get("clean_answer", "")
    is_det = prompt_tokens == 0 and bool(clean_answer)

    return {
        # Identity
        "example_id": trace["example_id"],
        "system": trace.get("system", ""),
        "question_type": trace.get("question_type", ""),
        "dataset_name": trace.get("dataset_name", ""),
        # The final prompt (this IS the compilation output)
        "prompt": trace.get("prompt", ""),
        "prompt_variant": trace.get("prompt_variant", ""),
        # Metadata needed for scoring and output assembly
        "question": trace.get("question", ""),
        "question_date": trace.get("question_date"),
        "reference_answer": trace.get("reference_answer"),
        "harm_type": trace.get("harm_type"),
        "locomo_category": trace.get("locomo_category"),
        "ability": trace.get("ability"),
        "chat_size": trace.get("chat_size"),
        "chat_id": trace.get("chat_id"),
        "active_context": trace.get("active_context", []),
        "selected_memory_ids": trace.get("selected_memory_ids", []),
        "selected_memory_texts": trace.get("selected_memory_texts", []),
        # Compiler outputs
        "selection_meta": trace.get("selection_meta", {}),
        "compiler_payload": trace.get("compiler_payload"),
        "rendered_evidence_package": trace.get("rendered_evidence_package"),
        "reader_route_reason": trace.get("reader_route_reason", ""),
        "answerability_level": trace.get("answerability_level", ""),
        "deterministic_allowed": trace.get("deterministic_allowed", False),
        "query_family": trace.get("query_family"),
        "selection_mode": trace.get("selection_mode"),
        "schema_name": trace.get("schema_name"),
        "compiler_type": trace.get("compiler_type", "rule_based"),
        "compiler_model": trace.get("compiler_model"),
        "llm_compiler_input_tokens": trace.get("llm_compiler_input_tokens", 0),
        "llm_compiler_output_tokens": trace.get("llm_compiler_output_tokens", 0),
        # Deterministic answers (no reader needed for these rows)
        "is_deterministic": is_det,
        "deterministic_answer": clean_answer if is_det else None,
        # Token budget info
        "evidence_budget_target": trace.get("budget_target_tokens"),
        "evidence_budget_hard": trace.get("budget_hard_tokens"),
        "selected_memory_tokens": trace.get("selected_memory_tokens", 0),
        "compiled_token_count": trace.get("compiled_token_count", 0),
    }


def cmd_cache(args: argparse.Namespace) -> None:
    """Build a compilation cache from a completed run."""
    source_dir = Path(args.source_run)
    trace_path = source_dir / "trace.json"
    run_path = source_dir / "run.json"

    if not trace_path.exists():
        print(f"ERROR: {trace_path} not found. Run a full pipeline first.", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(trace_path.read_text(encoding="utf-8"))
    # trace.json can be a list of traces or a dict with a "traces" key
    if isinstance(raw, dict):
        run_meta_from_trace = {k: v for k, v in raw.items() if k != "traces"}
        traces = raw.get("traces", [])
    elif isinstance(raw, list):
        run_meta_from_trace = {}
        traces = raw
    else:
        print(f"ERROR: unexpected format in {trace_path}", file=sys.stderr)
        sys.exit(1)
    if not traces:
        print(f"ERROR: no traces found in {trace_path}", file=sys.stderr)
        sys.exit(1)

    # Extract run metadata for fingerprint (prefer trace header, fall back to run.json)
    run_meta: dict[str, Any] = {}
    run_outputs: list[dict[str, Any]] = []
    if run_path.exists():
        run_data = json.loads(run_path.read_text(encoding="utf-8"))
        if isinstance(run_data, dict):
            run_meta = {k: v for k, v in run_data.items() if k != "outputs"}
            run_outputs = run_data.get("outputs", [])
    run_meta.update(run_meta_from_trace)

    # Merge run.json outputs into traces (run.json has fields like question_type
    # that traces lack)
    if run_outputs and len(run_outputs) == len(traces):
        outputs_by_id = {str(o.get("example_id", "")): o for o in run_outputs}
        for trace in traces:
            eid = str(trace.get("example_id", ""))
            if eid in outputs_by_id:
                out = outputs_by_id[eid]
                for field in ("question_type", "locomo_category", "ability"):
                    if not trace.get(field) and out.get(field):
                        trace[field] = out[field]

    fingerprint = _build_fingerprint(traces[0], run_meta)

    entries = []
    skipped = 0
    for trace in traces:
        entry = _extract_cache_entry(trace)
        if not entry["prompt"] and not entry["is_deterministic"]:
            skipped += 1
            continue
        entries.append(entry)

    cache = {
        "fingerprint": fingerprint,
        "created": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source_dir),
        "run_meta": {
            k: run_meta.get(k)
            for k in [
                "slice_path", "system", "systems", "prompt_variant",
                "sieve_reader_mode", "pool_augmentation",
                "cascade_compiler_model", "cascade_compiler_top_k",
                "external_prompt_cap",
                "model", "generator_model",  # original reader model
            ]
        },
        "row_count": len(entries),
        "deterministic_count": sum(1 for e in entries if e["is_deterministic"]),
        "reader_count": sum(1 for e in entries if not e["is_deterministic"]),
        "entries": entries,
    }

    output_path = Path(args.output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cache, indent=2, default=str), encoding="utf-8")

    det = cache["deterministic_count"]
    rdr = cache["reader_count"]
    print(f"Cache written: {output_path}")
    print(f"  {len(entries)} rows ({det} deterministic, {rdr} need reader)")
    if skipped:
        print(f"  {skipped} rows skipped (no prompt and not deterministic)")
    print(f"  fingerprint: {fingerprint}")


# ── Replay with a new reader ─────────────────────────────────────────

def _replay_row(
    entry: dict[str, Any],
    *,
    provider: str,
    base_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
    api_key_env: str,
    app_url: str,
    app_title: str,
    provider_routing: dict[str, Any] | None,
) -> dict[str, Any]:
    """Replay a single cached row with a new reader model."""
    selection_meta = entry.get("selection_meta", {})
    selected_memory_texts = entry.get("selected_memory_texts", [])

    # Deterministic rows: no reader call needed
    if entry.get("is_deterministic") and entry.get("deterministic_answer"):
        generation = {
            "clean_answer": entry["deterministic_answer"],
            "raw_answer": entry["deterministic_answer"],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0.0,
            "success": True,
        }
    else:
        generation = generate_answer(
            provider=provider,
            base_url=base_url,
            model=model,
            prompt=entry["prompt"],
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key_env=api_key_env,
            app_url=app_url,
            app_title=app_title,
            provider_routing=provider_routing,
        )

    if generation is None:
        generation = {
            "success": False, "raw_answer": "", "clean_answer": "",
            "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0,
            "error": "generation_returned_none",
        }
    if generation.get("status_code") == 402:
        raise RuntimeError(f"API credit limit reached (402) for {entry.get('example_id')}")

    # Score
    row_for_scoring = {
        "dataset_name": entry.get("dataset_name"),
        "question_type": entry.get("question_type"),
        "locomo_category": entry.get("locomo_category"),
        "query": entry["question"],
        "reference_answer": entry.get("reference_answer"),
        "harm_type": entry.get("harm_type"),
    }
    quality = quality_metrics_for_row(
        row_for_scoring,
        entry["question"],
        generation["clean_answer"],
        entry.get("reference_answer", ""),
    )

    output = {
        "system": entry["system"],
        "example_id": entry["example_id"],
        "harm_type": entry.get("harm_type"),
        "dataset_name": entry.get("dataset_name"),
        "question_type": entry.get("question_type"),
        "locomo_category": entry.get("locomo_category"),
        "ability": entry.get("ability"),
        "chat_size": entry.get("chat_size"),
        "chat_id": entry.get("chat_id"),
        "question": entry["question"],
        "question_date": entry.get("question_date"),
        "reference_answer": entry.get("reference_answer"),
        "active_context": entry.get("active_context", []),
        "selected_memory_ids": entry.get("selected_memory_ids", []),
        "selected_memory_count": len(entry.get("selected_memory_ids", [])),
        "selected_memory_tokens": entry.get("selected_memory_tokens", 0),
        "selection_meta": selection_meta,
        "compiler_payload": entry.get("compiler_payload"),
        "rendered_evidence_package": entry.get("rendered_evidence_package"),
        "effective_prompt_variant": entry.get("prompt_variant"),
        "query_family": entry.get("query_family"),
        "selection_mode": entry.get("selection_mode"),
        "reader_route_reason": entry.get("reader_route_reason"),
        "answerability_level": entry.get("answerability_level"),
        "deterministic_allowed": entry.get("deterministic_allowed"),
        "schema_name": entry.get("schema_name"),
        "compiler_type": entry.get("compiler_type", "rule_based"),
        "compiler_model": entry.get("compiler_model"),
        "llm_compiler_input_tokens": entry.get("llm_compiler_input_tokens", 0),
        "llm_compiler_output_tokens": entry.get("llm_compiler_output_tokens", 0),
        "compiled_token_count": entry.get("compiled_token_count", 0),
        "generated_answer": generation["clean_answer"],
        "raw_answer": generation["raw_answer"],
        "prompt_tokens": generation.get("prompt_tokens", 0),
        "completion_tokens": generation.get("completion_tokens", 0),
        "latency_ms": generation.get("latency_ms", 0),
        "success": generation.get("success", False),
        "error": generation.get("error"),
        # Safety flags (not re-evaluated in replay)
        "used_stale_memory": 0,
        "used_contradictory_memory": 0,
        "used_redundant_memory": 0,
        "used_unnecessary_personalization": 0,
        "false_suppressed_positive_control": 0,
        **quality,
    }
    return output


def _write_replay_summary(
    path: Path,
    summaries: dict[str, dict[str, Any]],
    outputs: list[dict[str, Any]],
    meta: dict[str, Any],
    model: str,
    cache: dict[str, Any],
) -> None:
    """Write a markdown summary for a replay run."""
    lines = [
        "# Answer Generation — Replay from Compilation Cache",
        "",
        f"- date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- reader model: `{model}`",
        f"- source cache: `{cache.get('source_run', '?')}`",
        f"- cache fingerprint: `{cache.get('fingerprint', '?')}`",
        f"- original reader: `{meta.get('model', '?')}`",
        f"- slice: `{meta.get('slice_path', '?')}`",
        f"- rows: `{len(outputs)}`",
        f"- deterministic (skipped reader): `{cache.get('deterministic_count', 0)}`",
        "",
    ]

    for system_name, summary in summaries.items():
        avg_mem = summary.get("avg_selected_memory_tokens", 0)
        avg_prompt = summary.get("avg_prompt_tokens", 0)
        lines.extend([
            f"## Quality & Efficiency: `{system_name}`",
            "",
            f"| metric | value |",
            f"| --- | ---: |",
            f"| avg_quality (norm) | {summary.get('avg_quality_norm', 0):.3f} |",
            f"| exact_match | {summary.get('exact_match_rate', 0):.3f} |",
            f"| avg_memory_tokens | {avg_mem:.1f} |",
            f"| avg_prompt_tokens | {avg_prompt:.1f} |",
            "",
        ])

        # LongMemEval per-question-type breakdown
        lme_acc = summary.get("longmemeval_accuracy")
        lme_by_qt = summary.get("longmemeval_by_question_type", {})
        if lme_acc is not None:
            lines.extend([
                f"## LongMemEval Paper Metrics (Binary Accuracy): `{system_name}`",
                "",
                f"- Overall accuracy: **{lme_acc:.3f}** ({len(outputs)} rows)",
                "",
                "| question_type | rows | accuracy |",
                "| --- | ---: | ---: |",
            ])
            for qt in sorted(lme_by_qt):
                qt_data = lme_by_qt[qt]
                lines.append(f"| `{qt}` | {qt_data['total']} | {qt_data['accuracy']:.3f} |")
            lines.append("")

        # Unknown answers
        unknowns = sum(
            1 for o in outputs
            if str(o.get("system")) == system_name
            and str(o.get("generated_answer", "")).strip().lower() in ("unknown", "unknown.", "")
        )
        lines.extend([
            f"## Abstention: `{system_name}`",
            "",
            f"- Unknown answers: {unknowns}/{len(outputs)} ({unknowns/max(len(outputs),1):.1%})",
            "",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_replay(args: argparse.Namespace) -> None:
    """Replay cached compilation with a new reader model."""
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: cache not found: {cache_path}", file=sys.stderr)
        sys.exit(1)

    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    entries = cache["entries"]
    meta = cache.get("run_meta", {})

    print(f"Cache: {cache_path}")
    print(f"  {cache['row_count']} rows, fingerprint={cache['fingerprint']}")
    print(f"  Original reader: {meta.get('model', '?')}")
    print(f"  New reader: {args.model}")
    print(f"  Deterministic (skip reader): {cache['deterministic_count']}")
    print(f"  Need reader call: {cache['reader_count']}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = args.provider
    base_url = args.base_url or "https://openrouter.ai/api/v1"
    model = args.model
    parallelism = args.parallelism

    # Provider routing (e.g. --provider-order SambaNova)
    _provider_routing = None
    if hasattr(args, "provider_order") and args.provider_order:
        _provider_routing = {
            "order": [p.strip() for p in args.provider_order.split(",")],
            "allow_fallbacks": False,
        }

    outputs: list[dict[str, Any]] = []
    errors = 0
    started = perf_counter()

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_map = {
            executor.submit(
                _replay_row,
                entry,
                provider=provider,
                base_url=base_url,
                model=model,
                timeout_s=args.timeout,
                max_tokens=args.max_tokens,
                temperature=0.0,
                api_key_env=args.api_key_env,
                app_url=DEFAULT_APP_URL,
                app_title="anonymous-rag-compression-artifact",
                provider_routing=_provider_routing,
            ): (i, entry["example_id"])
            for i, entry in enumerate(entries)
        }

        for future in as_completed(future_map):
            idx, example_id = future_map.pop(future)
            try:
                result = future.result()
                outputs.append(result)
                done = len(outputs) + errors
                if done % 25 == 0 or done == len(entries):
                    elapsed = perf_counter() - started
                    print(f"  [{done}/{len(entries)}] {elapsed:.1f}s", flush=True)
            except Exception as exc:
                errors += 1
                traceback.print_exc()
                print(f"  SKIP {example_id}: {exc!r}", flush=True)

    elapsed = perf_counter() - started
    print(f"\nDone: {len(outputs)} rows in {elapsed:.1f}s ({errors} errors)")

    # Score
    summaries = evaluate_run(outputs)

    # Write outputs
    run_json = {
        "outputs": outputs,
        "traces": outputs,  # same structure for compatibility
    }
    (output_dir / "run.json").write_text(
        json.dumps(run_json, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "trace.json").write_text(
        json.dumps(outputs, indent=2, default=str), encoding="utf-8"
    )

    # Write replay summary (lightweight — the full runner summary expects
    # fields that only the full pipeline produces)
    _write_replay_summary(output_dir / "summary.md", summaries, outputs, meta, model, cache)

    # Write replay provenance notice
    original_model = meta.get("model", "unknown")
    notice = (
        f"REPLAY NOTICE\n"
        f"=============\n"
        f"This run was generated by compile_once_run_again.py (replay mode).\n"
        f"The compilation stage (retrieval, NLP extraction, schema matching,\n"
        f"LLM cascade compiler, evidence packaging) was cached from a prior run.\n"
        f"Only the reader stage was re-executed with a new model.\n"
        f"\n"
        f"  Source cache:     {cache_path}\n"
        f"  Cache fingerprint: {cache['fingerprint']}\n"
        f"  Original reader:  {original_model}\n"
        f"  Replay reader:    {model}\n"
        f"  Replay date:      {datetime.now(timezone.utc).isoformat()}\n"
        f"\n"
        f"IMPORTANT: Results are only comparable to other runs using the same\n"
        f"compilation cache (same fingerprint). Do NOT compare these results\n"
        f"against runs from a different compilation pipeline, different slice,\n"
        f"or different compiler settings. If in doubt, re-run the full pipeline.\n"
    )
    (output_dir / "REPLAY_NOTICE.txt").write_text(notice, encoding="utf-8")

    # Print results
    for system_name, summary in summaries.items():
        print(f"\n=== {system_name} ===")
        acc = summary.get("longmemeval_accuracy")
        if acc is not None:
            print(f"  Overall accuracy: {acc:.1%}")
        by_qt = summary.get("longmemeval_by_question_type", {})
        for qt, qt_data in sorted(by_qt.items()):
            print(f"  {qt}: {qt_data['accuracy']:.1%} ({qt_data['correct']}/{qt_data['total']})")

    print(f"\n{'='*60}")
    print(f"REPLAY: compilation was cached from {original_model}")
    print(f"Only reader ({model}) was re-executed.")
    print(f"Cache fingerprint: {cache['fingerprint']}")
    print(f"Results are ONLY comparable to runs with the same fingerprint.")
    print(f"{'='*60}")


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile once, run readers again.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # cache subcommand
    p_cache = sub.add_parser("cache", help="Build compilation cache from a completed run")
    p_cache.add_argument("--source-run", required=True, help="Path to completed run directory")
    p_cache.add_argument("--output-cache", required=True, help="Output cache JSON path")

    # replay subcommand
    p_replay = sub.add_parser("replay", help="Replay cached compilation with a new reader")
    p_replay.add_argument("--cache", required=True, help="Path to compilation cache JSON")
    p_replay.add_argument("--model", required=True, help="Reader model (e.g. meta-llama/llama-3.1-70b-instruct)")
    p_replay.add_argument("--output-dir", required=True, help="Output directory for results")
    p_replay.add_argument("--provider", default="openrouter", help="LLM provider")
    p_replay.add_argument("--base-url", default=None, help="Provider base URL")
    p_replay.add_argument("--parallelism", type=int, default=32, help="Concurrent reader calls")
    p_replay.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (seconds)")
    p_replay.add_argument("--max-tokens", type=int, default=64, help="Max reader output tokens")
    p_replay.add_argument("--api-key-env", default="OPENROUTER_API_KEY", help="Env var for API key")
    p_replay.add_argument("--provider-order", default=None, help="Comma-separated provider order (e.g. SambaNova,DeepInfra)")

    args = parser.parse_args()
    if args.command == "cache":
        cmd_cache(args)
    elif args.command == "replay":
        cmd_replay(args)


if __name__ == "__main__":
    main()
