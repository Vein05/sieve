#!/usr/bin/env python3
"""Run answer-generation phase 1 on a selected evaluation slice."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
import json
from pathlib import Path
import re

from reader.client import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OPENROUTER_BASE_URL
from reader.selector import DEFAULT_SYSTEMS
from reader.runner import run_phase1, write_json, write_summary_markdown, write_trace_json


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "run"


def _default_slice_run_label(slice_path: Path) -> str:
    stem = slice_path.stem
    if stem == "longmemeval_controller_full_500_v1":
        return f"deprecated_{stem}"
    return stem


def _default_run_dir(slice_path: Path, provider: str, model: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model_slug = _slugify(model.split("/")[-1].replace(":", "-"))
    provider_slug = _slugify(provider)
    slice_label = _default_slice_run_label(slice_path)
    return Path("results/answer_generation_runs") / f"{stamp}-{slice_label}-{provider_slug}-{model_slug}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the answer-generation phase 1 matrix.",
        epilog=(
            "For full paper-facing runs, prefer "
            "--provider openrouter --model openai/gpt-4.1-mini."
        ),
    )
    parser.add_argument(
        "--slice",
        default="dataset-slices/kill_test_v1.jsonl",
        help=(
            "Path to the evaluation slice "
            "(default: historical smoke slice dataset-slices/kill_test_v1.jsonl)"
        ),
    )
    parser.add_argument(
        "--controller-config",
        default="configs/controller_v0.json",
        help="Path to the controller config (default: configs/controller_v0.json)",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openrouter"],
        default="ollama",
        help="Generation provider (default: ollama)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional provider base URL. Defaults to the standard URL for the selected provider.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4.1-mini",
        help="Generator model (default: openai/gpt-4.1-mini)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Generation timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum completion tokens (default: 64)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0)",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=[
            "baseline",
            "tail_focus_v1",
        ],
        default="baseline",
        help="Prompt variant to use without changing controller selection (default: baseline)",
    )
    parser.add_argument(
        "--sieve-reader-mode",
        choices=["calibrated_default", "structured_default", "compact_ablation"],
        default="calibrated_default",
        help="Reader routing for sieve rows that go through slots (default: calibrated_default)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of slice rows for a smoke run",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help="Optional split name to evaluate only a subset of rows, for example: test",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=32,
        help="How many generation requests to run in parallel (default: 32)",
    )
    parser.add_argument(
        "--systems",
        default=",".join(DEFAULT_SYSTEMS),
        help="Comma-separated assembly systems to run (default: the standardized four-method set).",
    )
    parser.add_argument(
        "--external-prompt-cap",
        type=int,
        default=None,
        help="Evidence token budget (memory/evidence tokens only). System prompt overhead is measured per-row and added automatically. E.g. --external-prompt-cap 400 = 400 tokens for evidence.",
    )
    parser.add_argument(
        "--no-pool-aug",
        action="store_true",
        default=False,
        help="Disable pool augmentation — reader sees only compiled evidence, no raw candidate pool.",
    )
    parser.add_argument(
        "--cascade-model",
        default=None,
        help="LLM compiler model for cascade fallback (e.g., qwen/qwen3-8b). None = rule-based only.",
    )
    parser.add_argument(
        "--cascade-top-k",
        type=int,
        default=5,
        help="Number of candidates to feed to LLM compiler (default: 5)",
    )
    parser.add_argument(
        "--cascade-provider",
        default="openrouter",
        help="Provider for cascade LLM compiler (default: openrouter)",
    )
    parser.add_argument(
        "--cascade-base-url",
        default=None,
        help="Base URL for cascade LLM compiler. Defaults to provider standard URL.",
    )
    parser.add_argument(
        "--learned-pruner-train-slices",
        default=None,
        help="Optional comma-separated JSONL slices used to train selector-v1 and pruner-v1 models.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable name holding the OpenRouter API key (default: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--app-url",
        default="https://anonymous.4open.science/r/from-reliable-to-random-BB62",
        help="HTTP-Referer header sent to OpenRouter",
    )
    parser.add_argument(
        "--app-title",
        default="anonymous-rag-compression-artifact",
        help="X-Title header sent to OpenRouter",
    )
    parser.add_argument(
        "--provider-routing-json",
        default=None,
        help="Optional JSON object passed to OpenRouter as the provider routing config.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional raw output JSON path",
    )
    parser.add_argument(
        "--summary-md",
        default=None,
        help="Optional markdown summary path",
    )
    parser.add_argument(
        "--trace-json",
        default=None,
        help="Optional trace JSON path. Defaults to a sibling trace.json next to the run JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional run artifact directory. When set, writes run.json, trace.json, and summary.md inside it.",
    )
    parser.add_argument(
        "--progress-log",
        default=None,
        help="Optional progress log path. Defaults to progress.log next to the run artifacts.",
    )
    args = parser.parse_args()

    slice_path = Path(args.slice)
    if args.output_dir:
        run_dir = Path(args.output_dir)
    elif not args.output_json and not args.summary_md and not args.trace_json:
        run_dir = _default_run_dir(slice_path, args.provider, args.model)
    else:
        run_dir = None

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        output_json = run_dir / "run.json"
        trace_json = run_dir / "trace.json"
        summary_md = run_dir / "summary.md"
    else:
        output_json = Path(args.output_json) if args.output_json else Path("results/answer_generation_runs") / f"{slice_path.stem}-phase1.json"
        summary_md = Path(args.summary_md) if args.summary_md else output_json.with_name("summary.md")
        trace_json = Path(args.trace_json) if args.trace_json else output_json.with_name("trace.json")
    progress_log = Path(args.progress_log) if args.progress_log else output_json.with_name("progress.log")
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    progress_log.write_text("", encoding="utf-8")
    base_url = args.base_url
    if not base_url:
        base_url = DEFAULT_OLLAMA_BASE_URL if args.provider == "ollama" else DEFAULT_OPENROUTER_BASE_URL
    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    training_slices = None
    if args.learned_pruner_train_slices:
        training_slices = [
            item.strip()
            for item in args.learned_pruner_train_slices.split(",")
            if item.strip()
        ]
    provider_routing = None
    if args.provider_routing_json:
        provider_routing = json.loads(args.provider_routing_json)

    run_start = time.perf_counter()
    payload = run_phase1(
        slice_path=slice_path,
        controller_config_path=Path(args.controller_config),
        provider=args.provider,
        base_url=base_url,
        model=args.model,
        timeout_s=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        parallelism=args.parallelism,
        api_key_env=args.api_key_env,
        app_title=args.app_title,
        prompt_variant=args.prompt_variant,
        sieve_reader_mode=args.sieve_reader_mode,
        max_examples=args.max_examples,
        eval_split=args.eval_split,
        systems=systems,
        training_slice_paths=training_slices,
        provider_routing=provider_routing,
        external_prompt_cap=args.external_prompt_cap,
        progress_log_path=progress_log,
        pool_augmentation=not args.no_pool_aug,
        cascade_compiler_model=args.cascade_model,
        cascade_compiler_top_k=args.cascade_top_k,
        cascade_compiler_provider=args.cascade_provider,
        cascade_compiler_base_url=args.cascade_base_url or (
            DEFAULT_OPENROUTER_BASE_URL if args.cascade_provider == "openrouter" else args.cascade_base_url
        ),
    )
    run_wall_seconds = time.perf_counter() - run_start

    write_json(output_json, payload)
    write_trace_json(trace_json, payload)
    write_summary_markdown(summary_md, payload)

    print(f"Wrote raw output to: {output_json}")
    print(f"Wrote trace output to: {trace_json}")
    print(f"Wrote summary to: {summary_md}")
    print(f"Wrote progress log to: {progress_log}")
    for system in payload["systems"]:
        summary = payload["summaries"][system]
        print(
            f"{system}: avg_quality_norm={summary['avg_quality_norm']}, "
            f"exact_correct={summary['exact_correct_rate']}, "
            f"avg_memory_tokens={summary['avg_selected_memory_tokens']}, "
            f"avg_prompt_tokens={summary['avg_prompt_tokens']}, "
            f"avg_latency_ms={summary['avg_latency_ms']}"
        )

    # Write performance.json alongside the other run artefacts.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from analysis.perf_report import build_performance_report, write_performance_json
        perf_outputs = list(payload.get("outputs") or [])
        perf_report = build_performance_report(
            outputs=perf_outputs,
            wall_total_seconds=run_wall_seconds,
            slice_name=slice_path.name,
            worker_count=args.parallelism,
            extra={
                "provider": args.provider,
                "model": args.model,
                "prompt_variant": args.prompt_variant,
            },
        )
        perf_path = write_performance_json(output_json.parent, perf_report)
        print(f"Wrote performance log to: {perf_path}")
    except Exception as exc:
        print(f"[perf] Could not write performance.json: {exc}")


if __name__ == "__main__":
    main()
