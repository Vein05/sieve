#!/usr/bin/env python3
"""Thin CLI over v2/runner.py: compile the variant cache, then replay readers.

Two subcommands mirror the compile-once / replay-many discipline of
``cli/replay.py``:

  # 1. Build the variant cache (styles x budgets x slice), compile once.
  python -m cli.run_v2 compile \\
      --slice dataset-slices/longmemeval_bm25_top20_v1.jsonl \\
      --styles raw,filtered_raw,extractive,structured,structured_quotes \\
      --budgets 200,400,800 \\
      --config v2/configs/response_surface_v0.json \\
      --out data/v2_variant_cache/response_surface_v0.json

  # 2. Replay a reader panel against the cache (cost-guarded; --yes to run).
  python -m cli.run_v2 replay \\
      --cache data/v2_variant_cache/response_surface_v0.json \\
      --slice dataset-slices/longmemeval_bm25_top20_v1.jsonl \\
      --config v2/configs/response_surface_v0.json \\
      --readers meta-llama/llama-3.1-8b-instruct,meta-llama/llama-3.1-70b-instruct \\
      --run-dir results/v2_runs/response_surface_v0 \\
      --yes

The ``summary`` style needs a compile-LLM: pass --compile-model / --provider and
it is wired from reader.client. It is never invoked in tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from v2.runner import (
    GenerationConfig,
    ReplayRequest,
    build_replay_plan,
    format_cost_plan,
    replay,
)
from v2.splits import SplitConfig, split_for_example
from v2.variant_cache import (
    VariantBuildSpec,
    build_and_save,
    load_variant_cache,
    slice_fingerprint,
)

# Thinking models (e.g. qwen3-8b) otherwise spend the whole max_tokens budget
# on reasoning and return truncated near-empty summaries.
COMPILE_REASONING: dict[str, str] = {"effort": "none"}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().strip().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cache_entries(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load a v1 compilation cache (paper_sieve/paper_naive) as example_id -> entry."""
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(e.get("example_id")): e for e in data.get("entries", [])}


def _make_compile_llm(
    *, model: str, provider: str, base_url: str, api_key_env: str
) -> Callable[[str, int], dict[str, Any]]:
    """Wire a compile-LLM callable from reader.client (for the summary style).

    IMPORTANT: this only *constructs* the callable. It is never invoked in the
    test-suite (tests pass a fake). The import is local so importing this CLI
    module does not pull in the client at module load.
    """
    from reader.client import generate_answer

    def _compile_llm(prompt: str, max_tokens: int) -> dict[str, Any]:
        gen = generate_answer(
            provider=provider,
            base_url=base_url,
            model=model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=int(max_tokens),
            api_key_env=api_key_env,
            reasoning=dict(COMPILE_REASONING),
        )
        if not gen.get("success"):
            raise RuntimeError(f"Summary compiler failed: {gen.get('error')}")
        return {
            "text": str(gen.get("clean_answer") or gen.get("raw_answer") or ""),
            "prompt_tokens": gen.get("prompt_tokens"),
            "completion_tokens": gen.get("completion_tokens"),
            "latency_ms": gen.get("latency_ms"),
        }

    return _compile_llm


def _compile_callable(
    args: argparse.Namespace,
    styles: list[str],
    config: dict[str, Any],
) -> Callable[[str, int], dict[str, Any]] | None:
    if "summary" not in styles:
        return None
    if not args.compile_model:
        raise ValueError(
            "--styles includes 'summary' but --compile-model was not given"
        )
    configured = dict(config.get("compile_model", {}))
    if args.compile_model != configured.get("model"):
        raise ValueError(
            "--compile-model must match config.compile_model.model"
        )
    if args.provider != configured.get("provider"):
        raise ValueError("--provider must match config.compile_model.provider")
    return _make_compile_llm(
        model=args.compile_model,
        provider=args.provider,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------

def cmd_compile(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    rows = _select_evaluation_rows(
        _load_jsonl(Path(args.slice), limit=args.limit), config
    )

    sieve_entries = _load_cache_entries(Path(args.sieve_cache) if args.sieve_cache else None)
    naive_entries = _load_cache_entries(Path(args.naive_cache) if args.naive_cache else None)

    try:
        compile_llm = _compile_callable(args, styles, config)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    spec = VariantBuildSpec(
        styles=styles,
        budgets=budgets,
        config=config,
        sieve_entries=sieve_entries,
        naive_entries=naive_entries,
        compile_llm=compile_llm,
    )
    path, fingerprint, rebuilt = build_and_save(
        args.out,
        rows,
        spec,
        force=args.force,
    )
    built_cache = load_variant_cache(path)
    n_variants = len(built_cache.entries)
    print(f"Variant cache: {path}")
    print(f"  fingerprint : {fingerprint}")
    print(f"  rows        : {len(rows)}")
    print(f"  styles      : {styles}")
    print(f"  budgets     : {budgets}")
    print(f"  variants    : {n_variants}")
    print(f"  rebuilt     : {rebuilt} ({'built' if rebuilt else 'up-to-date, no-op'})")


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def _annotate_splits(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    split_data = dict(config.get("evaluation_split", {}))
    split_config = SplitConfig(
        train_fraction=float(split_data.get("train_fraction", 0.6)),
        validation_fraction=float(split_data.get("validation_fraction", 0.2)),
        test_fraction=float(split_data.get("test_fraction", 0.2)),
        seed=int(split_data.get("seed", config.get("seed", 42))),
    )
    for row in rows:
        example_id = str(row.get("example_id", ""))
        row["v2_split"] = split_for_example(example_id, split_config)


def _select_evaluation_rows(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Annotate deterministic splits and honor an optional pilot split."""
    _annotate_splits(rows, config)
    pilot_split = config.get("evaluation_split", {}).get("pilot_split")
    if not pilot_split:
        return rows
    selected = [row for row in rows if row["v2_split"] == pilot_split]
    if not selected:
        raise ValueError(f"Pilot split {pilot_split!r} selected zero rows")
    return selected


def _generation_config(
    args: argparse.Namespace, config: dict[str, Any]
) -> GenerationConfig:
    reader_config = dict(config.get("reader", {}))
    return GenerationConfig(
        provider=args.provider,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        api_key_env=args.api_key_env,
        provider_routing=dict(reader_config.get("provider_routing", {})),
        reasoning=dict(reader_config.get("reasoning", {})),
    )

def cmd_replay(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    rows = _select_evaluation_rows(
        _load_jsonl(Path(args.slice), limit=args.limit), config
    )
    rows_by_id = {str(r.get("example_id")): r for r in rows}
    slice_fp = slice_fingerprint(rows)

    cache = load_variant_cache(
        args.cache, expected_config=config, expected_slice_fp=slice_fp
    )

    readers = [r.strip() for r in args.readers.split(",") if r.strip()]
    plan = build_replay_plan(cache, readers)

    # --- Cost guard: print the plan and require --yes to proceed. ---
    print(format_cost_plan(plan))
    if not args.yes:
        print("\nDry-run only. Re-run with --yes to execute the reader calls.")
        return

    from reader.client import generate_answer

    request = ReplayRequest(
        cache=cache,
        rows_by_id=rows_by_id,
        readers=tuple(readers),
        run_dir=Path(args.run_dir),
        system=args.system,
        generation=_generation_config(args, config),
        parallelism=args.parallelism if args.parallelism else int(config.get("parallelism", 16)),
    )
    run_json = replay(request, generate_answer)
    meta = run_json["meta"]
    print(f"\nReplay complete: {args.run_dir}")
    print(f"  calls run : {meta['n_calls_run']} / planned {meta['n_calls_planned']}")
    print(f"  outputs   : {len(run_json['outputs'])}")
    print(f"  run.json  : {Path(args.run_dir) / 'run.json'}")
    print("  (judge with: python -m cli.judge --run-dir <run_dir>)")


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SIEVE v2 response-surface runner (compile once, replay many).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_c = sub.add_parser("compile", help="Build the variant cache from a slice.")
    p_c.add_argument("--slice", required=True, help="Path to slice JSONL")
    p_c.add_argument("--styles", required=True, help="Comma-separated style names")
    p_c.add_argument("--budgets", default="200,400,800", help="Comma-separated token budgets")
    p_c.add_argument("--config", required=True, help="Path to response_surface config JSON")
    p_c.add_argument("--out", required=True, help="Output variant-cache path")
    p_c.add_argument("--sieve-cache", default=None, help="v1 SIEVE compilation cache (for structured styles)")
    p_c.add_argument("--naive-cache", default=None, help="v1 naive compilation cache")
    p_c.add_argument("--limit", type=int, default=None, help="Only build the first N rows (smoke runs)")
    p_c.add_argument("--force", action="store_true", help="Rebuild even if fingerprint matches")
    p_c.add_argument("--compile-model", default=None, help="Compile-LLM model (required for the 'summary' style)")
    p_c.add_argument("--provider", default="openrouter", help="Compile-LLM provider")
    p_c.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="Compile-LLM base URL")
    p_c.add_argument("--api-key-env", default="OPENROUTER_API_KEY", help="Env var for API key")
    p_c.set_defaults(func=cmd_compile)

    p_r = sub.add_parser("replay", help="Replay a reader panel against the variant cache.")
    p_r.add_argument("--cache", required=True, help="Path to variant cache JSON")
    p_r.add_argument("--slice", required=True, help="Path to slice JSONL (for row metadata + fingerprint)")
    p_r.add_argument("--config", required=True, help="Path to response_surface config JSON")
    p_r.add_argument("--readers", required=True, help="Comma-separated reader model IDs")
    p_r.add_argument("--run-dir", required=True, help="Output run directory")
    p_r.add_argument("--system", default="v2_response_surface", help="System name recorded in outputs")
    p_r.add_argument("--provider", default="openrouter", help="Reader provider")
    p_r.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="Reader base URL")
    p_r.add_argument("--parallelism", type=int, default=0, help="Concurrent reader calls (0 = from config)")
    p_r.add_argument("--max-tokens", type=int, default=64, help="Max reader output tokens")
    p_r.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (seconds)")
    p_r.add_argument("--api-key-env", default="OPENROUTER_API_KEY", help="Env var for API key")
    p_r.add_argument("--limit", type=int, default=None, help="Only replay rows present in the first N slice rows")
    p_r.add_argument("--yes", action="store_true", help="Execute reader calls (default is dry-run)")
    p_r.set_defaults(func=cmd_replay)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
