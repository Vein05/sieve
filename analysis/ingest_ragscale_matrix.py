"""
ingest_ragscale_matrix.py
=========================
Turn judged run outputs from the prior paper into a normalized training
matrix for a reader-conditioned representation router.

Usage
-----
python analysis/ingest_ragscale_matrix.py \
    --runs-dir /path/to/answer_generation_runs \
    --out data/router_matrix \
    [--validate]

Label rule (derived empirically from judge_score_v2 distribution)
-----------------------------------------------------------------
- For conversational-memory QA tasks (longmemeval, hotpotqa, musique, nq,
  scifact, convomem, locomo): judge_score_v2 is BINARY (0 = wrong, 2 = correct).
  Rarely an odd value of 1 appears (malformed judge output); treat 1 as NaN.
  correct = 1 iff judge_score_v2 == 2, else 0.  Missing -> NaN.
- For QMSum task (dataset_name == 'qmsum' or system starts with 'qmsum'):
  judge_score_v2 is a 3-point rubric (0/1/2) — not a binary correctness label.
  correct = NaN (excluded from binary label; task_type = 'qmsum').
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import pyarrow  # noqa: F401
    _HAVE_PARQUET = True
except ImportError:
    _HAVE_PARQUET = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_LABEL_SCORES = {0, 2}   # 1 is malformed; treat as NaN

# Map raw system name -> canonical style
STYLE_MAP: dict[str, str] = {
    "naive_top_k":         "raw",
    "bm25_top_k":          "raw",
    "bm25_sieve":          "structured_v1",
    "sieve":               "structured_v1",
    "llm_summarize":       "summary",
    "recomp_abstractive":  "summary_trained",
    "exit_extractive":     "extractive_trained",
    "provence_extractive": "extractive_trained",
    "llmlingua_v2":        "token_pruned",
    # QMSum variants — keep as-is after stripping 'qmsum_' prefix below
}

# Short alias -> canonical model ID, built from empirical inspection
# (ordered longest-match first within each group to avoid substring collisions)
MODEL_ALIAS_MAP: dict[str, str] = {
    # claude
    "claude-3.5-haiku":      "anthropic/claude-3.5-haiku",
    "claude35haiku":         "anthropic/claude-3.5-haiku",
    # cohere
    "command-r-08-2024":     "cohere/command-r-08-2024",
    "commandr":              "cohere/command-r-08-2024",
    # deepseek
    "deepseek-r1-distill-llama-70b": "deepseek/deepseek-r1-distill-llama-70b",
    "dsr1_llama70b":         "deepseek/deepseek-r1-distill-llama-70b",
    # google gemma
    "gemma-3-12b-it":        "google/gemma-3-12b-it",
    "gemma-3-27b-it":        "google/gemma-3-27b-it",
    "gemma12b":              "google/gemma-3-12b-it",
    "gemma27b":              "google/gemma-3-27b-it",
    "gemma4b":               "google/gemma-3-4b-it",
    # GLM
    "glm-4-32b":             "z-ai/glm-4-32b",
    "glm4_32b":              "z-ai/glm-4-32b",
    # GPT
    "gpt-4.1-mini":          "openai/gpt-4.1-mini",
    "gpt41mini":             "openai/gpt-4.1-mini",
    "gpt4omini":             "openai/gpt-4o-mini",
    "gpt4mini":              "openai/gpt-4.1-mini",  # dense alias
    # Grok
    "grok-4.1-fast":         "x-ai/grok-4.1-fast",
    "grok-3-mini-beta":      "x-ai/grok-3-mini-beta",
    "grok":                  "x-ai/grok-4.1-fast",  # generic alias -> latest known
    # Llama
    "llama-3.1-70b-instruct": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct":  "meta-llama/llama-3.1-8b-instruct",
    "llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
    "llama-4-scout":          "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3-1-70b":          "meta-llama/llama-3.1-70b-instruct",
    "llama70b":               "meta-llama/llama-3.1-70b-instruct",
    "llama8b":                "meta-llama/llama-3.1-8b-instruct",
    "llama33_70b":            "meta-llama/llama-3.3-70b-instruct",
    "llama33":                "meta-llama/llama-3.3-70b-instruct",
    "llama32_3b":             "meta-llama/llama-3.2-3b-instruct",
    "llama4_scout":           "meta-llama/llama-4-scout-17b-16e-instruct",
    # Mimo
    "mimo-v2.5":              "xiaomi/mimo-v2.5",
    "mimo25":                 "xiaomi/mimo-v2.5",
    # Olmo
    "olmo-3.1-32b-instruct":  "allenai/olmo-3.1-32b-instruct",
    "olmo32b":                "allenai/olmo-3.1-32b-instruct",
    # Phi
    "phi-4":                  "microsoft/phi-4",
    "phi4":                   "microsoft/phi-4",
    # Qwen
    "qwen-2.5-72b-instruct":  "qwen/qwen-2.5-72b-instruct",
    "qwen-2.5-7b-instruct":   "qwen/qwen-2.5-7b-instruct",
    "qwen3-14b":              "qwen/qwen3-14b",
    "qwen3-32b":              "qwen/qwen3-32b",
    "qwen3-8b":               "qwen/qwen3-8b",
    "qwen3.6-27b":            "qwen/qwen3.6-27b",
    "qwen72b":                "qwen/qwen-2.5-72b-instruct",
    "qwen7b":                 "qwen/qwen-2.5-7b-instruct",
    "qwen-2-5-7b":            "qwen/qwen-2.5-7b-instruct",
    "qwen36_27b":             "qwen/qwen3.6-27b",
    "qwen3_32b_v2":           "qwen/qwen3-32b",
    "qwen3_32b":              "qwen/qwen3-32b",
    "qwen3_14b_v2":           "qwen/qwen3-14b",
    "qwen3_14b":              "qwen/qwen3-14b",
    "qwen3_8b":               "qwen/qwen3-8b",
    # Seed
    "seed-2.0-mini":          "bytedance-seed/seed-2.0-mini",
    "seed2mini":              "bytedance-seed/seed-2.0-mini",
}

# Sorted by length descending so longest match wins
_ALIAS_SORTED = sorted(MODEL_ALIAS_MAP.keys(), key=lambda x: -len(x))

# Budget patterns to extract from directory name
_BUDGET_PATTERNS = [
    (re.compile(r'_at(\d+)(?:_|$)'),    lambda m: int(m.group(1))),
    (re.compile(r'_cap(\d+)(?:_|$)'),   lambda m: int(m.group(1))),
    (re.compile(r'@(\d+)(?:_|$)'),      lambda m: int(m.group(1))),
    (re.compile(r'_(\d+)(?:_|$)'),      lambda m: int(m.group(1))),  # e.g. llmlingua_50
]

# Slice identifier patterns
_SLICE_PATTERNS = {
    "hotpotqa":  re.compile(r'^(paper_hotpotqa|hotpotqa)'),
    "qmsum":     re.compile(r'^qmsum_'),
    "musique":   re.compile(r'^paper_musique'),
    "nq":        re.compile(r'^paper_nq'),
    "scifact":   re.compile(r'^scifact_'),
    "reinjection_hotpotqa": re.compile(r'^reinjection_hotpotqa'),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_reader_model(run_dir: str, meta: dict, top_level: dict) -> str | None:
    """Best-effort reader model extraction."""
    # 1. Newer meta block
    if meta and meta.get("reader_model"):
        return meta["reader_model"]
    # 2. Older top-level generator_model
    if top_level.get("generator_model"):
        return top_level["generator_model"]
    # 3. Parse from directory name
    name = run_dir.lower()
    # Remove known prefixes
    for prefix in ("paper_hotpotqa_", "paper_musique_", "paper_nq_",
                   "paper_locomo_", "paper_dense_", "paper_convomem_",
                   "reinjection_hotpotqa_", "paper_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Try longest alias match
    for alias in _ALIAS_SORTED:
        if name.startswith(alias):
            return MODEL_ALIAS_MAP[alias]
    # Partial search
    for alias in _ALIAS_SORTED:
        if alias in name:
            return MODEL_ALIAS_MAP[alias]
    return None


def _infer_slice(run_dir: str, meta: dict, top_level: dict) -> str:
    """Infer slice/dataset group from run dir name."""
    for slice_name, pat in _SLICE_PATTERNS.items():
        if pat.search(run_dir):
            return slice_name
    # Check slice_path
    sp = (meta or {}).get("slice") or top_level.get("slice_path") or ""
    sp_lower = sp.lower()
    rd_lower = run_dir.lower()
    if "dense" in rd_lower and rd_lower.startswith("paper_dense_"):
        # paper_dense_* runs use longmemeval_dense_top20 slice, different retrieval context
        return "longmemeval_dense"
    if "locomo" in sp_lower or "locomo" in rd_lower:
        return "locomo"
    if "convomem" in rd_lower:
        return "convomem"
    if "oracle" in rd_lower:
        return "longmemeval_oracle"
    # Default
    return "longmemeval"


def _parse_budget(run_dir: str, meta: dict, top_level: dict) -> int | None:
    """Extract token budget from dir name or top-level external_prompt_cap."""
    cap = (meta or {}).get("prompt_cap") or top_level.get("external_prompt_cap")
    if cap is not None:
        try:
            return int(cap)
        except (ValueError, TypeError):
            pass
    # Parse from dir name
    for pat, extractor in _BUDGET_PATTERNS:
        m = pat.search(run_dir)
        if m:
            val = extractor(m)
            # sanity check: budgets are generally 30-10000 tokens
            if 10 <= val <= 10000:
                return val
    return None


def _map_style(system_raw: str) -> str:
    """Map raw system name to canonical style."""
    # Strip qmsum_ prefix before lookup
    key = system_raw.removeprefix("qmsum_") if system_raw else system_raw
    return STYLE_MAP.get(key, system_raw)


def _get_timestamp(meta: dict, top_level: dict) -> str | None:
    if meta and meta.get("timestamp"):
        return meta["timestamp"]
    if top_level.get("generated_at"):
        return top_level["generated_at"]
    return None


def _is_qmsum(dataset_name: str | None, system_raw: str | None) -> bool:
    if dataset_name and "qmsum" in dataset_name.lower():
        return True
    if system_raw and "qmsum" in system_raw.lower():
        return True
    return False


def _correct_label(row: dict, is_qsum: bool) -> float | None:
    """Derive binary correct label (0/1) or NaN for QMSum."""
    if is_qsum:
        return None  # rubric task — no binary label
    score = row.get("judge_score_v2")
    if score is None:
        return None
    if score == 2:
        return 1
    if score == 0:
        return 0
    # score == 1 or other unexpected value — treat as missing
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_path: Path) -> dict | None:
    """Load and return judge_run_v2.json, or None on error."""
    try:
        with open(run_path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] Failed to load {run_path}: {exc}", file=sys.stderr)
        return None


def normalize_output_row(
    output: dict,
    run_dir: str,
    reader_model: str | None,
    provider: str | None,
    compiler_model_top: str | None,
    slice_name: str,
    budget: int | None,
    timestamp: str | None,
    unknown_systems: set[str],
) -> dict:
    """Convert a single output dict to a normalized flat row."""
    system_raw = output.get("system") or ""
    is_qsum = _is_qmsum(output.get("dataset_name"), system_raw)
    style = _map_style(system_raw)

    if style == system_raw and system_raw and system_raw not in STYLE_MAP:
        # Truly unknown (not qmsum_ prefix either)
        if not system_raw.startswith("qmsum_"):
            unknown_systems.add(system_raw)

    correct = _correct_label(output, is_qsum)

    return {
        "run_name":               run_dir,
        "reader_model":           reader_model,
        "provider":               provider,
        "compiler_model":         output.get("compiler_model") or compiler_model_top,
        "slice_name":             slice_name,
        "dataset_name":           output.get("dataset_name"),
        "system_raw":             system_raw,
        "style":                  style,
        "budget":                 budget,
        "example_id":             output.get("example_id"),
        "question_type":          output.get("question_type"),
        "task_type":              "qmsum" if is_qsum else "qa",
        "correct":                correct,
        "token_f1":               output.get("token_f1"),
        "prompt_tokens":          output.get("prompt_tokens"),
        "completion_tokens":      output.get("completion_tokens"),
        "selected_memory_tokens": output.get("selected_memory_tokens"),
        "latency_ms":             output.get("latency_ms"),
        "judge_score_v2":         output.get("judge_score_v2"),
        "judge_factuality_v2":    output.get("judge_factuality_v2"),
        "judge_completeness_v2":  output.get("judge_completeness_v2"),
        "judge_abstention_v2":    output.get("judge_abstention_v2"),
        "judge_reason_v2":        output.get("judge_reason_v2"),
        "timestamp":              timestamp,
    }


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def ingest(runs_dir: Path) -> tuple[list[dict], dict]:
    """Walk runs_dir and produce normalized rows plus stats."""
    all_rows: list[dict] = []
    stats = {
        "total_dirs": 0,
        "skipped_no_judge": 0,
        "loaded_runs": 0,
        "total_rows_raw": 0,
        "unknown_systems": set(),
        "missing_model_dirs": [],
    }

    dirs = sorted(os.listdir(runs_dir))
    stats["total_dirs"] = len(dirs)

    for run_dir in dirs:
        judge_path = runs_dir / run_dir / "judge_run_v2.json"
        if not judge_path.exists():
            stats["skipped_no_judge"] += 1
            continue

        data = load_run(judge_path)
        if data is None:
            stats["skipped_no_judge"] += 1
            continue

        meta = data.get("meta") or {}
        top_level = {k: v for k, v in data.items() if k not in ("outputs", "traces", "summaries", "meta")}

        reader_model = _infer_reader_model(run_dir, meta, top_level)
        provider = meta.get("provider") or top_level.get("generator_provider")
        compiler_model_top = meta.get("compiler_model") or top_level.get("compiler_model")
        slice_name = _infer_slice(run_dir, meta, top_level)
        budget = _parse_budget(run_dir, meta, top_level)
        timestamp = _get_timestamp(meta, top_level)

        if reader_model is None:
            stats["missing_model_dirs"].append(run_dir)

        outputs = data.get("outputs") or []
        stats["loaded_runs"] += 1
        stats["total_rows_raw"] += len(outputs)

        for output in outputs:
            row = normalize_output_row(
                output=output,
                run_dir=run_dir,
                reader_model=reader_model,
                provider=provider,
                compiler_model_top=compiler_model_top,
                slice_name=slice_name,
                budget=budget,
                timestamp=timestamp,
                unknown_systems=stats["unknown_systems"],
            )
            all_rows.append(row)

    return all_rows, stats


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def dedup_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """
    Deduplicate on (reader_model, style, budget, slice_name, example_id).
    Keep the row with the latest timestamp.  Track disagreement rate.
    """
    key_fn = lambda r: (r["reader_model"], r["style"], r["budget"], r["slice_name"], r["example_id"])

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)

    kept: list[dict] = []
    n_dups = 0
    n_disagree = 0

    def _ts(r: dict) -> datetime:
        ts = r.get("timestamp")
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
        return datetime.min

    for key, group in groups.items():
        if len(group) == 1:
            group[0]["n_duplicates"] = 0
            group[0]["label_disagreement"] = False
            kept.append(group[0])
            continue

        n_dups += len(group) - 1
        # Sort by timestamp desc, keep most recent
        group_sorted = sorted(group, key=_ts, reverse=True)
        best = group_sorted[0]

        # Check label disagreement
        labels = {r["correct"] for r in group if r["correct"] is not None}
        disagrees = len(labels) > 1
        if disagrees:
            n_disagree += 1

        best["n_duplicates"] = len(group) - 1
        best["label_disagreement"] = disagrees
        kept.append(best)

    dedup_stats = {
        "n_duplicates_removed": n_dups,
        "n_disagreement_pairs": n_disagree,
        "total_after_dedup": len(kept),
        "disagreement_rate": n_disagree / len(kept) if kept else 0.0,
    }
    return kept, dedup_stats


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_outputs(rows: list[dict], out_dir: Path) -> Path:
    """Write v0.parquet (or v0.csv.gz) to out_dir. Returns path written."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if _HAVE_PARQUET:
        import pandas as pd
        df = pd.DataFrame(rows)
        out_path = out_dir / "v0.parquet"
        df.to_parquet(out_path, index=False, compression="snappy")
        return out_path
    else:
        import csv, io
        out_path = out_dir / "v0.csv.gz"
        if not rows:
            out_path.write_bytes(b"")
            return out_path
        fieldnames = list(rows[0].keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            fh.write(buf.getvalue())
        return out_path


def write_summary(
    rows: list[dict],
    ingest_stats: dict,
    dedup_stats: dict,
    validation: dict | None,
    out_dir: Path,
) -> None:
    """Write v0_summary.md."""
    from collections import Counter

    def _counter(field: str) -> Counter:
        return Counter(r[field] for r in rows)

    slice_counts  = _counter("slice_name")
    reader_counts = _counter("reader_model")
    style_counts  = _counter("style")

    # Per-(style x slice) grid
    style_slice: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        style_slice[r["style"]][r["slice_name"]] += 1

    all_slices = sorted(slice_counts.keys())
    all_styles = sorted(style_slice.keys())

    # Label balance
    correct_rows  = [r for r in rows if r["correct"] is not None]
    n_correct     = sum(1 for r in correct_rows if r["correct"] == 1)
    n_incorrect   = sum(1 for r in correct_rows if r["correct"] == 0)
    n_qmsum       = sum(1 for r in rows if r["task_type"] == "qmsum")

    # Per-(reader_model, style) accuracy on LongMemEval
    lme_rows = [r for r in rows if r["slice_name"] == "longmemeval" and r["correct"] is not None]
    acc_table: dict[tuple, tuple[int, int]] = defaultdict(lambda: [0, 0])
    for r in lme_rows:
        k = (r["reader_model"], r["style"])
        acc_table[k][0] += int(r["correct"])
        acc_table[k][1] += 1

    lines: list[str] = []
    lines.append("# Router Matrix v0 Summary\n")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z\n")
    lines.append("")
    lines.append("## Ingestion Stats")
    lines.append(f"- Total run dirs scanned: {ingest_stats['total_dirs']}")
    lines.append(f"- Dirs skipped (no judge_run_v2.json): {ingest_stats['skipped_no_judge']}")
    lines.append(f"- Runs loaded: {ingest_stats['loaded_runs']}")
    lines.append(f"- Raw rows before dedup: {ingest_stats['total_rows_raw']}")
    lines.append(f"- Rows after dedup: {dedup_stats['total_after_dedup']}")
    lines.append(f"- Duplicates removed: {dedup_stats['n_duplicates_removed']}")
    lines.append(f"- Disagreement pairs (judge noise estimate): {dedup_stats['n_disagreement_pairs']}")
    lines.append(f"- Disagreement rate: {dedup_stats['disagreement_rate']:.4f}")
    if ingest_stats.get("missing_model_dirs"):
        lines.append(f"- Dirs with unresolvable reader_model ({len(ingest_stats['missing_model_dirs'])}): "
                     + ", ".join(ingest_stats['missing_model_dirs'][:10])
                     + ("..." if len(ingest_stats['missing_model_dirs']) > 10 else ""))
    lines.append("")

    lines.append("## Rows per Slice")
    for sl, cnt in sorted(slice_counts.items()):
        lines.append(f"- {sl}: {cnt:,}")
    lines.append("")

    lines.append("## Reader Models")
    lines.append(f"Total distinct reader_models: {len(reader_counts)}")
    for m, cnt in sorted(reader_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {m}: {cnt:,}")
    lines.append("")

    lines.append("## Style Coverage (rows per style)")
    for st, cnt in sorted(style_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {st}: {cnt:,}")
    lines.append("")

    lines.append("## Per-(Style x Slice) Run Coverage Grid")
    header = "| style | " + " | ".join(all_slices) + " |"
    sep    = "|---|" + "|".join(["---"] * len(all_slices)) + "|"
    lines.append(header)
    lines.append(sep)
    for style in all_styles:
        row_vals = [str(style_slice[style].get(sl, 0)) for sl in all_slices]
        lines.append(f"| {style} | " + " | ".join(row_vals) + " |")
    lines.append("")

    lines.append("## Label Balance")
    lines.append(f"- Correct (1): {n_correct:,}")
    lines.append(f"- Incorrect (0): {n_incorrect:,}")
    lines.append(f"- QMSum (NaN, excluded from binary): {n_qmsum:,}")
    lines.append(f"- Missing label: {len(rows) - len(correct_rows) - n_qmsum:,}")
    lines.append("")

    lines.append("## Unknown Systems Encountered")
    unk = ingest_stats.get("unknown_systems", set())
    if unk:
        for s in sorted(unk):
            lines.append(f"- {s}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Per-(reader_model, style) Accuracy on LongMemEval")
    lines.append("| reader_model | style | n | accuracy |")
    lines.append("|---|---|---|---|")
    for (rm, st), (correct_n, total_n) in sorted(acc_table.items(), key=lambda x: (-x[1][1], x[0])):
        acc = correct_n / total_n if total_n > 0 else float("nan")
        lines.append(f"| {rm} | {st} | {total_n} | {acc:.3f} |")
    lines.append("")

    if validation:
        lines.append("## Validation Gate (Computed vs Published)")
        lines.append("| condition | metric | computed | published | delta | status |")
        lines.append("|---|---|---|---|---|---|")
        for entry in validation.get("results", []):
            lines.append(
                f"| {entry['condition']} | {entry['metric']} | "
                f"{entry['computed']:.3f} | {entry['published']:.3f} | "
                f"{entry['delta']:+.3f} | {entry['status']} |"
            )
        lines.append("")

    out_path = out_dir / "v0_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to {out_path}")


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------

PUBLISHED = {
    # (reader_model, style, slice, budget) -> accuracy
    # budget=None means "uncapped / default run" (no external_prompt_cap in dir name)
    # Note: paper_llama8b_sieve (main LongMemEval sieve for llama-3.1-8b) has no
    # judge_run_v2.json — only trace.json — so the structured_v1 cell will be MISSING.
    ("meta-llama/llama-3.1-8b-instruct", "raw",           "longmemeval", None): 0.278,
    ("meta-llama/llama-3.1-8b-instruct", "structured_v1", "longmemeval", None): 0.410,
    ("openai/gpt-4.1-mini",              "raw",           "longmemeval", None): 0.438,
    ("openai/gpt-4.1-mini",              "structured_v1", "longmemeval", None): 0.534,
}


def validate(rows: list[dict]) -> dict:
    """
    Compute accuracy for the validation gate rows and compare to published.
    Returns a dict with results list.
    """
    results = []
    tolerance = 0.015

    for (model, style, slice_name, budget), published_acc in PUBLISHED.items():
        subset = [
            r for r in rows
            if r.get("reader_model") == model
            and r.get("style") == style
            and r.get("slice_name") == slice_name
            and r.get("correct") is not None
            # When published budget is None, match only truly uncapped rows (budget IS None).
            # When published budget is specified, match that exact budget value.
            and (r.get("budget") is None if budget is None else r.get("budget") == budget)
        ]

        if not subset:
            results.append({
                "condition": f"{model} / {style}",
                "metric": f"accuracy@{slice_name}" + (f"_b{budget}" if budget else ""),
                "computed": float("nan"),
                "published": published_acc,
                "delta": float("nan"),
                "n": 0,
                "status": "MISSING",
            })
            continue

        n = len(subset)
        acc = sum(r["correct"] for r in subset) / n
        delta = acc - published_acc
        status = "OK" if abs(delta) <= tolerance else "FAIL"
        results.append({
            "condition": f"{model} / {style}",
            "metric": f"accuracy@{slice_name}" + (f"_b{budget}" if budget else ""),
            "computed": acc,
            "published": published_acc,
            "delta": delta,
            "n": n,
            "status": status,
        })

    return {"results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest judged RAG run outputs into a normalized router training matrix."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Path to the answer_generation_runs directory (READ-ONLY)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/router_matrix"),
        help="Output directory for matrix and summary (default: data/router_matrix)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the validation gate after ingestion",
    )
    args = parser.parse_args(argv)

    runs_dir = args.runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        print(f"ERROR: runs-dir does not exist: {runs_dir}", file=sys.stderr)
        return 1

    out_dir = args.out.expanduser().resolve()

    print(f"Ingesting from: {runs_dir}")
    print(f"Output to:      {out_dir}")
    print()

    # --- Ingest ---
    rows, ingest_stats = ingest(runs_dir)
    print(f"Loaded {ingest_stats['loaded_runs']} runs, {ingest_stats['total_rows_raw']:,} raw rows")
    print(f"Skipped {ingest_stats['skipped_no_judge']} dirs (no judge_run_v2.json)")
    if ingest_stats["unknown_systems"]:
        print(f"Unknown systems: {sorted(ingest_stats['unknown_systems'])}")
    if ingest_stats["missing_model_dirs"]:
        print(f"Missing reader_model in {len(ingest_stats['missing_model_dirs'])} dirs")

    # --- Dedup ---
    rows, dedup_stats = dedup_rows(rows)
    print(f"After dedup: {dedup_stats['total_after_dedup']:,} rows "
          f"({dedup_stats['n_duplicates_removed']} dups removed, "
          f"{dedup_stats['n_disagreement_pairs']} disagreements, "
          f"rate={dedup_stats['disagreement_rate']:.4f})")

    # --- Validate ---
    validation = None
    if args.validate:
        print("\n--- Validation Gate ---")
        validation = validate(rows)
        all_ok = True
        for entry in validation["results"]:
            status = entry["status"]
            if status != "OK":
                all_ok = False
            print(
                f"  {entry['condition']:50s}  computed={entry['computed']:.3f}  "
                f"published={entry['published']:.3f}  delta={entry['delta']:+.3f}  "
                f"n={entry['n']}  [{status}]"
            )
        if all_ok:
            print("Validation PASSED")
        else:
            print("Validation FAILED (check label rule or dedup)")

    # --- Write ---
    out_path = write_outputs(rows, out_dir)
    print(f"\nMatrix written to: {out_path}")
    write_summary(rows, ingest_stats, dedup_stats, validation, out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
