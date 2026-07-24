# SIEVE Experiment Repo

This is a NEW paper building on top of SIEVE, the post-retrieval evidence compilation method. The original conversational memory evaluation paper is already published at EMNLP 2025 Findings. We are NOT rewriting that paper. This repo is for a standalone SIEVE method paper that extends and improves the compiler, targeting a main conference venue.

## What this is

SIEVE compresses BM25-retrieved conversation memories into structured evidence before passing to a reader LLM. The original finding (from the EMNLP Findings paper): compression helps small readers (+13pp) but becomes a coin flip for strong readers (+0.8pp). This repo extends that work with learned compilers, GPU-accelerated retrieval, and reader-adaptive compilation to close the gap for strong readers.

## Running experiments

```bash
# Full pipeline: retrieve + compile + read
python run_answer_generation_phase1.py \
  --slice dataset-slices/longmemeval_bm25_top20_v1.jsonl \
  --controller-config configs/controller_v0.json \
  --provider openrouter \
  --model meta-llama/llama-3.1-8b-instruct \
  --systems bm25_sieve \
  --parallelism 32

# Replay cached compilation with a new reader (no recompilation cost)
python compile_once_run_again.py replay \
  --cache data/compilation_cache/paper_sieve.json \
  --model <new-model-id> \
  --output-dir results/answer_generation_runs/<run_name> \
  --parallelism 32

# Score with LLM judge
python scoring_judge.py \
  --run-dir results/answer_generation_runs/<run_dir> \
  --provider openrouter \
  --model deepseek/deepseek-chat-v3-0324 \
  --parallelism 32
```

## Key files to modify

- `assembly_methods/compiler/execution/selection.py` -- main evidence selection logic
- `assembly_methods/compiler/planner/planner.py` -- schema selection
- `assembly_methods/compiler/schemas/*.py` -- add new schema types
- `assembly_methods/llm_compiler.py` -- LLM compilation fallback
- `configs/controller_v0.json` -- signal weights and thresholds

## Canonical results locations

- `data/compilation_cache/` -- Pre-compiled evidence from the paper (paper_sieve.json, paper_naive.json). These are the gold compilation outputs; replay new readers against them with compile_once_run_again.py.
- `results/answer_generation_runs/<run_name>/` -- Each experiment run gets its own directory with per-query answers and scores.
- `dataset-slices/` -- Benchmark slices (LongMemEval, HotpotQA) in JSONL format. Source of truth for evaluation queries.

All files in data/, results/, and dataset-slices/ are tracked by git LFS.

## Python code standards

These rules apply to all Python in this repo. Follow them when writing new code or refactoring existing code.

### File and function limits

- **Max 300 lines per file.** If a file exceeds this, split it by responsibility.
- **Max 50 lines per function.** If a function exceeds this, decompose into named helpers.
- **Max 4 levels of nesting.** If deeper, extract the inner block into a function.
- **Max 5 parameters per function.** Group related params into a dataclass or TypedDict if more are needed.

### No magic numbers or strings

- Every numeric threshold, weight, or limit must be a named constant at module level or in a config file.
- String literals used as keys, modes, or categories must be constants or enums.
- Hardcoded paths, URLs, and API keys are forbidden. Use constants, config files, or CLI args.

### Imports

- All imports at module top level. No inline/deferred imports unless guarding an optional dependency (document why with a comment).
- No wildcard imports (`from x import *`).
- No circular imports. If two modules need each other, extract shared code into a third.

### Error handling

- No bare `except Exception: pass`. Always log the exception or re-raise.
- No silent `.get()` with default `0` for keys that should exist. If a key is expected, access it directly and let it fail loud.

### Comments and docstrings

- No stale, TODO, or "work-in-progress" comments. Delete or fix.
- No comments that restate what the code does. Only comment the WHY when non-obvious.
- No multi-paragraph docstrings. One line max.

### Data and constants

- Mutable module-level collections (`set`, `dict`, `list`) used as constants must be `frozenset`, `MappingProxyType`, or `tuple`.
- Dataset-specific vocabularies (entity lists, concept specs) go in `configs/*.json`, not in Python source.
- Constants used by multiple modules go in one canonical location and are imported everywhere.

### Entry points

- Every script must have an `if __name__ == "__main__":` guard.
- Shared CLI args (`--provider`, `--model`, `--parallelism`, etc.) must use a shared `add_common_args(parser)` helper, not copy-pasted argparse blocks.

### Type safety

- All function signatures must have type annotations.
- Use `from __future__ import annotations` in every file.
- Prefer dataclasses over raw dicts for structured data with known keys.

### Testing

- New code must have tests. No exceptions.
- Tests go in `tests/` mirroring the source tree (e.g., `tests/assembly_methods/test_common.py`).

### Thread safety

- No module-level mutable dicts used as caches under ThreadPoolExecutor without a lock or `functools.lru_cache`.

## Provenance

Forked from `memory-eligibility-feasibility` repo (2026-07-13). Original results preserved there.

## Concurrent work (active, do not cross-contaminate)

The prior paper "Fixed RAG Compression Collapses Measured Reader Scaling" (arXiv 2606.21807, `../memory-eligibility-feasibility/paper/`) is UNPUBLISHED and under active ARR revision/resubmission (May 2026 cycle scored 2/3/3; reviews on OpenReview, submission 12575). Papers in this repo (`paper.md` long draft, `short-paper.md` ARR short for EACL 2027) build on that paper's ragscale dataset but must not re-claim it as a contribution or share any table/figure/number with it. Rules: (1) cite it as the dataset source and the claim under test; (2) declare it as related anonymous concurrent work on any ARR submission form; (3) the reader-conditioning impossibility result belongs to `short-paper.md` — do not also add it to the prior paper's revision. Data provenance: `data/router_matrix/v0.parquet` (191,277 cells, built 2026-07-14) ingests the prior paper's published runs (validation gate in `data/router_matrix/v0_summary.md` matches published numbers) PLUS post-freeze additions: ConvoMem and LoCoMo slices, LME dense/oracle slices, readers mimo-v2.5 and qwen3.6-27b, and additional summary cells.
