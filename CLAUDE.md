# SIEVE Experiment Repo

Clean fork of the SIEVE post-retrieval evidence compilation pipeline for new experiments.

## What this is

SIEVE compresses BM25-retrieved conversation memories into structured evidence before passing to a reader LLM. The core finding: compression helps small readers (+13pp) but becomes a coin flip for strong readers (+0.8pp). This repo is for running new experiments with improved compilers and more models.

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

## Provenance

Forked from `memory-eligibility-feasibility` repo (2026-07-13). Original results preserved there.
