# SIEVE

**Structured Information Extraction and Verification for Evidence** -- a post-retrieval evidence compilation pipeline for conversational memory QA.

## What it does

SIEVE sits between a retriever (BM25) and a reader LLM. Instead of dumping raw retrieved conversation turns into the reader's context, SIEVE compiles them into structured evidence: it identifies the query's intent and schema, extracts relevant facts, resolves temporal references, and packages a compact evidence summary.

**Key finding:** Compression helps small readers significantly (+13pp) but becomes a coin flip for strong readers (+0.8pp). The structured compilation matters most when the reader can't do the filtering itself.

## Architecture

```
Query --> Retrieval --> SIEVE Compiler --> Reader LLM --> Answer
              |              |
          retrieval/     compiler/
                         evidence/
```

- **`compiler/`** -- Core compilation pipeline: schema selection, slot filling, greedy evidence selection with rescue passes, deterministic answer extraction
- **`evidence/`** -- Evidence data structures: units, roles, rendering, pack building
- **`retrieval/`** -- BM25 retrieval, memory object construction, query target extraction
- **`reader/`** -- LLM client, prompt construction, answer scoring
- **`controller/`** -- Orchestration logic, signal routing, text utilities
- **`shared/`** -- Common NLP utilities, constants, caching
- **`cli/`** -- Entry points for running experiments
- **`analysis/`** -- One-off analysis scripts (bootstrap CI, correlation stats)

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (retrieve + compile + read)
python -m cli.run_pipeline \
  --slice dataset-slices/longmemeval_bm25_top20_v1.jsonl \
  --controller-config configs/controller_v0.json \
  --provider openrouter \
  --model meta-llama/llama-3.1-8b-instruct \
  --systems bm25_sieve \
  --parallelism 32

# Replay cached compilation with a new reader (skip recompilation)
python -m cli.replay replay \
  --cache data/compilation_cache/paper_sieve.json \
  --model <model-id> \
  --output-dir results/answer_generation_runs/<run_name>

# Score with LLM judge
python -m cli.judge \
  --run-dir results/answer_generation_runs/<run_dir> \
  --provider openrouter \
  --model deepseek/deepseek-chat-v3-0324
```

## Data

- **`dataset-slices/`** -- LongMemEval and HotpotQA benchmark slices (JSONL)
- **`data/compilation_cache/`** -- Pre-compiled evidence from the paper (enables replaying with new readers at zero compilation cost)
- **`results/`** -- Experiment outputs

Large files are tracked with Git LFS.

## Tests

```bash
python -m pytest tests/ -v
```

## Citation

Paper under review. Citation details will be added upon publication.
