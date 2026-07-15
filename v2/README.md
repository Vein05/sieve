# SIEVE v2: reader-adaptive evidence compilation (response-surface phase)

## The experiment in one paragraph

Different readers need different evidence representations. Before building a
learned compiler we establish the phenomenon empirically: for every query we
produce multiple evidence variants over a controlled style/budget grid, run
every reader against every variant, and map the
`(reader, style, budget) -> accuracy/cost` response surface. If oracle-adaptive
representation beats every fixed policy (and a simple reader-conditioned router
recovers most of that gain) the reader-adaptive thesis holds and we proceed to
the learned compiler. See `research/idea-v2.md` and `research/v2-iclr-assessment.md`.

## Module map

| Module | Purpose |
| --- | --- |
| `types.py` | Frozen value types: `EvidenceVariant`, `VariantKey`, `ReaderResult`. |
| `budget.py` | `count_tokens` (tiktoken, never `.split()`) and greedy whole-unit `truncate_units_to_budget`. |
| `packagers/` | Evidence policies and matched-content controls. |
| `scorer.py` | `StemOverlapScorer` (Phase 1) and `CrossEncoderScorer` (Phase-2 stub). |
| `variant_cache.py` | Build / save / load the variant cache with fingerprint integrity. |
| `policy.py` | Representation policies: `FixedPolicy`, `HeuristicTierPolicy`, `LearnedPolicy` (stub). |
| `calibration/` | Capability profile `z_r`: `ProbeSpec`, `ReaderProfile` (Phase-2 stubs). |
| `runner.py` | The response-surface runner (compile-once / replay-many). |
| `configs/response_surface_v0.json` | All experiment parameters (styles, budgets, readers, seed, encoding). |
| `../cli/run_v2.py` | Thin argparse CLI over `runner.py`. |

### Evidence policies and controls (`research/idea-v2.md`)

1. `raw` — top-20 candidates in retrieval order; budget `0` is uncapped.
2. `filtered_raw` — distractors dropped, survivors kept **verbatim**.
3. `extractive` — top sentences, **verbatim**, emitted in chronological order.
4. `selected_raw` — v1-selected content rendered verbatim.
5. `selected_extractive` — the same selected content rendered sentence-wise.
6. `structured` — v1's effective structured policy, including routed fallbacks.
7. `structured_quotes` — structured evidence plus exact selected-source lines.
8. `summary` — generic abstractive summary via an injected compile-LLM.
9. `sieve_v1` — exact published system control, outside the factorial grid. It
   replays cached prompts and preserves the 40 deterministic no-reader routes.

## Running the smoke test

```bash
python -m pytest tests/v2/
```

All v2 tests are fully offline: no network, no real LLM. The summary packager
and the runner take an *injected* callable that is a deterministic fake in tests.

## Two-command workflow

```bash
# 1. Compile the variant cache once (styles x budgets x slice).
python -m cli.run_v2 compile \
    --slice dataset-slices/longmemeval_bm25_top20_v1.jsonl \
    --styles raw,filtered_raw,extractive,selected_raw,selected_extractive,structured,structured_quotes,sieve_v1 \
    --budgets 200,400,800 \
    --config v2/configs/response_surface_v0.json \
    --sieve-cache data/compilation_cache/paper_sieve.json \
    --naive-cache data/compilation_cache/paper_naive.json \
    --out data/v2_variant_cache/response_surface_v0.json

# The 'summary' style additionally needs a compile-LLM:
#   --styles ...,summary --compile-model qwen/qwen3-8b --provider openrouter

# 2. Replay a reader panel (cost-guarded; dry-run unless --yes).
python -m cli.run_v2 replay \
    --cache data/v2_variant_cache/response_surface_v0.json \
    --slice dataset-slices/longmemeval_bm25_top20_v1.jsonl \
    --config v2/configs/response_surface_v0.json \
    --readers meta-llama/llama-3.1-8b-instruct,meta-llama/llama-3.1-70b-instruct \
    --run-dir results/v2_runs/response_surface_v0 \
    --yes

# 3. Judge unchanged (the runner's run.json matches cli/judge.py's layout).
python -m cli.judge --run-dir results/v2_runs/response_surface_v0
```

## The shared-prompt-template control (critical)

Every factorial response-surface style is wrapped in one reader prompt template
(`runner.build_reader_prompt`). Only the evidence *representation* varies between
conditions; the prompt scaffolding (system line, rules, question placement) is
held constant. If scaffolding varied by style, measured accuracy differences
could come from the prompt rather than the representation, invalidating the
response surface. `sieve_v1` is the explicit exception: it replays the exact
published system—cached prompt or deterministic answer route—as a fixed-system
accuracy and cost control.

Router evaluation uses deterministic held-out query splits recorded in every
output. Final transfer claims must also hold out reader families; evaluating a
new reader on router-training queries is not sufficient.

## The cost guard

`replay` is a dry-run by default: it prints the exact `n_variants x n_readers`
call count and the estimated LLM calls, and requires `--yes` to execute. Each
`(variant, reader)` result is written to its own shard as it completes, so runs
are **resumable** (a crashed run loses at most the in-flight pairs) and re-running
only regenerates missing shards.

## What is a Phase-2 stub (owner to author)

* **Calibration probe battery** (`calibration/probes.py`) — `load_probe_battery()`
  returns an empty battery; the ~50-100 probe items across the six dimensions are
  authored by the owner (do not invent probe content).
* **Capability profile harness** (`calibration/profile.py`) — `compute_profile`
  aggregates per-dimension scores; the probe-running + per-probe scoring harness
  is the owner's to implement.
* **Learned policy** (`policy.py::LearnedPolicy`) — a GBM/MLP over
  `(query features, z_r)` trained on the response-surface utility matrix.
* **Cross-encoder scorer** (`scorer.py::CrossEncoderScorer`) — a real
  cross-encoder (`deberta-v3-small` / MiniLM) to replace stem overlap.

## Deferred experiment inputs

* Pin at least two dated **frontier reader IDs** in a run-specific config. The
  base config intentionally leaves `reader_panel.strong` empty instead of
  containing callable placeholder model IDs.
* Author the **probe battery** content.
* Train the **cross-encoder** and the **learned policy** once the response
  surface passes the go/no-go gate (oracle-adaptive >> best-fixed).
