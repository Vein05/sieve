# SIEVE v2: Capability-Conditioned Evidence Compilation

## Thesis

Different readers require different evidence representations. SIEVE should adapt both evidence content and representation to reader capability. The optimal structure and aperture of retrieved evidence depend on the downstream decoder, not just query complexity or a fixed token budget.

## One-sentence pitch

We show that the optimal structure and aperture of retrieved evidence depend on reader capability, and introduce a capability-conditioned compiler that preserves weak-reader gains without bottlenecking stronger readers.

## Why this is a contribution

The EMNLP Findings paper established the phenomenon: fixed compilation helps weak readers (+13pp) but is neutral or harmful for strong ones (+0.8pp). Every existing compression method (RECOMP, ECoRAG, SEER, LLMLingua, ACC-RAG) optimizes a single compression policy. Nobody conditions on the reader. Nobody evaluates on conversational memory at 1M+ token scale.

## Reader-adaptive evidence representation

The core idea: the compiler produces different evidence depending on who reads it.

| Reader capability | Evidence style | What it gets |
|---|---|---|
| Weak (8B) | Full structured | Normalized facts, explicit relations, computed answers, distractors stripped |
| Mid (70B) | Extractive + light structure | Selected quotations plus section headers, obvious distractors removed |
| Strong (frontier) | Filtered raw | Minimally transformed evidence, original wording preserved, competing hypotheses retained, contextual and pragmatic cues kept |

## Architecture

```
pi_theta(style, budget, evidence | q, C, z_r)
```

Where:
- q: query
- C: retrieved candidates
- z_r: reader capability profile (NOT a model ID)
- style: structured / extractive / raw / hybrid
- budget: adaptive evidence aperture
- evidence: selected evidence set

### Capability profile z_r

Derived from a small calibration set (~50 queries), not hard-coded model names. Dimensions:
- Distractor tolerance
- Multi-hop integration ability
- Temporal reasoning ability (implicit vs explicit)
- Robustness to paraphrased vs quoted evidence
- Context utilization depth
- Abstention calibration

This enables zero-shot transfer: train on 3-5 readers, generalize to unseen readers via calibration.

### System design

```
candidate scorer (cross-encoder or similar)
  -> coverage/completeness estimator
  -> representation policy (conditioned on z_r)
  -> one of:
       structured compiler (v1 SIEVE, retained as one expert)
       extractive packager
       filtered-raw packager
       hybrid structured + quotes
```

Important: v1's structured compiler is retained as one representation option inside v2, not discarded. The contribution is learning WHEN to invoke it, not assuming maximal structure is always correct.

## What NOT to do

- Do not condition on model ID categorically (memorizes training readers, fails on new ones)
- Do not train primarily on SIEVE-v1's selected/rejected units (encodes heuristic policy, not reader utility)
- Do not promise +5-8pp for strong readers (likely too optimistic)
- Do not claim SIEVE "compresses 1M tokens" (it compresses top-k retrieved passages from 1M-token histories)
- Do not use GPU pipeline / dense retrieval as headline contribution (infrastructure, not science)

## Training signal

Use downstream reader utility, not SIEVE-v1 labels. The existing compilation cache is a seed, not ground truth.

Counterfactual evidence lattice: for each query, produce multiple evidence variants, run every reader against every variant, map the utility surface:

```
(q, reader_capability, evidence_package) -> accuracy, damage, cost
```

## The first experiment: reader x representation response surface

Before building a learned compiler, establish the phenomenon empirically.

### Evidence variants to produce (per query)

1. Raw top-20 retrieved turns
2. Filtered raw top-k (distractors removed, original wording kept)
3. Extractive sentences (top sentences by relevance)
4. V1-selected passages rendered raw (matched-content control)
5. V1-selected passages rendered extractively (matched-content control)
6. Current effective structured SIEVE policy
7. Structured SIEVE + exact source quotations
8. Generic LLM summary (abstractive)

Also retain exact uncapped raw and the exact published SIEVE-v1 system (cached
prompts plus deterministic answer routes) outside the factorial grid. Router
evaluation holds out both queries and reader families.

Each at ~200, 400, and 800 reader tokens.

### Readers to test

- 2 weak: e.g. Llama-3.1-8B, Phi-3.5-mini
- 2 mid: e.g. Llama-3.1-70B, Qwen-2.5-32B
- 2 strong: e.g. GPT-4o, Claude Sonnet

### What to measure

- Best fixed representation per reader
- Best representation per query type
- Oracle best representation per (query, reader) pair
- How much of oracle gain a simple capability-conditioned router recovers
- Rescue and damage transitions relative to raw
- Transfer to held-out readers (train router on 4, test on 2)

### What this tells us

- If oracle adaptive >> every fixed policy: the thesis is real, proceed to learned compiler
- If a simple router recovers most of oracle: the method works, train the full system
- If all representations perform within noise for strong readers: the effect is too small, rethink

## Ablation sequence for the paper

1. Fixed SIEVE-v1 (baseline)
2. Query-adaptive style (style varies by query type, fixed reader)
3. Reader-adaptive style (style varies by reader capability)
4. Reader-adaptive style + budget (both what and how much)
5. Full reader-adaptive set selection (learned scorer)
6. Ablate components: remove exact quotations, provenance, completeness checks

## Realistic success criteria

- Retain +10-13pp for weak readers
- Eliminate or sharply reduce compression damage for strong readers (statistically tied at 5x fewer tokens is already excellent)
- Dominate fixed compression on the accuracy-cost Pareto frontier
- Recover most of raw-to-oracle headroom where retrieval contains the answer
- Generalize to unseen readers with small calibration set

## Repo structure

```
sieve/
  baseline/        # frozen v1 heuristic pipeline (current code)
  learned/          # v2 learned compiler (new, clean, ~2-3K lines)
    scorer.py       # cross-encoder evidence scorer
    calibration.py  # reader capability profiling
    policy.py       # representation policy conditioned on z_r
    packagers/      # structured, extractive, raw, hybrid formatters
  experiments/      # response surface experiment scripts
  analysis/         # result analysis and visualization
```

## Key risks

1. The reader-dependent effect may be too small to justify a learned system
2. Capability profiles may not transfer to truly novel architectures
3. The response surface experiment is expensive (6 representations x 3 budgets x 6 readers x 500 queries = 54K reader calls)
4. BEAM's low SIEVE performance (30% vs 54% raw) may reflect retrieval/completeness failures, not compilation failures

The response surface experiment answers risk 1 before any training investment.
