# Heuristic ladder, re-scored offline on the judged pilot (2026-07-14)

> **RETRACTED same day — the ladder does not transfer.** See "Transfer test
> on the router matrix" below. The pool-fit ladder beat fixed policies on
> the two-benchmark pilot ONLY because the pilot's summary cells used the
> broken one-line prompt. Against the canonical summarizer on the full
> matrix suite, no pool-token threshold beats always-summary on any slice,
> even tuned in-sample. Kept as a record of the overfit catch.

Step 2 of the learned-components sequencing
(`research/learned-components-design-2026-07.md`): the no-learning ladder
baseline that L2 must beat, evaluated at $0 against the already-judged
generic cross-bench pilot (100 ConvoMem + 100 HotpotQA queries x 7 conditions
x 3 readers; deepseek-chat-v3 judge, normalized 0-1 scores x100).

## Fixed-policy reference (pooled = mean of the two benchmarks)

| policy | convomem | hotpotqa | pooled | mean evidence tokens |
|---|---:|---:|---:|---:|
| raw@0 (uncapped) | 75.0 | 63.3 | **69.2** | 715 |
| summary@400 | 61.5 | 71.3 | 66.4 | 44 |
| extractive@400 | 72.2 | 56.2 | 64.2 | 400 |
| filtered_raw@400 | 73.7 | 35.5 | 54.6 | ~350 |
| guarded_structured@400 | 68.8 | 36.8 | 52.8 | ~350 |
| structured_generic@400 | 67.5 | 29.2 | 48.3 | ~210 |
| domain-aware fixed (cheat) | 75.0 | 71.3 | 73.2 | — |
| reader-blind per-query oracle (7 conds) | 80.5 | 83.2 | 81.9 | — |

Note: on this two-benchmark pool the best fixed policy is **uncapped raw**,
not summary (the matrix's 63.2% always-summary result pools six slices
including LME/MuSiQue where raw is much weaker).

## Ladder results (compile-time signals only, no domain identity)

The load-bearing signal is **pool fit**: does the retrieved pool fit the
budget with margin? `pool_tokens > 1.5 x budget` separates the domains almost
perfectly (ConvoMem 20/100, HotpotQA 93/100) and is exactly the mechanism
H-ii exposed (truncation destroys multi-doc evidence).

| ladder | rule | convomem | hotpotqa | pooled | tokens |
|---|---|---:|---:|---:|---:|
| **ladder3** | pool>600 -> summary, else raw@0 | 72.7 | 69.0 | **70.8** | 244 |
| ladder5 | + filtered_raw when filter survival >=0.5 | 72.2 | 67.3 | 69.8 | 196 |
| ladder4 | + structured when guard passes | 66.7 | 69.2 | 67.9 | 196 |
| (earlier) guard-first ladders, no raw branch | — | 63-69 | 62-71 | 65-67 | — |

Paired query bootstrap (2000 resamples, both benchmarks):

- ladder3 − always-raw@0: **+1.7pp, 95% CI [−2.1, +5.6]** — not significant
  on accuracy alone at n=200, but achieved at **1/3 the evidence tokens**
  (244 vs 715). The claim is Pareto, not accuracy.
- ladder3 − always-summary: **+4.4pp, 95% CI [+0.8, +8.1]** — significant.

## Findings

1. **A one-signal, no-learning ladder beats every fixed policy** (70.8 vs
   69.2 best-fixed) at a third of raw's token cost, using zero benchmark
   identity. "If the pool fits the budget, do not compile; if it does not,
   abstract" is the do-no-harm principle in its minimal form.
2. **Ladders that lack a raw-passthrough branch lose to best-fixed** (65-67
   pooled). The earlier guard->filtered_raw design was wrong on this pool:
   the safe action on conversational memory is raw, not filtered/structured.
3. **The structured branch hurts here** (ladder4 -2.9 vs ladder3): structure
   never wins on ConvoMem/HotpotQA. Its value lives on LME
   (temporal/knowledge-update questions, +7-10pp for strong readers in the
   matrix). The ladder's structured branch must be validated on an LME slice
   at matched budget before it stays in the method.
4. **Residual to oracle is the L2 target**: 70.8 -> 81.9 = +11.1pp per-query
   headroom over the ladder on these two benchmarks. This is what utility
   regression chases; if it captures less than ~+2pp under LOBO, the ladder
   ships as the method (pre-registered in the design doc).
5. **Both summary cells here use the OLD one-line prompt.** The packager now
   uses the canonical LLM-Summarize prompt (fixed 2026-07-14,
   `v2/packagers/summary.py`); canonical summaries score 71-81 on ConvoMem vs
   61.5 here, so ladder3's summary branch (and pooled 70.8) is a lower bound.

## Decomposing the +11.1pp residual (added same day)

Where does ladder->oracle headroom live, and how much is real?

| | convomem | hotpotqa |
|---|---:|---:|
| naive per-query oracle gap over ladder3 | +7.8pp (17/100 queries) | +14.2pp (28/100 queries) |
| leave-one-reader-out oracle gap (pick on 2 readers, score on 3rd) | **+3.8pp** | **+2.3pp** |

- **~3/4 of the naive headroom is selection noise** (max over 7 conditions x
  3 near-binary readers), the same inflation the matrix mining found. The
  honest L2 target on this pool is ~+3pp pooled, not +11pp — consistent with
  the matrix's +2.5pp realizable at question-type granularity.
- Where the oracle genuinely wins it most often picks **extractive** (9/17
  ConvoMem, 16/28 HotpotQA wins) — the style that never wins as a fixed
  policy.
- **Win queries are not separable by any recorded signal**: pool tokens,
  guard coverage, filter survival, and summary length have near-identical
  means for win vs rest queries. The current feature set has no handle on
  the residual; whatever is learnable there needs content-level features
  (L1/L3 scores), not pack statistics.
- LORO with 3 readers also *under*-estimates (the 2-reader selector is
  itself noisy); truth is between +3 and +11, likely nearer the bottom.

Implication for sequencing: do not fund a bigger ConvoMem/HotpotQA response
surface expecting L2 to mine this pool — the value case for L2 rests on LME
(where structure pays +7-10pp for strong readers), BEAM, and LoCoMo-Plus,
with ConvoMem/HotpotQA as do-no-harm controls.

## Caveats

- Threshold 1.5x was selected while looking at this data; the rule is
  a-priori natural ("fits with margin") but the number needs held-out
  validation (LoCoMo, BEAM, LME slices) before any claim.
- Two benchmarks, 100 queries each, 3 cheap readers; no LME cells at these
  budgets, so the pooled numbers do not represent the full suite.
- Ladder escalation past summary (to raw@0 on predicted summary
  insufficiency) is untested — no summary-insufficiency signal is recorded
  in the current caches (candidate: "No relevant evidence found" marker +
  tiny raw_summary_tokens).

Script: scratchpad `ladder_rescore.py` (session); reproduce from
`results/v2_runs/generic_cross_bench_*/judge_run_v2.json` +
`data/v2_variant_cache/generic_cross_bench_*.json`.

## Transfer test on the router matrix (same day) — the ladder FAILS

Frozen-rule test of ladder3 on the four slices it never saw, using
`data/router_matrix/v0.parquet` uncapped cells (canonical summarizer,
19-21 readers on LME/QA slices; ConvoMem has the same llama-8b/qwen-7b/72b
readers as the pilot):

| slice | always-raw | always-summary | best fixed | ladder3 (frozen) | ladder (threshold tuned IN-SAMPLE) |
|---|---:|---:|---:|---:|---:|
| longmemeval | 46.7 | 50.0 | structured 52.8 | 46.7 | 55.0 = always-summary* |
| convomem | 75.6 | **80.0** | summary | 75.6 | 78.7 = always-summary* |
| hotpotqa | 62.6 | **85.7** | summary | 76.3 | 86.1 = always-summary* |
| musique | 19.6 | **49.0** | summary | 26.1 | 48.8 = always-summary* |
| nq | 73.8 | 74.4 | summary | 74.6 | 76.1 |

*the in-sample-optimal threshold routes essentially every query to summary.
Pooled over 5 slices: tuned ladder 68.9 = always-summary 68.9. **The
pool-fit signal has zero routing value against a properly-prompted
summarizer.**

Post-mortem (three compounding errors, caught by this test):

1. **Broken comparator.** The pilot's summary cells used the one-line
   prompt; the canonical prompt scores 71-81 on ConvoMem with the SAME
   readers (matrix per-reader cells). Every ladder "win" was against a
   crippled summary.
2. **Two-benchmark rule fitting.** Pool-fit separated ConvoMem from
   HotpotQA, so it looked causal; on MuSiQue raw scores 19.6 regardless of
   pool size — "pool fits" never implies "raw is safe" on distractor-heavy
   QA. Raw passthrough is NOT a universal do-no-harm action.
3. **Token-accounting drift**: `selected_memory_tokens` units are
   inconsistent across matrix slices (LME p50=69, ConvoMem p50=263 vs pilot
   ~534 for the same slice) — cross-slice thresholds on this column are
   meaningless anyway. Re-derive pool sizes from slice files for any future
   feature.

## Honest "best possible" (LORO-corrected per-query oracle vs always-summary)

| slice | always-summary | LORO oracle | realizable headroom | readers |
|---|---:|---:|---:|---:|
| longmemeval | 50.0 (structured 52.8) | 60.8 (3 styles) | **+8-11pp** | 21 (reliable) |
| convomem | 80.0 | 87.8 (3 styles) | **+7.8pp** | 4 (directional) |
| hotpotqa | 85.7 | 85.9 | +0.2pp | 19 (reliable) |
| musique | 49.0 | 51.8 | +2.8pp | 19 (reliable) |
| nq | 74.4 | 76.5 | +2.1pp | 19 (reliable) |
| locomo | no summary cells | — | untested (gap) | — |

The consistent picture across pilot, matrix mining, and this test: **the
only realizable routing wins are in conversational memory (LME +8-11,
ConvoMem ~+8); QA is match-only (0-3pp).** The correct no-learning baseline
is **always-summary (canonical prompt)** — 68.9 pooled — and the ladder
concept survives only inverted: summary is the base action, escalation to
structured/raw must be earned by a sufficiency/utility signal, and L2's
value case lives entirely on the conversational-memory slices plus the
engineered ops (supersession, standing digest) that summary cannot express.
