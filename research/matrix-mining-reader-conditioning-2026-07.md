# Matrix mining: is reader conditioning learnable? (2026-07-14)

Offline analysis of `data/router_matrix/v0.parquet` (191,277 deduped rows).
Zero API calls. Scripts: session scratchpad (`oracle_gap.py`, `learnability.py`,
`loro_transfer.py`); rerunnable in ~2 minutes against the parquet.

## Setup

Panel: 21 readers with complete uncapped LongMemEval triples
{`naive_top_k@0` (raw), `bm25_sieve@0` (structured_v1), `llm_summarize@0`
(summary)} x 500 rows. Excluded: llama-3.1-8b (missing structured judge run),
grok-3-mini, gemma-3-4b, gpt-4o-mini, llama-3.2-3b (incomplete cells).
Action space = 3 styles; this is the v1-era surface, not the v2 taxonomy.

## Step 0: the oracle gap survives noise correction, but shrinks

| quantity | value |
| --- | ---: |
| always raw / structured / summary | 46.7 / 52.8 / 50.0 |
| best fixed per reader | 53.6 |
| reader-blind per-row oracle | 62.4 |
| per-(row, reader) oracle | 65.8 |
| gap | **+3.4pp** (query-bootstrap 95% CI [2.9, 4.0]) |

4.3% of cells carry `label_disagreement` judge flags. Dropping flagged pairs:
gap = **+2.6pp**. Injecting random flips into flagged cells *inflates* the gap
to 4.5pp — hindsight per-(row, reader) oracles harvest label noise
preferentially. Read the honest reader-specific headroom as ~2.6-3.4pp, not the
published 30%-of-headroom framing.

Mixed-label rows (structured vs raw): 28.0% on this panel (published: 37% on
the 20-reader compress/no-compress framing).

## Aggregate reader preferences are real and stable

- Split-half reliability of per-reader action deltas (vs raw): **r = 0.845**
  [0.737, 0.911].
- Existence pairs (equal baseline, opposite effect) confirmed:
  Phi-4 raw 47.2 / summary 43.6 vs Gemma-3-12B raw 47.4 / summary **54.6**.
  OLMo-32B summary **+18.2pp** vs MiMo-v2.5 summary **-10.0pp**.
- The pattern is monotone in capability: summary/structure help weak readers,
  converge to zero effect for the strongest (Seed, MiMo, Grok-4.1-fast,
  Qwen3-8B all have structured within ~0-3pp of raw). At the strong end all
  actions tie, so there is nothing for a policy to condition on.

## But realizable reader conditioning captures almost none of it

Policy ladder, 10-fold CV over queries (honest held-out-query evaluation):

| policy | accuracy |
| --- | ---: |
| P0 always structured | 52.8 |
| P1 per-reader best fixed | 53.0 (+0.2) |
| P2 per-question-type, reader-blind | **55.3** (+2.5) |
| P3 per-(reader, question-type) | 55.3 (**+0.0 over P2**) |
| reader-blind row oracle | 62.4 |
| per-(row, reader) oracle | 65.8 |

Calibration sweep (K labeled rows per reader): P3 at K=200 reaches 54.8,
still at/below P2's CV value. Reader conditioning at question-type granularity
adds nothing a query-only policy does not already capture.

Row-level transfer test (leave-family-out, 100-query calibration, 100 reps):

| policy for held-out reader | accuracy |
| --- | ---: |
| always structured | 52.8 |
| unweighted row-majority vote of other-family readers | 60.5 |
| calibration-similarity-weighted vote | 60.8 (**+0.25pp**, 17/21 readers) |
| single most-similar donor (chosen on test = upper bound) | 57.6 |

The best single donor is *worse* than the unweighted majority: per-row
disagreement between any two readers is noise-dominated (pairwise correlation
of per-row structured-minus-raw deltas: mean 0.38; summary-minus-raw: 0.16).
Averaging across readers denoises; conditioning on reader identity mostly
re-injects noise.

Budget axis: only 3 readers have the full capped grid; within-structured budget
effects are ~2-3pp, best-budget split-half agreement 0.50 (chance ~0.33), and
the 7-condition specific-minus-blind gap is 1.2pp. Thin panel, but no sign of a
stable reader x budget interaction either.

## Verdict

On the v1-era style surface, the reader-specific oracle gap is real (~2.6-3.4pp
after noise correction) but **essentially unlearnable**: every realizable
reader-conditioned policy tested captures <=0.3pp over its reader-blind
counterpart, under both query holdout and reader-family holdout. The robust
reader x style interaction that does exist is monotone in baseline capability —
a scalar already captures it, and even that scalar buys only +0.2pp (P1 vs P0)
because structured is best-or-tied for nearly every reader.

This fires the kill criterion for bet 2 (z_r capability profile) **on this
action space**, consistent with the three v2 pilots (reader conditioning added
2/273 pairs). What remains alive and learnable is **query-side**: +2.5pp
realizable at question-type granularity, with reader-blind row-oracle headroom
of 62.4 vs 55.3 suggesting ~7pp more available to richer query/evidence
features — plus the v2 pilot's +13.2pp per-query representation oracle on the
new 6-style surface.

## Caveats

1. Three correlated v1-era styles; the v2 taxonomy (filtered-raw, extractive,
   quotes, budgets) could expose disagreement these styles cannot. But all
   three pilots already chose structured@400 unanimously on the new surface.
2. No true frontier readers in the panel (strongest raw ~57%). The strong-end
   flatness argument suggests frontier readers make conditioning less useful,
   not more, but this is extrapolation.
3. The MiMo shared-prompt collapse (57.1 -> 35.2 on identical questions) shows
   prompt/presentation x reader effects of ~20pp — an axis absent from both
   the matrix styles and the v2 style taxonomy. If reader conditioning lives
   anywhere, it may be at the presentation/format level (cf. LongMemEval CP4:
   10pp JSON-vs-NL swings at oracle retrieval), not the compression-style level.

## Decision implication

Do not build the capability profiler or spend on a larger reader response
surface for style routing. Either (a) reposition the paper around
query-adaptive representation with the reader-conditioning negative as a
finding that corrects the prior paper's "30% inaccessible headroom"
implication, or (b) test the one surviving reader-conditioning hypothesis —
presentation/format-level adaptation — before abandoning the capability-
conditioned framing entirely.
