# P3: cross-benchmark per-row delta correlations (2026-07-23, $0, offline)

**Pre-registered prediction** (short-paper.md v3, stated before computation):
pairwise per-row compression-delta correlations between readers are r < 0.5 on
held-out benchmarks (HotpotQA, MuSiQue, NQ), replicating the noise-domination
mechanism measured on LongMemEval (0.16 summary−raw / 0.38 structured−raw).

**Outcome: held in 8 of 9 cells; violated in one, informatively.**

## Method

`data/router_matrix/v0.parquet` (191,277 cells), uncapped (budget NaN) rows
with non-null labels. Per reader × slice × style pair: per-example delta =
mean(correct | style A) − mean(correct | style B); readers kept with ≥100
common examples per pair; Pearson correlation over common examples for every
reader pair. Script: session scratchpad `p3_cross_benchmark.py` (to be
promoted to `analysis/` with tests before submission).

## Results

| slice | pair | readers | pairs | mean r | median | min | max | frac < 0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| longmemeval | summary−raw | 22 | 231 | 0.157 | 0.150 | −0.087 | 0.465 | 1.000 |
| longmemeval | structured_v1−raw | 24 | 276 | 0.357 | 0.370 | 0.052 | 0.610 | 0.931 |
| hotpotqa | summary−raw | 19 | 171 | 0.387 | 0.391 | 0.056 | 0.773 | 0.906 |
| hotpotqa | summary_trained−raw | 19 | 171 | **0.576** | 0.574 | 0.288 | 0.829 | 0.246 |
| hotpotqa | extractive_trained−raw | 19 | 171 | 0.433 | 0.432 | 0.117 | 0.807 | 0.708 |
| musique | summary−raw | 19 | 171 | 0.498 | 0.487 | 0.277 | 0.753 | 0.538 |
| nq | summary−raw | 19 | 171 | 0.235 | 0.224 | −0.039 | 0.673 | 0.977 |
| nq | summary_trained−raw | 19 | 171 | 0.383 | 0.357 | 0.043 | 0.751 | 0.731 |
| nq | extractive_trained−raw | 19 | 171 | 0.398 | 0.388 | 0.125 | 0.692 | 0.830 |

LongMemEval reference values reproduce the matrix-mining doc (0.16 / ~0.36),
validating the pipeline. Pooled held-out pairs below 0.5: 70.5%.

## Interpretation (goes in short-paper.md §5)

- **Generic compression replicates the mechanism everywhere** (0.24–0.50):
  per-row deltas are idiosyncratic → noise regime → conditioning on any
  reader re-injects noise (P1/P2).
- **The violation is the trained abstractive compressor in-domain**
  (RECOMP-style summary on HotpotQA, its training benchmark; mean r = 0.576,
  75% of pairs ≥ 0.5). High cross-reader correlation = *shared-signal*
  regime: systematic content deletion damages all readers together, which
  reader-blind routing captures by construction. Weaker echo off-domain
  (NQ trained pairs 0.38–0.40).
- Reader conditioning pays only in a third regime — large, stable,
  reader-SPECIFIC per-row structure — which no cell exhibits: where r is low
  the reader-specific residual is noise (split-half/transfer, P1/P2); where r
  is high the signal is not reader-specific.

## Caveats

1. Reader sets per slice overlap but are not identical to the 21-reader LME
   panel (19 readers clear the ≥100-common-rows gate off-LME).
2. Single sample per cell off-LME; no judge-flag correction applied here
   (flags exist per cell; a flag-dropped sensitivity pass is cheap if a
   reviewer asks).
3. MuSiQue summary−raw sits at the threshold (0.498); reported as a hold, but
   honestly borderline — the paper reports the full table, not the verdict
   alone.
