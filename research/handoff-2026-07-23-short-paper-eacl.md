# Handoff 2026-07-23: short paper → ARR Aug 3 → EACL 2027

## The decision

Ship the READER-CONDITIONING IMPOSSIBILITY result as an ARR **short paper**
(4pp) for the EACL 2027 cycle (deadline **Aug 3, 2026** AoE; reviewer signup
by Aug 5). Draft: `short-paper.md` (v3). The long paper (`paper.md`, the
four-axis audit + SIEVE v2 method) targets the **next** ARR cycle. The prior
paper (arXiv 2606.21807) continues its own revision track separately — the
impossibility result lives in the short paper ONLY (see CLAUDE.md
"Concurrent work" rules).

## What happened today

1. **Review panels run.** (a) 3-agent panel on the prior paper: 4 / 3.5 /
   AC 20%-main-45%-Findings-35%-reject; converged with the real ARR 2/3/3
   reviews on issues, inflated on scores. (b) 2-agent panel (Opus+Sonnet) on
   short-paper v1: 3.5 and 3; both said Findings floor. (c) Sonnet re-review
   of v2: Soundness 3→4, "ready for Findings as-is; main needs the deferred
   analyses in the body." (d) Real OpenReview reviews of the prior paper
   analyzed → v3 design rules: everything main-text, mechanism as
   predictions-not-narrative, artifact gets its own section, leanness budget
   (supp ≤4pp, two tables), never rely on rebuttal.
2. **short-paper.md v3**: hypothesis (noise-dominated routing signal) → P1/P2/
   P3 derived and confirmed; inversion result (57.6 vs 60.5 at maximal
   information) leads abstract+intro; §7 Data and Code Release claims only
   the analysis layer + extension cells, citing 2606.21807 as dataset source.
3. **P3 RUN (the held-out mechanism test).** Pre-registered r<0.5; held 8/9
   cells. Violation: trained abstractive compressor in-domain (HotpotQA,
   r=0.576) → two-regime argument (noise vs shared-signal; both exclude
   reader conditioning). Full numbers:
   `results/v2_runs/p3_cross_benchmark/RESULTS.md`. LME reference values
   reproduced (0.157/0.357), validating the pipeline.
4. **Provenance check**: router matrix v0.parquet = prior paper's published
   runs (validation gate matches) + post-freeze extensions (ConvoMem, LoCoMo,
   LME dense/oracle, mimo-v2.5, qwen3.6-27b, extra summary cells). Encoded in
   CLAUDE.md.
5. **EACL logistics**: mandatory reviewer signup (solo author = you; likely
   below the 2-papers assignment bar → register, probably no assignments);
   fees only if accepted (~$300 student virtual floor, EACL-2026 proxy);
   proceedings archival in ACL Anthology (main and Findings); an accepted
   paper = second qualifying pub for ARR reviewer eligibility.
6. **DONE (verified + committed)**: P3 frozen → `analysis/p3_cross_benchmark.py`
   (8 tests pass; real-parquet rerun matches RESULTS.md exactly); `release/`
   artifact folder (MIT, claims analysis layer + extension cells only);
   `paper/` LaTeX skeleton (ACL layout, Makefile, compiles to 4pp PDF, 0
   undefined refs). Title changed to **"Condition on the Query, Not the
   Reader"** (paper/main.tex + short-paper.md).

## Remaining before Aug 3 (all offline/free)

- [x] Verify agent output: P3 rerun matches RESULTS.md; tests pass; PDF builds.
- [ ] Baseline-vs-delta scatter exhibit (monotonicity shown, not asserted).
- [ ] Per-question-type decomposition of the +2.5pp + fold stability.
- [ ] Six-style pilot supplementary table.
- [ ] Realizable-transfer row for Table 2 (calibration-split donor votes).
- [ ] Re-implement policy-ladder / transfer / oracle-noise scripts in
      analysis/ (originals in defunct scratchpad; see release/TODO.md).
- [ ] References pass (research/novelty-scan-2026-07.md has the list).
- [ ] Optional: judge-flag-dropped sensitivity for P3 (MuSiQue 0.498 is
      borderline).
- [ ] ARR form: declare 2606.21807 as related anonymous concurrent work;
      fill Datasets/Software fields (the prior cycle's Datasets:1 lesson).

### From external review 2026-07-23 (scored 3/4/3.5 — converges with panel)

- [ ] **Earn the "bound": full-donor learned baseline.** Cross-fitted
      learner (regularized logistic + one nonlinear) per held-out reader over
      the 20-dim donor-outcome vector vs blind majority. If it loses too,
      "bounded" is earned; if it wins, we must know first. Offline, matrix.
- [ ] **Equivalence stats for ≤0.3pp**: 95% upper confidence bound on the
      paired reader-conditioning increment (hierarchical bootstrap over
      queries×readers). Claimed to move Soundness 3→3.5-4.
- [ ] **Variance decomposition** Δ_qr = μ+α_q+β_r+γ_qr+ε: estimate stable
      interaction γ. Caveat: single obs/cell confounds γ with ε — use
      pre-dedup duplicates (28,688) + judge flags as noise floor for a
      bounded estimate.
- [ ] **Make the two-regime account predictive**: shared-signal predicts
      damage-row overlap across readers is high in the violating cell
      (trained-in-domain) vs low-r cells. Direct overlap statistic, offline.
- [ ] **ANONYMITY (mandatory)**: rewrite "our own prior work"/self-correction
      framing as third-person citations for the review version.
- [ ] **P3 exact values + CIs** per cell (MuSiQue 0.498 rounds to the
      threshold; "8 of 9" is rounding-vulnerable as stated).
- [ ] **Title candidate**: "Condition on the Query, Not the Reader: A
      Negative Result for Reader-Aware Evidence Selection" (actionable-first,
      scope-safe).
- [ ] Merge Tables 1+2; compact P3 for the 4-page budget.

- [ ] Submit by Aug 3 AoE; reviewer signup by Aug 5.

## Review-cycle dates (EACL 2027)

Reviews/author response Sept 14–19 · meta-review Oct 8 · commitment Oct 11 ·
notification Nov 12 · camera-ready Nov 26 · conference Mar 9–14, 2027.
