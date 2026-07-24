# SIEVE v2 — final paper spec (2026-07-14, post-kill-week)

Target: ARR (NAACL/ACL/EMNLP, Findings acceptable). Design goal is not ceiling
but FLOOR: every claim rests on data that already exists or on runs with no
scientific risk (re-measurement, not new bets). Nothing in the spine can die.

## What SIEVE v2 is now

Not a compressor and not a router of readers. **SIEVE v2 is a query-regime-
conditioned evidence compiler with a do-no-harm guard** — and the paper is the
complete answer to the question the prior paper left open:

> Fixed compression is reader-dependent and corrupts measurement (prior,
> EMNLP Findings 2025). So which axis of adaptivity actually pays?
> Answer: conditioning on the QUERY REGIME pays; conditioning on the READER
> does not (unlearnable); conditioning on ACQUISITION does not (evidence
> presence is not the binding constraint). We map the axes, ship the one that
> works, and pre-registered-kill the two that don't.

One sentence for reviewers: "we audited every axis of adaptive post-retrieval
compilation and report which one is real."

## The spine (5 sections, all evidence in hand or risk-free)

1. **Response surface / the stakes.** Representation choice is a 30-40pp,
   SIGN-FLIPPING effect across benchmark regimes under one pipeline
   (ConvoMem: structured 66-70 vs v1-structured 46-48; HotpotQA@400: summary
   71 vs 25-40 for every extract/filter/structure condition; MuSiQue: raw 19.6
   vs summary 49.0; aperture raw@0 vs raw@400 = +21-41pp). The cost of the
   wrong fixed choice dwarfs any routing gain — this is the do-no-harm
   motivation and the headline table. EXISTS (pilots + matrix; key cells
   re-validated under fixed eval).
2. **Axis 1, reader conditioning: NEGATIVE (pre-registered).** 21 readers x
   500 rows x styles: aggregate reader x style preferences stable (split-half
   r=0.845), existence pairs real (OLMo +18.2 vs MiMo -10.0 on summary) — yet
   no realizable reader-conditioned policy beats reader-blind (+0.0 to
   +0.25pp under reader-family holdout); per-row disagreement is
   noise-dominated; reader-blind captures 93%+ of headroom. Kills the natural
   follow-up to our own prior paper. EXISTS (matrix mining doc).
3. **Axis 2, acquisition: NEGATIVE (pre-registered, 3 gates).** Offline
   probe-exclusive reachability is real on BEAM (24.0% at B=30) and provably
   mere aperture on LME (decays to 0 by B=180; negative control). Conversion
   to reader accuracy: null at matched budget, and the gold-injection ceiling
   moves +0-1.2pp — evidence PRESENCE is not the constraint. NEEDS RE-RUN
   under fixed eval (~$10-15, 3 samples/row) — re-measurement risk only; if
   the ceiling opens instead, that is a better paper, not a dead one.
4. **Axis 3, query-regime conditioning: POSITIVE, modest, honest.** SIEVE v2
   policy = question-type -> {summary | structured | raw@budget | extractive}
   + stem-coverage guard + budget in the action space. Evaluated
   leave-one-benchmark-out against the strong no-learning baseline
   (always-summary, canonical prompt, 68.9 pooled): +2.5pp realized at
   question-type granularity, ~7-11pp row-level headroom reported as
   headroom, and — the real selling point — avoids every 30-40pp collapse in
   the Section-1 table. Frame as insurance + gain, report damage rates, not
   just means. MOSTLY EXISTS (matrix); needs the LORO confirm run and LoCoMo
   summary cells (known matrix gap).
5. **Robustness/measurement layer (woven in, NOT the thesis):** per-model
   best-prompt reporting, judge validity audit (gold-through-judge), 3-sample
   reliability on headline cells, presentation-sensitivity appendix (the four
   artifact sightings, incl. the span-prompt/eval fix), supersession appendix
   (KU chain rendering negative w/ prescriptive-norm-override mechanism; CR
   0->33->50 ladder reported as an evidence-side behavior observation with
   the format-circularity caveat stated by us before a reviewer says it).

## Why this clears the Findings bar with margin

- It is the direct, complete sequel reviewers of the prior paper asked for
  ("diagnostic, no method" -> here is the method AND the audit of why the
  fancier methods can't exist).
- Three pre-registered negative arcs with kill thresholds is the
  methodological standard reviewers reward and almost never see.
- Artifact release: 176K-row interaction matrix + new response-surface runs +
  replay/compile-once infra + frozen harnesses.
- No overclaim surface: the positive result is small and stated small; the
  large numbers (30-40pp) are about the cost of non-adaptivity, which is
  unimpeachable in our data.
- Adjacent work (DeferMem, TRACE, supersession cluster) is cited as parallel
  axes, none of which does the axis audit.

## Must-run list (all re-measurement, no new bets; ~$30-40, ~2-3 weeks)

1. Eval fix everywhere numbers enter tables: task-appropriate prompts,
   max_tokens 512, no forced abstention; judge audit per benchmark. ($0-5)
2. Conversion-experiment re-run under fixed eval, 3 samples/row. (~$10-15)
3. LoCoMo summary cells (matrix gap) + LoCoMo in the LORO loop. (~$5-10)
4. LORO confirm run of the question-type policy vs always-summary. (~$5)
5. Frontier probe on the 20 event_ordering rows (~$1) — decides one
   PARAGRAPH (capability-bound vs benchmark-limitation), not the paper.
6. 3-sample reliability on headline cells. (~$5)

## Explicitly OUT (and why, so it never gets re-litigated)

- z_r capability profiling / reader routing as method (unlearnable — Sec 2 IS
  the negative).
- Interrogator as method (Sec 3 IS the negative; reachability survives as
  analysis only).
- Chain rendering as headline method (KU: benchmark-artifact motivation,
  oracle-negative; CR: format-circular; appendix observation only).
- Benchmark-critique-as-thesis (that is the prior paper's genre; here it is
  rigor, not thesis).
- Any compressor leaderboard framing (prior paper's own result forbids it).

## Failure modes to pre-empt in writing

- "Why not just always summarize?" -> because Section 1 shows regimes where
  summary loses 20-30pp (ConvoMem structured beats it; aperture cases), and
  the policy's value is regime detection + guard; report per-regime.
- "+2.5pp is small" -> insurance framing + damage-rate table + headroom
  honestly split realized/unrealized.
- "Isn't this your last paper again?" -> last paper: fixed compression breaks
  measurement. This paper: what adaptivity is real. Zero overlapping tables.
- Write-time systems (Honcho 90.4 LME) -> different regime (ingestion cost,
  post-hoc applicability); one positioning paragraph + long-context
  full-history control where feasible.
