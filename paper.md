# Which Adaptivity Pays? An Audit of Post-Retrieval Evidence Compilation, and a Compiler That Transfers

*Working draft (paper.md, 2026-07-23). Not in submission format. All numbers below
come from runs/analyses that already exist in this repo or the router matrix;
cells marked [PENDING] are on the must-run list (~$30-40 total, re-measurement
risk only). Zero tables are shared with arXiv 2606.21807.*

**Target:** ARR (EACL 2027 cycle is Aug 3; realistically the following cycle) — long paper.

---

## The one-sentence claim

Representation choice in post-retrieval evidence compilation is a 30–40pp,
*sign-flipping* effect across benchmark regimes — yet of the four axes of
adaptivity the field proposes to exploit it (reader, acquisition, format,
query-regime), we show with pre-registered kill criteria that only
**query-regime conditioning** is real, and we ship a compiler that captures it
while never collapsing where every fixed policy (including two published
compressors and our own prior system) does.

## Elevator pitch for reviewers

Prior work (ours included) showed fixed compression is reader-dependent and
corrupts measurement. The obvious follow-ups are "adapt the compression to the
reader," "retrieve more," "fix the format," "mark the conflicts." Nobody has
tested which of these axes actually pays. We audited all of them under one
controlled pipeline with cached, replayable compilation. Three die, with kill
thresholds stated in advance. One survives — and the honest size of the win is
+2.5pp realized plus insurance against 30–40pp collapses, which we argue is
the correct, deflated shape of this entire research area.

---

## Abstract (draft)

Retrieval-augmented pipelines must choose how retrieved evidence reaches the
reader: raw, filtered, extracted, summarized, or structured. We show this
choice is not a detail: under a single fixed pipeline, the best and worst
representation differ by 30–40 points across benchmark regimes, *with opposite
signs* — the representation that wins on conversational memory
(structure) is near-worst on multi-document QA at matched budget, and vice
versa (summary). Fixed compilation policies therefore fail to transfer: our own
prior hand-built compiler loses 25–35 points when moved to an adjacent
conversational-memory benchmark, mirroring how a published trained compressor
(RECOMP) collapses when moved onto conversational memory. We then audit, with
pre-registered kill criteria, the four axes of adaptivity proposed to fix this:
conditioning on the reader (unlearnable: every realizable reader-conditioned
policy captures ≤0.3pp of a 2.6–3.4pp oracle gap, because per-row disagreement
between readers is noise-dominated), acquiring more evidence (gold-evidence
injection moves the ceiling +0–1.2pp: presence is not the binding constraint),
syntactic format (2–6pp, indistinguishable from noise for 3 of 4 readers), and
evidence-side conflict marking (fully substituted by a one-line prompt).
Only query-regime conditioning survives: a question-type-conditioned policy
with a coverage-based do-no-harm ladder gains +2.5pp over the strongest fixed
baseline under leave-one-benchmark-out evaluation [PENDING: confirm run],
while avoiding every collapse in our response-surface table. We release the
compiler, the frozen harnesses for all four audits, and per-cell results.

---

## 1. Introduction (full draft)

Every retrieval-augmented system makes a decision that rarely appears in its
paper: after retrieval and before reading, the evidence is *compiled* — passed
raw, truncated to a budget, filtered, extracted, summarized, or restructured.
The compression literature treats this step as an efficiency knob. The memory-
systems literature buries it inside write-time design. Both treat the choice of
representation as second-order.

It is first-order. Under one pipeline, one retriever, one reader set, and one
judge, we measure the gap between the best and worst representation choice at a
matched token budget and find it is not a few points but 30–40 — and the sign
flips across regimes. On ConvoMem (conversational memory), generic structured
compilation scores 66.5–70.5 while our prior published compiler scores 46–48
and summary-style compilation wins elsewhere; on HotpotQA at a 400-token
budget, query-focused summary scores 71 while *every* extractive, filtered, and
structured condition scores 25–40, because two gold paragraphs do not survive
truncation but do survive summarization. On MuSiQue, summary nearly triples raw
(49.0 vs 19.6). Budget itself flips: going from 0 to 400 evidence tokens is
worth +21–41pp on HotpotQA and ~0 on ConvoMem. A system that fixes any single
policy is silently paying tens of points somewhere; a paper that evaluates one
representation on one benchmark is reporting a coordinate, not a result.

This raises the question the adaptive-RAG literature is implicitly racing to
answer: *conditioned on what* should compilation adapt? Four candidate axes
have been proposed or are natural. (i) **The reader**: compression gains are
known to be reader-dependent [prior paper], so condition the compiler on reader
capability. (ii) **Acquisition**: if compiled evidence fails, retrieve more or
interrogate the store. (iii) **Format**: presentation effects of up to 10–20pp
have been reported [LongMemEval CP4]; condition on syntax. (iv) **The query
regime**: different question types demand different representations. Adaptive
compression papers currently occupy only slices of this space — ratio
adaptation by query complexity (AdaComp, ACC-RAG, PoC), task-signal RL (ZipRL),
white-box reader familiarity (FAVICOMP) — and none audits the axes against each
other under one pipeline.

We do that audit. Our instrument is compile-once/replay-many: every compilation
policy's output is cached and replayed unchanged across readers, so reader
variation, policy variation, and benchmark variation are measured on the same
fixed evidence. Each axis gets a pre-registered kill criterion *before* the
deciding run. Three axes die:

- **Reader conditioning is unlearnable.** On a 21-reader × 500-row × 3-style
  panel, the per-(row, reader) oracle beats the reader-blind row oracle by only
  +2.6–3.4pp after judge-noise correction — and every realizable
  reader-conditioned policy (per-reader best-fixed, per-(reader, question-type),
  calibration-weighted row transfer) captures ≤0.3pp of it under both query
  holdout and reader-family holdout. The mechanism is diagnosable: per-row
  compression deltas of any two readers correlate at only 0.38 (structured) /
  0.16 (summary); averaging across readers denoises, conditioning on reader
  identity re-injects noise. Robust reader×style preferences do exist (split-half
  r = 0.845; OLMo-32B +18.2 on summary where MiMo-v2.5 is −10.0) — but they are
  monotone in baseline capability, so a scalar captures them, and that scalar
  buys +0.2pp.
- **Acquisition is not the binding constraint.** Probe-based interrogation of
  the memory store finds evidence unreachable by one-shot retrieval (24.0% of
  rows on BEAM at budget 30) — but the effect decays to zero by budget 180 on
  LongMemEval (negative control), conversion to reader accuracy is null at
  matched budget, and injecting *gold* evidence moves the ceiling only
  +0–1.2pp [PENDING: 3-sample re-run under fixed eval]. The hard rows are
  reasoning-bound, not information-starved.
- **Format is noise.** Holding content byte-identical and varying only
  syntactic wrapping (prose/numbered/JSON/table/headers) across 4 readers ×
  100 rows: best-vs-worst swings of 2–6pp, with 95% CIs including zero for 3 of
  4 readers — and the readers disagree on which format is best. A previously
  observed ~20pp "format" effect is shown to be a prompt artifact.
- (**Conflict marking**, a fifth, narrower axis:) deterministic evidence-side
  contradiction marks, which looked load-bearing on BEAM, are fully substituted
  by a one-line conflict-aware prompt on WikiContradict (marks+prompt −0.2pp vs
  prompt alone); the real variable is whether read-time processing *preserves
  both sides* at all.

One axis survives. Pooled over benchmarks, per-query representation routing has
+6.5pp of headroom over the best fixed policy, and it concentrates exactly
where fixed policies collapse: +9.6pp on LongMemEval, +11.2pp on ConvoMem,
+8.6pp on LoCoMo, versus only +1.8–3.8pp on open-domain QA. At realizable
granularity — question type — a reader-blind policy gains +2.5pp over the
strongest fixed baseline, and on LongMemEval the headroom decomposes cleanly:
temporal reasoning → structured (+8.8pp), knowledge update → raw (+10.3pp),
multi-session → summary.

We ship this as **SIEVE v2**: a query-regime-conditioned compiler that maps
question type to {filtered-raw, extractive, summary, structured+quotes} × budget,
wrapped in a do-no-harm ladder driven by a stem-coverage sufficiency signal
(fall back filtered-raw → summary → uncapped raw). The guard is not decoration;
it is where the transfer claim lives. Our prior compiler's cross-benchmark
collapse decomposes into evidence starvation (56% of damage: mean 14.4 compiled
tokens against 617-token raw prompts) and stale-value binding (30%); the
generic guarded compiler eliminates the collapse (+20pp on ConvoMem over the
prior system, with zero benchmark-specific code) and its guard fire-rate alone
(20% on ConvoMem vs 93% on HotpotQA) separates regimes for free. Evaluated
leave-one-benchmark-out against always-summary — the embarrassingly strong
global best fixed policy (63.2% pooled) — the policy [PENDING: LORO confirm
run; kill criterion: beat always-summary+raw-fallback or we publish the audit
without the method claim].

Our contributions: (1) the response-surface result — representation choice is a
large, sign-flipping, regime-dependent effect, and the cost of the wrong fixed
choice dwarfs any routing gain; (2) a pre-registered audit of all four
adaptivity axes under one controlled pipeline, with three negative results that
each kill an active research direction, including the direct follow-up to our
own prior paper; (3) SIEVE v2, a query-regime-conditioned compiler with a
do-no-harm ladder that transfers where fixed policies collapse; (4) released
artifacts: the frozen audit harnesses, per-cell results, and the replay
infrastructure.

We are explicit about what this paper is not. It does not claim a large
headline gain — the realized routing win is +2.5pp, and we report it as such,
alongside the ~7pp row-level headroom we could not realize. The paper's claim
is that this deflation is *the finding*: post-retrieval re-representation is a
second-order effect on every axis once the first-order obligation — do no harm
across regimes — is met, and we give the field the map, the metrics, and the
one policy worth deploying.

---

## 2. Section plan with the initial results that go in each

### §3 The response surface (the stakes)

One table, one figure. Rows: benchmark regime; columns: representation @
budget; entries: accuracy (2–3 readers, pooled). Existing numbers:

| regime | raw | filtered | extractive | summary | structured (guarded) |
|---|---:|---:|---:|---:|---:|
| ConvoMem | 48–79 | — | — | 71–81 | **66.5–70.5** (v1: 46–48) |
| HotpotQA@400 | 25–40 range | 25–40 | 25–40 | **71** | 25–40 |
| MuSiQue | 19.6 | — | — | **49.0** | — |
| LME (strong readers) | ~ raw | — | — | 46–48.6 | **53.4–55.6** |

Plus the aperture row: raw@0 → raw@400 = +21–41pp (HotpotQA), ~0 (ConvoMem).
[PENDING: LoCoMo summary cells to complete the grid.]

### §4 Axis 1 — reader conditioning: negative (pre-registered)

The policy ladder table (from the 191k-row matrix analysis):

| policy | acc |
|---|---:|
| always structured | 52.8 |
| per-reader best fixed | 53.0 |
| per-question-type, reader-blind | **55.3** |
| per-(reader, question-type) | 55.3 (+0.0) |
| reader-blind row oracle | 62.4 |
| per-(row, reader) oracle | 65.8 |

Plus: leave-family-out row transfer (+0.25pp max), the 0.38/0.16 noise
correlations, the existence pairs (OLMo +18.2 / MiMo −10.0), and the
noise-corrected oracle gap (2.6–3.4pp). This section *corrects the
implication* of our prior paper's "30% of headroom needs reader identity."

### §5 Axis 2 — acquisition: negative (pre-registered, 3 gates)

BEAM probe-exclusive reachability 24.0% @B=30 → LME decay to 0 by B=180
(negative control) → conversion null at matched budget → gold-injection
ceiling +0–1.2pp. [PENDING: conversion re-run under fixed eval, 3 samples.]

### §6 Axis 3 — format: negative (pre-registered)

The 4-reader × 5-format table (content byte-identical, providers locked):
swings 2–6pp, bootstrap CIs include 0 for 3/4 readers, no capability trend
(MiMo v2.5 vs v2.5-pro), best-format disagreement across readers. The ~20pp
sighting resolved as a prompt artifact (documented in the artifact appendix).

### §7 Axis 4 (narrow) — conflict marking: substitutable

WikiContradict 2×2 (n=253 × 2 readers): plain+aware 23.7 vs marked+standard
4.9 vs marked+aware 23.5. Marks help only when the prompt is absent. The BEAM
0→33→50 ladder reinterpreted: write-time resolution deletes a side; *any*
read-time preservation fixes it. (One open angle, stated as future work:
detector-gated marking on mixed conflict/no-conflict cohorts.)

### §8 The axis that pays — SIEVE v2 (method + LORO evaluation)

Policy definition (question-type → representation × budget), the do-no-harm
ladder, the coverage signal. Damage decomposition of v1's collapse (starvation
56%, stale-binding 30%), guard fire-rate as regime detector (20% vs 93%).
Headline evaluation: LORO vs always-summary (63.2 pooled) and vs
domain-aware-fixed (+0.9). [PENDING: the LORO confirm run — this is the
paper's one live bet.] Report: realized gain, per-regime damage rates,
headroom split realized/unrealized (+2.5 realized / ~7pp row-level),
and the insurance claim (zero cells below raw−CI on any held-out benchmark).

### §9 Measurement layer (woven in, not a thesis)

Per-model best-prompt reporting, judge validity audit, 3-sample reliability on
headline cells [PENDING], the presentation-sensitivity appendix (four artifact
sightings incl. the span-prompt fix), supersession appendix (KU chain-rendering
negative with mechanism; CR ladder with the format-circularity caveat stated
by us first).

---

## 3. How this is NOT the previous paper (and where the contamination risk is)

| | arXiv 2606.21807 (prior) | this paper |
|---|---|---|
| Axis varied | **reader** (20 readers, evidence frozen) | **query regime / distribution** (representations × benchmarks; readers are the instrument) |
| Genre | diagnosis: fixed compression corrupts *measurement* | audit + method: which adaptivity is real; ship it |
| Headline numbers | r = −0.887, 80% upgrade absorption, 31% rank flips | 30–40pp sign-flips; 4-axis audit; +2.5pp realized routing; +20pp collapse elimination |
| Method claim | explicitly none ("we do not propose a new compressor") | SIEVE v2 policy + do-no-harm ladder, LORO-evaluated |
| Benchmarks | LME-S primary; HotpotQA/MuSiQue/NQ/QMSum as replication of the reader trend | ConvoMem, LME, LoCoMo, BEAM, HotpotQA/MuSiQue as *transfer regimes* (ConvoMem and the cross-benchmark axis do not appear in the prior paper at all) |
| Tables reused | — | **none** (rule: every table here is from v2 runs or new matrix analyses that postdate submission) |

**Contamination rules (the "stinky paper" guards):**

1. **No shared tables, no shared figures.** Every number in §3–§9 comes from
   v2 runs (`results/v2_runs/`), the router-matrix analyses dated 2026-07, or
   new pilot runs. The prior paper's numbers appear only as *cited* context
   (one sentence in the intro, one row in related work).
2. **The shared asset is infrastructure, not results** — compile-once/replay-many
   and the reader panel. That is legitimate reuse (an instrument), and we say
   so explicitly in the apparatus section, citing the prior paper as the
   instrument's validation.
3. **⚠ One genuine double-dipping risk to resolve:** the reader-conditioning
   negative (§4) is also the single best *revision* for the prior paper's
   resubmission (it answers its reviewers' "diagnostic, no method" by proving
   the method can't exist on that axis). **It cannot appear as a contribution
   in both.** Decision needed: (a) if the prior paper is being revised for this
   ARR cycle, §4 belongs there, and this paper compresses §4 to a citation +
   one confirming paragraph on the new 6-style surface; (b) if the prior paper
   ships as-is, §4 stays here in full. Pick before writing either.
4. ARR submission form: declare the prior paper under "reuse of artifacts,"
   link the matrix release, state the zero-table-overlap rule in the
   reproducibility checklist.

---

## 4. Honest status ledger

**In hand:** response-surface pilots (ConvoMem/HotpotQA, guarded-generic
compiler built + tested, 341 tests passing); matrix analyses (policy ladder,
LORO-style pooled routing, noise decomposition); format-lever run (complete,
frozen); WikiContradict 2×2 (complete, deterministic scoring); interrogator
arc (complete except eval-fix re-run); damage forensics (starvation/stale
split); BEAM retrieval-completeness diagnosis per ability.

**Pending (the ~$30–40 must-run list):** eval fix everywhere numbers enter
tables; conversion re-run (3 samples); LoCoMo summary cells + LoCoMo in LORO;
**the LORO confirm run of the policy vs always-summary (the one live bet)**;
frontier probe on 20 event_ordering rows (decides one paragraph); 3-sample
reliability on headline cells.

**Kill criterion (pre-registered here):** if the question-type policy does not
beat always-summary+raw-fallback on pooled LORO accuracy with damage-rate no
worse than raw−CI, §8's method claim is withdrawn and the paper ships as the
four-axis audit with the response surface as headline — which is the floor
version, and still a paper.

**Failure modes to pre-empt in writing** (from the spec): "why not always
summarize" → §3 shows regimes where summary loses 20–30pp, and the policy's
value is regime detection + insurance, reported as damage rates, not means;
"+2.5pp is small" → stated small, framed as the honest size of the entire
area; "isn't this your last paper" → §3 table above, zero overlap; write-time
systems (Honcho 90.4 LME) → different regime (ingestion cost, post-hoc
applicability), one positioning paragraph + long-context control where
feasible.
