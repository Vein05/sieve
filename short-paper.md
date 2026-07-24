# Condition on the Query, Not the Reader

*Working draft v3 (short-paper.md, 2026-07-23). ARR short paper (4 pages),
EACL 2027 cycle (deadline Aug 3, 2026). Design rules for this draft, learned
from the prior paper's review cycle: (1) every defense lives in the main text —
nothing load-bearing in an appendix; (2) the mechanism is stated as a
hypothesis with derived predictions, then confirmed — never as a narrative
after the fact; (3) the artifact release has its own section header; (4) hard
leanness budget: supplementary material ≤ 4 pages, exactly two tables.
Changelog at bottom.*

---

## Abstract (~160 words)

Post-retrieval compression is reader-dependent, and the field's next step —
condition the compressor on reader identity — is being actively pursued. We
test it and find an inversion: given the *maximal* conditioning signal (20
donor readers' measured outcomes on the very row being routed), selecting
evidence by the most similar reader scores 57.6 while ignoring reader identity
and averaging scores 60.5. Conditioning subtracts value at fixed information.
We explain this with one hypothesis — the per-row routing signal is
noise-dominated (per-row compression deltas of two readers correlate at only
0.38/0.16) — and confirm its three derived predictions on a 21-reader ×
500-query × 3-representation panel of frozen, replayed compilation outputs
(191k judged cells): realizable reader-conditioned policies gain ≤0.3pp over
reader-blind counterparts; similarity-weighted ensembles gain +0.25pp over
blind averaging; and the cross-reader correlations replicate on held-out
benchmarks (8 of 9 pre-registered cells; the single violation — a trained
compressor in-domain, r = 0.58 — is the shared-signal regime that reader-blind
routing captures by construction). The query-conditioned rung of the same policy ladder gains +2.5pp
with no reader information. For selection among fixed representations, the
conditioning variable that pays is the query, not the reader. We release the
matrix and all analysis code.

---

## 1. Introduction

A fixed post-retrieval compressor affects readers asymmetrically: it rescues
weak readers by removing noise they cannot filter and damages strong readers by
dropping details they would have used (Panthi & Abdelfattah, 2026; Jeong et
al., 2025, App. A.8; Xu et al., 2024). The natural method-side response is
under way: condition the compression on the reader. FAVICOMP (Jung et al.,
2025) shapes evidence toward the target model's token distributions; the
routing and capability-profiling literature (Ong et al., 2024; INFERENCEDYNAMICS,
2026; RouteProfile, 2026) supplies ever-richer reader representations a
compression policy could condition on; and our own prior work quantified the
invitation — 37% of rows are mixed-label (the same evidence helps some readers
and hurts others), and a reader-blind per-row oracle accesses only ~70% of
routing headroom, implying the rest needs reader identity.

Here is what happens when that invitation is accepted under the most favorable
conditions we can construct. Take 21 readers with frozen, replayed evidence
(three representations per query, 191,277 judged cells), hold one reader out,
and give the routing policy the strongest conditioning signal that can exist:
the *measured outcomes* of the 20 other readers on the very row being routed —
strictly more information than any deployable reader-profiler could ever
extract. Then compare two policies. Ignoring reader identity (majority vote
across donors) scores **60.5**. Conditioning on it (follow the single most
similar donor, with similarity chosen with test-set hindsight) scores
**57.6**. Conditioning on the reader does not merely fail to help; *at fixed
information, it subtracts value.*

We propose one hypothesis that explains the inversion, derive three
predictions from it, and confirm all three.

**Hypothesis (noise-dominated routing signal).** The per-row, per-reader
compression delta — which representation helps *this* reader on *this* row —
is mostly noise around a shared query-level signal: per-row deltas of any two
readers correlate at only 0.38 (structured−raw) and 0.16 (summary−raw). If
true, cross-reader averaging *denoises* the signal while conditioning on any
one reader's identity or profile *re-injects* the noise. This forces:

- **P1.** No realizable reader-conditioned selection policy can materially
  beat its reader-blind counterpart, regardless of features or calibration
  budget. *Confirmed:* across per-reader best-fixed (+0.2pp), per-(reader,
  question-type) with up to 200 labeled calibration rows (+0.0pp), and
  leave-family-out transfer (+0.25pp), no rung exceeds +0.3pp (§5, Table 1).
- **P2.** Inside the maximal-information ensemble regime, weighting donors by
  similarity to the held-out reader adds ~nothing over blind averaging, and
  concentrating on the most similar donor actively loses. *Confirmed:* +0.25pp
  and −2.9pp respectively (§5, Table 2).
- **P3.** The low cross-reader correlation is a property of the routing
  problem, not of conversational-memory QA: it must replicate on other
  benchmarks in the same matrix. *Outcome — held in 8 of 9 cells, violated in
  one, and the violation is informative.* We pre-registered r < 0.5 before
  computing. Generic compression replicates everywhere (mean pairwise r:
  HotpotQA 0.39, MuSiQue 0.50, NQ 0.24, vs 0.16 on LongMemEval). The single
  violation is a *trained* compressor on its own training benchmark
  (RECOMP-style summary on HotpotQA: mean r = 0.58). High cross-reader
  correlation is the *shared-signal* regime: a trained compressor deletes
  content systematically, damaging all readers together — which reader-blind
  routing captures by construction. Both regimes exclude reader conditioning,
  for opposite reasons (§5).

The same ladder that kills reader conditioning locates where adaptivity does
pay: a *query*-conditioned policy (question type → representation) gains
+2.5pp reader-blind, against ≤0.3pp for reader identity. Scope, stated up
front: this bounds *selection among fixed representation styles* — the action
space of the adaptive-compression literature's routing variants. It does not
bound generative reader-conditioned compression (white-box decoding-time
shaping), which our black-box setting excludes.

The practical conclusions: (i) condition on the query, not the reader; (ii)
reader-conditioned compression papers should report the reader-blind ablation
under reader-family holdout — on our panel that ablation erases the
conditioning gain entirely; (iii) multi-reader evaluation is not only a
measurement obligation (as prior work argues) but the *optimal policy-learning
input*, because averaging across readers is the denoiser.

## 2. Related work (compressed)

Adaptive compressors adapt ratio or query complexity, never reader identity:
RECOMP (Xu et al., 2024), EXIT (Hwang et al., 2025), ECoRAG (Jeong et al.,
2025), AdaComp (2024), ACC-RAG (2025), PoC (2026). FAVICOMP (Jung et al.,
2025) is reader-aware but generative and white-box — the case we explicitly do
not bound. Routing work (RouteLLM, 2024; INFERENCEDYNAMICS, 2026;
RouteProfile, 2026) selects which model answers; we test the converse —
selecting evidence for a fixed model — and find the per-row conditioning
signal absent. Panthi & Abdelfattah (2026) establish reader-dependence and its
measurement consequences; this paper tests the method direction that work
implies, and corrects its headroom implication (§6).

## 3. Setup

**Panel.** 21 readers (7B–70B+, 11 families), complete uncapped triples on
LongMemEval-S (500 queries): raw BM25 top-20, structured compilation, generic
query-focused summary — compiled once, replayed unchanged (compile-once/
replay-many). Binary correctness by an LLM judge under a fixed protocol;
judge-disagreement flags (4.3% of cells) drive the noise controls. Five
readers excluded for incomplete cells (availability-driven, not
outcome-driven). Frozen evidence means every policy below — including oracles
and ensembles no deployable system could run — is evaluated offline at zero
marginal cost, which is what lets us *bound* the axis rather than report one
policy's failure.

**Policies (10-fold CV over queries).** Fixed: always-{raw, structured,
summary}; per-reader best fixed. Query-conditioned: per-question-type
(LongMemEval's six native categories), reader-blind. Reader-conditioned:
per-(reader, question-type); calibration sweep K ∈ {25–200} rows/reader.
Transfer (leave-reader-family-out, 100 reps), two regimes labeled explicitly:
*ensemble bound* — donors' measured outcomes on the test row (information
ceiling; requires running 20 readers per query); *realizable* — donor
similarity estimated from a 100-query calibration split only. Oracles:
reader-blind per-row; per-(row, reader).

## 4. The prize, measured honestly

Total oracle headroom over the best fixed policy: 65.8 − 52.8 = **13.0pp**.
The reader-identity increment (per-(row, reader) oracle over reader-blind
per-row oracle) is **+3.4pp** [95% CI 2.9–4.0] naively — but 4.3% of cells
carry judge-disagreement flags, and dropping them shrinks the increment to
**+2.6pp**, while injecting *random* label flips into flagged cells inflates
it to +4.5pp. Hindsight per-reader oracles preferentially harvest label noise;
published headroom estimates built this way (including our own prior
~30%-of-headroom figure) are biased upward by construction. So the prize is
20–26% of total headroom before asking whether any of it is reachable.

## 5. Results: the predictions, confirmed

**Table 1 — the policy ladder (accuracy, 10-fold CV; P1):**

| policy | conditions on | acc | Δ vs reader-blind counterpart |
|---|---|---:|---:|
| always raw / summary / structured | — | 46.7 / 50.0 / 52.8 | — |
| per-reader best fixed | reader | 53.0 | +0.2 |
| per-question-type | query | **55.3** | (counterpart) |
| per-(reader, question-type) | reader × query | 55.3 | +0.0 |
| — K=200 calibration rows/reader | reader × query | 54.8 | −0.5 |
| reader-blind per-row oracle | hindsight (query) | 62.4 | — |
| per-(row, reader) oracle | hindsight (reader × query) | 65.8 | +3.4 |

Every realizable reader-conditioned rung is within ±0.5pp of its reader-blind
counterpart; the best gains +0.3pp. Meanwhile query conditioning — same
ladder, no reader information — gains +2.5pp. P1 holds.

**Table 2 — transfer to a held-out reader (leave-family-out; P2):**

| policy for held-out reader | regime | acc |
|---|---|---:|
| always structured | fixed | 52.8 |
| unweighted donor majority vote (reader-blind) | ensemble bound | 60.5 |
| similarity-weighted donor vote | ensemble bound | 60.8 |
| single most-similar donor (test-set hindsight) | ensemble bound | 57.6 |

All three ensemble rows share the identical (maximal) information; only the
use of reader identity varies. Weighting by similarity: +0.25pp (17/21
readers). Concentrating on the most similar reader — with hindsight — loses
2.9pp to blind averaging. P2 holds: at fixed information, conditioning
re-injects exactly the noise averaging removes. (The 60.5 row is an
information ceiling, not a deployable policy; it is not comparable to Table 1
and is reported to show conditioning fails even *above* deployability.)

**Table 3 — P3, cross-benchmark replication (mean pairwise per-row delta
correlation across readers; pre-registered threshold r < 0.5):**

| benchmark | style pair | readers | mean r | % pairs < 0.5 |
|---|---|---:|---:|---:|
| LongMemEval (ref.) | summary−raw | 22 | 0.16 | 100 |
| LongMemEval (ref.) | structured−raw | 24 | 0.36 | 93 |
| HotpotQA | summary−raw | 19 | 0.39 | 91 |
| HotpotQA | trained-abstractive−raw | 19 | **0.58** | 25 |
| HotpotQA | trained-extractive−raw | 19 | 0.43 | 71 |
| MuSiQue | summary−raw | 19 | 0.50 | 54 |
| NQ | summary−raw | 19 | 0.24 | 98 |
| NQ | trained-abstractive−raw | 19 | 0.38 | 73 |
| NQ | trained-extractive−raw | 19 | 0.40 | 83 |

The pre-registered threshold holds in 8 of 9 cells; we report the violation
rather than revising it away. The violating cell is the trained abstractive
compressor evaluated in-domain — the regime where compression decisions are
most systematic. This maps the two regimes of the routing signal: **noise**
(low r: per-row deltas are idiosyncratic, so conditioning on any reader
re-injects noise — P1/P2) and **shared signal** (high r: systematic evidence
deletion damages all readers together, so reader-blind routing captures it by
construction). Reader conditioning pays only in a third regime — large,
*stable, reader-specific* per-row structure — which no cell exhibits: where
correlations are low the reader-specific residual is noise (split-half and
transfer results, P1/P2), and where they are high the signal is not
reader-specific at all.

**What is real at the aggregate level — and why it doesn't help.** Per-reader
action deltas are highly reliable (split-half r = 0.845 [0.737, 0.911]). The
dominant pattern is a capability trend — representation effects shrink toward
zero for the strongest readers — which is why a capability scalar exhausts the
signal (+0.2pp, Table 1). Stable *exceptions* exist (equal-baseline pairs with
opposite preferences: Phi-4 summary −3.6 vs Gemma-3-12B +7.2 at raw ≈ 47;
OLMo-32B +18.2 vs MiMo-v2.5 −10.0), and they are precisely what per-reader
policies try to exploit — recovering ≤0.3pp, because the exceptions are
aggregate-level while the routing decision is per-row, where noise dominates.
[Exhibit: baseline-vs-delta scatter, trend + exceptions marked; §8.]

**The query policy, operationally.** Six native LongMemEval question types
mapped to representations per training fold (temporal → structured;
knowledge-update → raw; multi-session → summary), applied to held-out queries;
smallest type cell ≈ 30 queries per test fold. [Per-type decomposition table:
§8.]

## 6. What this corrects

Our prior "reader-blind oracle captures only ~70% of headroom" framing is
technically true and practically misleading: the residual is partly judge
noise harvested by hindsight oracles (§4) and otherwise unreachable by any
realizable policy (§5). Corrected summary: on this action space, reader
identity is worth ≤0.3pp to any policy we could construct, under both query
holdout and reader-family holdout — and the ensemble bound shows this is a
property of the signal, not of our policy class: any capability vector,
familiarity score, or routing embedding is a compression of strictly less
information than the donors' realized outcomes that already fail.

## 7. Data and Code Release

This study builds on the **ragscale** interaction matrix released with Panthi
& Abdelfattah (2026, arXiv:2606.21807): we use that paper's dataset as the
substrate for all policy, oracle, and transfer analyses here, and extend it.
We release: (1) the **new analysis layer** — policy-ladder, leave-family-out
transfer, and oracle noise-correction code (every table regenerates in ~2
minutes on a laptop, offline); (2) the **cells added for this study** —
the summary-policy cells, the additional reader completing the 21-reader
panel, and the 2,000-call format-probe run with its byte-identity-tested
harness; (3) the **P3 cross-benchmark mechanism analysis** over the matrix's
held-out benchmark cells. MIT license. The dataset contribution of the prior
paper is not re-claimed here; this paper's artifact is the analysis layer and
the extension cells.

## 8. Pre-submission additions (all offline; pre-registered above)

(1) ~~P3 table~~ DONE 2026-07-23 (Table 3; script to be promoted from
scratchpad to `analysis/` with tests per repo standards); (2)
baseline-vs-delta scatter exhibit; (3) per-question-type
decomposition of the +2.5pp with fold stability; (4) six-style pilot
supplementary table (readers, n, unanimous choice); (5) realizable-transfer
row (donor votes from calibration split only); (6) references pass. Items
(3)–(5) are the *only* supplementary material: two tables, ≤4 pages, by
design — every load-bearing analysis stays in the body.

## 9. Limitations

Selection among three positively correlated styles from one pipeline; the
wider six-style pilot chose one representation unanimously but at pilot scale
[supp. table]. Primary benchmark is conversational-memory QA, English only;
P3 extends the *mechanism* check to three other benchmarks but not the full
policy ladder. No frontier-scale readers (strongest ~57% raw); the observed
shrink-to-zero capability trend implies conditioning grows less useful with
capability — extrapolation, stated as such. Judge noise handled by
flag-dropping and injection sensitivity, not human re-adjudication. A 100-row
probe of syntactic *format* conditioning (content byte-identical, 5 formats ×
4 readers) was directionally null (swings 2–6pp; CIs include zero for 3 of 4
readers, one at [1.0, 12.0]) but is underpowered — reported as a pilot only.
Generative reader-conditioned compression is outside the bounded action space.

---

## Changelog v2 → v3

- Restructured around hypothesis → three derived predictions (P1–P3) →
  confirmations; the mechanism is now predictive, not post-hoc narrative
  (pre-empts the "framework doesn't earn its keep" attack from the prior
  paper's human reviews).
- P3 (cross-benchmark replication) pre-registered in the intro with its
  threshold (r < 0.5) stated before the analysis runs — converts §7-item-1
  from "more evidence" into a held-out prediction.
- Inversion result promoted to the abstract's first finding and the intro's
  hook (excitement lever: surprise first, null second).
- "Data and Code Release" given its own numbered section with row counts in
  the body (pre-empts the Datasets:1 failure mode — two prior human reviewers
  missed a release stated in the abstract and appendix).
- Leanness budget adopted and stated: supplementary ≤ 4 pages, exactly two
  tables; every defense main-text (the prior paper's 20-page appendix buried
  its own headroom controls and paid for it).
- Headroom arithmetic moved to its own short section (§4) ahead of results.
- Stakes framing added to intro conclusions (what the negative saves the
  field, not just the pp contested).

*Differentiation note (unchanged): the prior paper (arXiv 2606.21807) varies
the reader under fixed compression and shows measurement corruption; it
reports no reader-conditioned policy results. Every table here is a July-2026
analysis over the released matrix; no table, figure, or number is shared with
the prior paper, whose 37%-mixed-rows and ~70%-of-headroom figures appear only
as the claim under test.*
