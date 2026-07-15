# Learned components: concrete design (2026-07-14)

Companion to `v3-learned-compiler-proposal-2026-07.md` (construction) and
`read-time-compilation-theory-2026-07.md` (capabilities). Design stance:
**three small learned pieces, everything else engineered.** The learning
lives mostly in the labels (counterfactual reader outcomes), not in model
size. DeferMem's DistillPO shows RL is viable here, but we choose supervised
utility regression first: sample-efficient, stable, auditable, and our label
assets are exactly the supervision RL would have to rediscover.

## L1 — Sufficiency estimator  s(q, pack) -> [0,1]

*Predicts whether an evidence pack supports answering q. Drives the
do-no-harm ladder: escalate filtered_raw -> summary -> uncapped raw until
predicted sufficient or budget cap.*

- **Supervision (two independent sources, in order of cleanliness):**
  1. Gold-evidence containment: HotpotQA supporting facts, BEAM
     `answer_bearing_memory_ids`, LongMemEval gold session ids. Label =
     fraction of gold units surviving in the pack (binary at 1.0/partial/0).
     ~1,100 rows available today at zero cost.
  2. Reader outcomes from the matrix + pilots: pack judged correct by
     majority of readers. Noisier (judge noise 2.5%, reader variance) but
     covers packs without gold annotations.
- **Model ladder:** v0 = logistic regression / GBM over overlap features
  (stem coverage, entity coverage, date coverage, unit count, pack/query
  length ratio) — the pilot guard is the 1-feature special case and fired
  20% vs 93% across domains, so signal exists. v1 = small cross-encoder
  (bge-reranker-base or deberta-v3-small fine-tune, runs on the M1) only if
  v0's transfer AUC is inadequate.
- **Eval / kill:** AUC on held-out benchmark (train LME+HotpotQA, test
  ConvoMem/BEAM). Must beat the raw stem-overlap guard by a margin that
  changes ladder decisions; otherwise ship the guard and say so.
- **v0 VERDICT (2026-07-14): NEGATIVE — kill criterion fired; ship the
  guard.** (`research/l1-sufficiency-v0-results-2026-07.md`, 8,167 labeled
  rows.) GBM beats the guard on transfer AUC (+18pp on every held-out
  benchmark) but produces WORSE decisions at matched fire rate on all
  three, and the plain stem-coverage feature outcorrelates it with reader
  outcomes on ConvoMem transfer (rho 0.56 vs 0.36-0.48). Root cause:
  pack-statistics features saturate (BEAM guard fire rate 98.5%); the
  ranking gain never converts to routing gain. Consistent with the
  pilot-decomposition finding that oracle-win queries are inseparable by
  pack statistics. The v1 rung (small cross-encoder over content) is the
  justified next step ONLY as part of L3 pretraining — do not iterate more
  hand features.

## L2 — Representation + budget policy  pi(style, budget | q, C)

*The core learned object. Chooses among engineered actions:
{filtered_raw, extractive, summary, structured_generic(+supersession)}
x {200, 400, 800, uncapped} x {standing digest on/off}.*

- **Formulation:** NOT classification. Multi-output utility regression
  u_hat(a | q, C) for every action, then argmax_a [u_hat - lambda * cost(a)].
  This gives the Pareto knob for free, handles action-set changes, and lets
  us report calibrated expected-utility gaps rather than opaque choices.
- **Features (all cheap, domain-general, no benchmark vocabulary):**
  - query: length, wh-form, small frozen embedding (e.g. bge-small, 384-d);
  - candidate set: n units, total tokens, retrieval-score max/entropy/gap,
    pairwise redundancy, date-parse rate, date span;
  - structure signals: entity/attribute cluster count, max cluster size,
    conflict count (clusters with divergent values), supersession-chain
    length — computed by the deterministic supersession module;
  - sufficiency signals: L1 score per candidate action, guard coverage.
- **Labels:** per-(query, action) normalized accuracy averaged over the
  cheap reader panel, from: existing matrix cells (raw/structured_v1/summary
  on 6 slices), the $1.50 pilot (7 conditions x 2 slices), and one funded
  response-surface run (~500 queries x 4 benchmarks x 8 actions x 3 readers
  ~= $25-40 at pilot rates). Order of 3-4K labeled (query, action) rows.
- **Model:** LightGBM, deliberately small. The contribution is the
  conditioning and the labels, not the architecture (same stance as C3 in
  the ICLR assessment).
- **Protocol:** leave-one-benchmark-out AND query-holdout within benchmark;
  isotonic calibration of u_hat before the lambda sweep.
- **Kill / honesty bar:** must beat, under LOBO: (a) **always-summary with
  the canonical prompt** — the strongest no-learning policy (68.9 pooled
  over 5 matrix slices; the pool-fit ladder was retracted after failing
  frozen-rule transfer) — and (b) domain-aware best-fixed. If the GBM
  cannot beat always-summary on the conversational-memory slices (the only
  place with realizable headroom: LME +8-11pp, ConvoMem +7.8pp LORO), the
  method reduces to always-summary + engineered ops and the learned framing
  drops to an ablation — stated in advance. Every candidate heuristic or
  learned rule gets a frozen-rule transfer test on unseen slices before
  being reported.

## L3 — Per-unit utility scorer  u(q, unit | C)

*Ranks units for selection within a budget; feeds coverage features to L1/L2.*

- **Two-stage supervision:**
  1. Pretrain on gold-evidence labels (HotpotQA supporting facts as
     positives vs distractor paragraphs; BEAM answer-bearing turns; LME gold
     sessions). This is plain cross-encoder training, ~10K unit labels free.
  2. Calibrate on counterfactual outcome labels: ContextCite-style masked
     ablations (random unit masks per query, fit a per-query linear
     surrogate of answer correctness/probability) on a cheap open reader.
     Since API logprobs are unreliable, use answer-level correctness with
     ~16-24 masks per query on llama-3.1-8b: 100 queries ~= 2K calls ~= $0.15.
     Full ContextCite (logprob surrogate) reserved for a local model if
     needed.
- **Why not train on v1 selected/rejected or retrieval rank:** both encode
  the policies we are replacing; the matrix shows inter-reader delta
  correlation is only 0.38, so single-reader counterfactuals must be
  averaged over >= 3 weak readers before use as labels.
- **Eval / kill:** recall@budget of gold units vs BM25-rank truncation
  (the incumbent). Downstream flip test: packs built by u vs by rank on a
  held-out slice, one cheap reader. If recall gains do not produce answer
  flips, the scorer is not worth its latency.

## Explicitly NOT learned (engineered ops the policy invokes)

- Supersession chains: deterministic entity/attribute clustering + date
  ordering + explicit update rendering. Auditable; its conflict counts are
  policy features.
- Standing-constraints digest: one prompted LLM pass per history, cached;
  provenance-linked. Learning what goes in the digest is future work.
- Packagers, floors, budget truncation: deterministic, tested.

## Sequencing (each step gates the next)

1. **L1 v0** on existing gold labels + pilot packs — days, $0.
2. **Heuristic ladder** (L1 v0 + escalation rules) re-scored on the pilot
   runs offline — this is the baseline L2 must beat.
   **DONE and RETRACTED 2026-07-14**
   (`results/v2_runs/heuristic_ladder_offline_v0.md`): the pool-fit ladder
   won on the two pilot benchmarks only because the summary comparator used
   the broken prompt. Frozen-rule transfer to the matrix slices fails
   everywhere; even in-sample-tuned thresholds never beat always-summary
   (68.9 = 68.9 pooled over 5 slices). Consequences now binding: the
   no-learning baseline L2 must beat is **always-summary (canonical
   prompt)**; the ladder inverts — summary is the base action and
   escalation to structured/raw must be earned by a predicted-utility
   signal; raw passthrough is NOT universal do-no-harm (MuSiQue raw 19.6 vs
   summary 49.0 at any pool size). Honest LORO headroom over
   always-summary: LME +8-11pp, ConvoMem +7.8pp, QA +0.2-2.8pp. Any new
   heuristic rule gets a frozen-rule transfer test on unseen slices BEFORE
   being reported as a result.
3. **Response-surface run** for L2 labels (~$25-40), then L2 + LOBO eval.
4. **L3** only if L2's residual-to-oracle analysis shows selection (not
   style/budget) is the binding constraint.
5. Supersession + standing-digest evaluations (BEAM contradiction,
   LoCoMo-Plus) run in parallel with 3 — they are engineered, not gated on
   learning.

## Principal risks

- **Label noise ceiling:** 2.5% judge disagreement + reader variance bounds
  achievable AUC/regression fit; average over readers, drop flagged rows,
  report noise-adjusted headroom (as in the matrix mining).
- **Benchmark-identity leakage:** features like date-parse rate correlate
  with domain. Acceptable — the deployment claim is transfer to *unseen*
  benchmarks, which LOBO tests directly; report feature ablations.
- **Small-N policy overfit:** ~3-4K rows, 8 actions; GBM with strong
  regularization + repeated LOBO; no deep nets.
- **Summarizer confound — FIXED 2026-07-14:** the v2 summary packager now
  uses the exact canonical LLM-Summarize prompt from the published baseline
  (`v2/packagers/summary.py`, test-pinned). The pilot's summary cells
  (59-63 on ConvoMem vs canonical 71-81) predate the fix; re-compile summary
  cells as part of the response-surface run.
