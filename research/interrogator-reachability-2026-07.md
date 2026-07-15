# Interrogator reachability kill-test (2026-07-14)

Pre-registered kill-test for the "compiler as interrogator" bet (CONTEXT.md,
"Direction decision 2026-07-14 (evening)"). Question: for rows where gold
evidence is NOT fully in the BM25 top-20 pool, what fraction of missing gold is
**probe-reachable but NOT k-reachable** at matched acquired-unit budget? That
exclusive fraction is the interrogator's only defensible value; everything a
bigger `k` recovers is mere aperture.

Pre-registered read: exclusive fraction (B) >= ~15% of missing gold on the
retrieval-bound types => bet lives; < ~5% => bet dies; between => judgment call.

Code: `analysis/interrogator_{bm25,probes,reachability,report}.py`.
Tests: `tests/analysis/test_interrogator.py` (12, all pass). Local, deterministic,
no network, no LLM. Run: `python -m analysis.interrogator_reachability`.

## Data inventory

**Found (usable):**
- BEAM top-20 slice with gold labels:
  `/Users/vein/Documents/research/memory-eligibility-feasibility/dataset-slices/beam_transfer_bm25_top20_v0.jsonl`
  (280 rows = 7 abilities x 40; per-row `evidence_sufficiency.answer_bearing_memory_ids`
  are gold `turn_NNNN` ids; `turn_NNNN` encodes global turn order).

**Found but NOT usable as full history:**
- BEAM raw chats: `.../data/beam/repo/chats/` — only the **100K** size is
  materialized locally (21 dirs). The slice is **100% 128K** (`chat_size`), and
  128K turn contents do **not** match the 100K chats (BEAM regenerates histories
  per size; verified by content lookup). The 128K/500K/1M/10M full histories live
  on HuggingFace (`Mohammadta/BEAM`) — network-gated, disallowed here.
- BEAM adapted slices (`beam_adapted_bm25_top20_v{1,2}.jsonl`) are also 128K
  (`example_id` = `beam-128k-*`), same absent-history problem.

**Not found at all:**
- LongMemEval-S haystack `data/longmemeval/longmemeval_s_cleaned.json` (the source
  the slice builder `scripts/build_dense_retrieval_slice.py` reads) — directory is
  empty in both repos; no LFS pointer; not embedded in any compilation cache or run
  artifact. The reinjection artifact stores only single reinjected facts.
- LongMemEval gold ids: the sieve-repo slice
  `dataset-slices/longmemeval_bm25_top20_v1.jsonl` has **empty**
  `evidence_sufficiency` and `gold_decision`. Gold is only implicitly marked by the
  `answer_`-prefixed `memory_id` convention, and **all** answer sessions are
  injected into the top-20 by construction (498/500 rows carry >=1 `answer_` unit
  in-pool). Without the full haystack there is no observable "missing gold"
  population for LME and nowhere to probe.

**Consequence:** the kill-test runs on **BEAM only**. LME cannot be run locally.

### Reconstructed store (the enabling workaround)

The true 128K history is absent, but ~14 questions share each of the 20 distinct
128K chats. Unioning the 20 top-20 pools per chat reconstructs a partial content
store of **61-101 distinct turns per chat (median 93)** — vs BEAM's reported
~107 turns/128K-chat, so we recover ~85% of turns. 80.9% of missing-from-own-pool
gold turns have their content present in this union store; the remaining 19.1% are
counted as `content_absent` (unreachable by construction). Both the k-expansion
arm and the probe arm operate over this same reconstructed store, so the head-to-head
exclusive comparison is fair; the store is only a lower bound on the true store.

## Matched-budget protocol

At budget B extra units: k-arm = original query BM25, take next B beyond the
row's pool; probe-arm = P deterministic probes x top-m each, dedup vs pool and each
other, capped at B unique new units. Both exclude the row's own 20 pool ids and rank
over the reconstructed store. Budgets: B=30 (P=5,m=6), B=80 (P=10,m=8), B=180
(P=15,m=12). BM25: own Okapi impl (k1=1.5, b=0.75), lowercase-alphanumeric tokens.

Probe types implemented (all pool+query only, no gold/answer/id access — guarded
by `test_probe_no_gold_leakage`):
- **entity**: top capitalized spans from pool units, prepended to the query.
- **chain**: entities appearing with >=2 distinct dates in the pool (candidate
  update chains) + attribute terms (date/when/changed/updated...).
- **temporal**: distinct dates / date-expressions mined from pool units.
- **decomposition**: query content-word bigrams not covered by any pool unit.
- **adjacency** (BEAM-specific): turns at `turn_index +/- 2` of any in-pool turn
  (BEAM ids encode order); a legitimate probe type since order is a pool signal.

## The store-size caveat is decisive for budget choice

The reconstructed store is only ~90 units. B=80 and B=180 exceed 50% of the store
for **99/99 rows** — the k-arm trivially vacuums the entire store, driving
probe-exclusive to 0 by saturation, not by k-superiority. Those budgets are
scientifically void here. **Only B=30 (0/99 rows saturated) is interpretable.**
All headline numbers below are B=30.

## A/B/C classification (B=30, the valid budget)

Accessible = missing gold whose content is in the reconstructed store.
A = k-reachable; B = probe-reachable AND NOT k-reachable (the number); both =
recovered by either; C unreachable = accessible-not-recovered + content_absent.

| Cohort | rows | missing | accessible | A k-reach | **B probe-excl** | both | k-excl | C unreach | **B / accessible** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| POOLED (all 6 answerable abilities) | 99 | 183 | 148 | 94 | **32** | 91 | 3 | 57 | **21.6%** |
| RETRIEVAL-BOUND (event_ordering, multi_session, temporal, contradiction) | 83 | 167 | 136 | 84 | **31** | 81 | 3 | 52 | **22.8%** |

Most important type breakdowns (B=30):

| Ability | rows | accessible | A k-reach | **B probe-excl** | C unreach | **B / accessible** |
|---|---:|---:|---:|---:|---:|---:|
| event_ordering | 38 | 89 | 50 | **24** | 34 | **27.0%** |
| multi_session_reasoning | 24 | 33 | 23 | **6** | 9 | **18.2%** |
| temporal_reasoning | 7 | 5 | 3 | **1** | 3 | 20.0% |
| knowledge_update | 5 | 5 | 4 | **1** | 0 | 20.0% |
| contradiction_resolution | 14 | 9 | 8 | **0** | 6 | 0.0% |
| information_extraction | 11 | 7 | 6 | **0** | 5 | 0.0% |

Reverse check: k-exclusive (k recovers, probes miss) is only 3/136 — probes nearly
dominate k on what k finds. Absolute k-reachability (arm A) is 61.8% of accessible
missing gold: a bigger k alone already recovers most missing gold, confirming the
"mere aperture" risk is real but leaves a probe-only residue.

## Probe-type ablation (B=30, what did the recovering)

Retrieval-bound recovered-by-type: **entity 83, adjacency 29**, chain/temporal/
decomposition 0. Two probe families carry the entire exclusive signal:
- **entity probes** (query + a distinctive capitalized span from the pool) are the
  workhorse — they re-rank the store toward a specific project/person/feature the
  broad query under-weights.
- **adjacency probes** (fetch turns temporally next to in-pool turns) recover the
  event_ordering residue specifically — unsurprising, since ordering gold is
  contiguous with retrieved anchors. This is the one type that is genuinely a
  "state-semantics" probe rather than a rephrased query.

Chain/temporal/decomposition probes contributed nothing exclusive here. On the
tiny reconstructed store, date-string and bigram probes retrieved units the entity
probe or k-arm already had.

## Honest limitations

1. **BEAM-only, no LME.** The retrieval-bound LME types (multi-session,
   temporal-reasoning) — the CONTEXT-flagged bottleneck — could not be tested. The
   verdict rests entirely on 83 BEAM rows.
2. **Reconstructed store, not true history.** The store is the union of *already
   BM25-retrieved* pools, so it is pre-filtered toward query/entity relevance. This
   almost certainly **inflates** entity-probe reachability (the missing gold that
   survived into some sibling pool is exactly the gold an entity probe can re-find)
   and **understates** the k-arm's disadvantage on a full noisy store. The true
   128K store is ~5-15% larger and much noisier; on it, both arms would recover
   less, and the exclusive fraction could move either way.
3. **19% of missing gold has no accessible content** (never entered any sibling
   pool) and is scored unreachable — a floor artifact, not a measured failure.
4. **Small per-type counts** (temporal n=5, knowledge_update n=5) — those rows are
   directional only.
5. Only B=30 is usable; the budget-scaling curve (does exclusive fraction hold as B
   grows on a *real* store?) is unanswerable locally.

## Verdict

**Pre-registered outcome: BET LIVES (with a strong asterisk).** On the
retrieval-bound BEAM abilities at the only valid budget (B=30), the
exclusive-probe-reachable fraction is **22.8%** (31/136 accessible missing gold),
above the 15% live threshold; pooled 21.6%. event_ordering (27.0%) and
multi_session_reasoning (18.2%) — the two most retrieval-bound abilities — both
clear the bar individually.

The interrogator therefore has room to exist: ~1 in 5 missing gold units on the
retrieval-bound types is recoverable by a pool-derived probe and NOT by simply
deepening the original query's `k`. But two caveats temper this to a
"proceed-with-verification", not a green light:

- The exclusive signal is **entirely entity + adjacency probes**; the semantically
  ambitious chain/temporal/decomposition probes added nothing. A minimal
  interrogator = "entity-refocus + order-adjacency expansion", which is closer to
  query expansion than to the "detect broken update chains / missing comparison
  sides" framing in the bet.
- The 22.8% is measured on a store biased in the interrogator's favor
  (union-of-retrieved-pools); on the true 128K history it is a soft upper bound.

**Recommended gate before building:** re-run this exact harness on the real 128K
BEAM histories and on LME multi-session/temporal (both require downloading the
source data — the single blocking dependency). If entity+adjacency exclusive
reachability survives >=15% on the *true* store, build the interrogator; if it
collapses toward the 5% floor once the store is full and noisy, the effect was
store-reconstruction bias and the bet dies.
