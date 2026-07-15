# BEAM reader-free diagnostics: gold survival + supersession structure (2026-07-14)

$0 analyses over `beam_transfer_bm25_top20_v0.jsonl` (predecessor repo; 280
rows = 7 abilities x 40, gold `answer_bearing_memory_ids` per row). No LLM
calls; deterministic v2 packagers run locally at budget 400.

## 1. Gold-unit survival through deterministic packagers

Survival = fraction of gold answer-bearing units (that are present in the
BM25 top-20 pool) whose content survives into the compiled pack. Measured by
80-char prefix containment: exact for filtered_raw (keeps/drops whole
units); *understates* extractive (which keeps sentences, so a unit can
survive via a non-prefix sentence).

| ability | n (gold in pool) | filtered_raw@400 | extractive@400 | full survival fr / ex |
|---|---:|---:|---:|---:|
| information_extraction | 29 | 37.9 | 69.0 | 38% / 69% |
| knowledge_update | 34 | 26.5 | 79.4 | 26% / 79% |
| temporal_reasoning | 40 | 33.8 | 90.0 | 12% / 82% |
| contradiction_resolution | 40 | 41.2 | 88.8 | 18% / 80% |
| event_ordering | 35 | 10.0 | 25.6 | 3% / 14% |
| multi_session_reasoning | 38 | 33.3 | 65.4 | 16% / 55% |
| abstention | 0 | — | — | (no gold by design) |

Findings:

1. **filtered_raw is an evidence destroyer on BEAM**: it deletes 60-90% of
   gold units. Stem-overlap relevance is a poor proxy for answer-bearing on
   conversational data. Third independent strike against "filtered raw is
   the safe action" (after the MuSiQue 19.6 result and the ladder
   retraction).
2. **Extractive is by far the strongest deterministic selector** (65-90%
   survival on the five compilation-addressable abilities) — consistent
   with extractive being the most common per-query oracle winner in the
   pilot decomposition. Pack construction (incl. supersession chains)
   should build on extractive-selected content, not filtered units.
3. **event_ordering fails for both** (10/26%) — matches the
   retrieval-completeness diagnosis (5% complete pools): the evidence is
   spread over more units than any 400-token pack can hold. Aperture
   problem, not selection.
4. **L3's concrete bar**: the learned per-unit scorer must beat *extractive
   sentence selection* on gold recall at equal budget, not merely BM25 rank
   truncation. If it cannot, L3 is unnecessary and extractive is the
   selection op.

## 2. Supersession structure (same file, contradiction + knowledge_update)

For gold units with a same-topic rival in the pool (token Jaccard > 0.35):

| ability | gold units with rival | gold is LATEST version | rival outranks gold in BM25 |
|---|---:|---:|---:|
| contradiction_resolution | 16/63 (25%) | 81% | 31% |
| knowledge_update | 20/34 (59%) | **35%** | 30% |

- ~30% BM25 rank inversions (stale rival ranked above gold) corroborate the
  published staleness hazard (MemStrata's AUROC-0.59 embedding result;
  see theory doc supersession-cluster table) in the BM25 setting.
- **Resolve-to-latest is the wrong fix for knowledge_update**: gold is the
  latest version only 35% of the time (questions need prior values or the
  update event). The supersession packager must render CHAINS
  ("X at t1 -> Y at t2", all values preserved with provenance), not bind a
  single current value. This is the empirical wedge against the entire
  resolve-to-current cluster (MemStrata / Supersede / max(serial)).
- Caveat: crude Jaccard rival detection, 16-20 gold units per ability —
  directional; re-check with real entity/attribute clustering once the
  supersession module exists.

Scripts: scratchpad `beam_gold_survival.py` + inline probes (session).
