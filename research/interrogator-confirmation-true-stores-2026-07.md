# Interrogator confirmation gate on TRUE stores (2026-07-14)

Re-run of the frozen, pre-registered kill-test
(`research/interrogator-reachability-2026-07.md`) for the "compiler as
interrogator" bet, replacing yesterday's biased reconstructed BEAM store with
the true 128K histories, and adding the previously untestable LongMemEval-S
half. This is the confirmation gate defined in CONTEXT.md ("Direction decision
2026-07-14 (evening)").

Pre-registered read (unmoved): exclusive-probe fraction (B) >= 15% of
accessible missing gold on retrieval-bound types => bet CONFIRMED on that
benchmark; < 5% => bet DIES; between => judgment call reported faithfully.
Verdicts are per-benchmark, never pooled.

Code: frozen harness `analysis/interrogator_{bm25,probes,reachability,report}.py`
(untouched, byte-identical probe generators; `test_probe_no_gold_leakage`
green) + new store loaders `analysis/interrogator_beam_store.py`,
`analysis/interrogator_confirm_beam.py`, `analysis/interrogator_lme_store.py`,
`analysis/interrogator_lme.py`. Tests:
`tests/analysis/test_interrogator_confirm.py` (11 new; full suite 397 pass).
Deterministic, local, no network, no LLM, $0. Run:
`python -m analysis.interrogator_confirm_beam` and
`python -m analysis.interrogator_lme`.

## Data inventory (all local, verified)

**BEAM true 128K store:**
`data/beam_source/data/100K-00000-of-00001.parquet` — 20 rows, one per
conversation (`conversation_id` "1".."20"), `chat` = 3 batches of messages
with global integer ids, roles, `question_type`, `time_anchor`. Despite the
"100K" filename this is the 128K tier: BEAM has no 100K tier, the 20
conversations match the 128K count, and every slice unit matches the chat
content verbatim (below). True store sizes: 78, 78, 78, 84, 84, then 115 x 15
turns (median 115; yesterday's reconstruction recovered 61-101, median 93).

**BEAM slice with gold labels:**
`/Users/vein/Documents/research/memory-eligibility-feasibility/dataset-slices/beam_transfer_bm25_top20_v0.jsonl`
(280 rows = 7 abilities x 40; gold = `evidence_sufficiency.answer_bearing_memory_ids`).

**LongMemEval-S official haystack:**
`data/longmemeval_source/longmemeval_s` — 500 questions, 39-66 haystack
sessions each (median 50), `answer_session_ids` gold, all present in
`haystack_session_ids` (0 missing). The sieve slice
`dataset-slices/longmemeval_bm25_top20_v1.jsonl` is unusable for missing-gold
analysis: 498/500 pools contain an injected `answer_` unit by construction.
This run rebuilds pools honestly from the haystack.

### BEAM turnization and alignment (verified before trusting)

The slice's `turn_NNNN` ids are NOT `message_id // 2`. A turn is a
**main_question-anchored group**: each user message with
`question_type == "main_question"` opens a turn; all following messages
(assistant replies, `answer_ai_question` user follow-ups, `followup_question`)
belong to it until the next anchor. Turn numbers are **global, 1-indexed,
continuous across batches** (chat 1 batch 2 starts at turn 27 after batch 1's
26 anchors). Rendered content is
`[{time_anchor} | ]batch {b} | turn {t} | {role}[{msg_id}]: {content} ...`
with newlines replaced by single spaces; the time-anchor prefix appears only
on turns whose anchor message carries one.

Alignment verification: **all 5,600 slice candidate units across 280 rows
match the parquet-derived rendering verbatim** (exact string equality), and
**all 511 gold turn ids exist in the true stores** — so `content_absent = 0`
(yesterday: 35/183 = 19.1% of missing gold was unreachable by construction).
`verify_alignment()` runs inside `run_true()` and fails loud on any mismatch.

### LME unit granularity (documented decision)

The original sieve slice used **session-level** units (memory_id = session id,
whole-session content). At session granularity the store is 39-66 sessions and
the top-20 pool already occupies 30-51% of it: even B=10 exceeds 50% of the
beyond-pool store on the smallest haystacks, and B=19 vacuums the entire
remainder of a 39-session store. Session granularity cannot host a meaningful
matched-budget test, so per the pre-registered escape hatch we drop to
**message granularity**: one unit per non-empty message, rendered
`{date} | {role}: {content}` (session id deliberately excluded from the text so
no id-string retrieval channel exists). Stores are then 396-616 units (median
492); unit ids are position-based (`s{pos:03d}_m{msg:03d}`) because 15
questions carry duplicate session ids. Gold accounting stays **session-level**
(matching `answer_session_ids` semantics): a missing gold session counts as
recovered by an arm iff any of its messages is acquired by that arm.

Pool = top-20 messages by the same Okapi BM25 (k1=1.5, b=0.75, same tokenizer)
for the original question, **no gold injection**. Adjacency probe adapted as
mandated: sessions ordered by parsed `haystack_dates` (position tiebreak;
289/500 haystacks are not stored chronologically, so parsing is required);
candidates = all messages of sessions within +/-2 date-order ranks of any
pooled session, capped at B — the direct analogue of BEAM's turn_index +/-2.
Abstention rows (`*_abs`, 30 questions) are excluded from the primary numbers:
their `answer_session_ids` point at sessions that deliberately do NOT contain
the queried fact, so they are not gold (with-abs sensitivity reported below).

## Matched-budget protocol (frozen)

Identical to yesterday: at budget B, k-arm = original query BM25 next-B beyond
the row's pool; probe-arm = P deterministic probes (entity, chain, temporal,
decomposition — byte-identical generators, pool+query only) x top-m, deduped,
capped at B, plus the adjacency channel capped at B. Both arms exclude the
row's own pool ids and rank over the same full store. Budget specs (B, P, m):
(30, 5, 6), (80, 10, 8), (180, 15, 12).

## Saturation analysis

- **BEAM true store (78-115 turns):** B=30 saturates 0/99 rows (B < 50% of
  every store) — the valid budget. B=80 saturates **99/99** rows (80 >= 50% of
  even the largest 115-turn store): the k-arm trivially vacuums the store, so
  B=80 is reported only as a saturation-caveat check. B=180 exceeds every
  store outright and is not run.
- **LME message store (396-616 units):** 0/44 rows saturated at ALL of B=30,
  B=80, B=180. The largest budget keeping saturation at 0 rows is **B=180**
  (30-45% of store). B=30 is the matched-comparison budget vs BEAM; B=180 is
  the deepest honest aperture probe.

## PART A — BEAM on the true store

A/B/C at B=30 (the valid budget; content_absent = 0 everywhere):

| Cohort | rows | missing=accessible | A k-reach | **B probe-excl** | both | k-excl | C unreach | **B / accessible** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| POOLED (6 answerable abilities) | 99 | 183 | 98 | **40** | 95 | 3 | 45 | **21.9%** |
| RETRIEVAL-BOUND (event_ordering, multi_session, temporal, contradiction) | 83 | 167 | 86 | **40** | 83 | 3 | 41 | **24.0%** |

Per ability (B=30, true store):

| Ability | rows | accessible | A k-reach | **B probe-excl** | C unreach | **B / accessible** |
|---|---:|---:|---:|---:|---:|---:|
| event_ordering | 38 | 108 | 50 | **30** | 28 | **27.8%** |
| multi_session_reasoning | 24 | 38 | 20 | **8** | 10 | **21.1%** |
| temporal_reasoning | 7 | 7 | 5 | **1** | 1 | 14.3% |
| contradiction_resolution | 14 | 14 | 11 | **1** | 2 | 7.1% |
| knowledge_update | 5 | 5 | 4 | **0** | 1 | 0.0% |
| information_extraction | 11 | 11 | 8 | **0** | 3 | 0.0% |

B=80 (saturated 99/99 — void as a k-vs-probe comparison, reported per
mandate): RB probe-exclusive drops to 9/167 = 5.4%, k-reach 157/167, C unreach
1. Under full saturation the k-arm recovers nearly everything by construction.

Reverse exclusivity (B=30): k-exclusive is 3/167 on retrieval-bound — probes
still nearly dominate k on what k finds. Absolute k-reachability is 86/167 =
**51.5%** of missing gold (reconstructed store: 61.8%): on the true, noisier
store the original query's deep ranks recover LESS, i.e. the pre-filtered
reconstruction had been flattering the k-arm more than the probe arm.

### Reconstruction bias quantified (yesterday vs today, B=30)

| Quantity | Reconstructed store (2026-07-13) | True store (today) | Delta |
|---|---:|---:|---:|
| Store size (median) | 93 turns (~85% of true) | 115 turns | +24% units |
| content_absent | 35/183 (19.1%) | **0/183** | bias floor removed |
| Pooled B / accessible | 32/148 = 21.6% | 40/183 = 21.9% | +0.3pp |
| RB B / accessible | 31/136 = **22.8%** | 40/167 = **24.0%** | **+1.2pp** |
| RB B / ALL missing gold (common denominator) | 31/167 = 18.6% | 40/167 = 24.0% | +5.4pp |
| RB k-reach / accessible | 84/136 = 61.8% | 86/167 = 51.5% | -10.3pp |
| RB k-exclusive | 3/136 | 3/167 | ~0 |
| event_ordering excl | 27.0% | 27.8% | +0.8pp |
| multi_session excl | 18.2% | 21.1% | +2.9pp |
| contradiction excl | 0.0% | 7.1% | +7.1pp |

The feared direction of the bias did not materialize. The
union-of-sibling-pools store was expected to inflate probe reachability; in
fact it deflated it: the 35 content-absent gold turns became accessible on the
true store and probes claimed a share of them, while the added noise hurt the
k-arm (61.8% -> 51.5%) more than the probe arm. Yesterday's 22.8% was a mild
UNDER-estimate, not reconstruction inflation.

### Probe-type ablation on the true store (B=30, recovered-by-type)

Retrieval-bound: **entity 84, adjacency 39; chain 0, temporal 0,
decomposition 0** (pooled-all: entity 96, adjacency 39). Identical structure
to the reconstructed store (entity 83, adjacency 29): the exclusive signal is
still carried entirely by entity-refocus + order-adjacency probes. This was
NOT store bias — on the complete, noisy history the semantically ambitious
chain/temporal/decomposition probes still contribute nothing at B=30 (chain
probes only start recovering units at the saturated B=80, where they are
redundant with k). Adjacency grew the most on the true store (29 -> 39),
consistent with the reconstruction having been missing exactly the
never-retrieved neighbor turns adjacency needs.

**BEAM verdict vs pre-registered thresholds: CONFIRMED.** RB exclusive
fraction 24.0% >= 15% at the only valid budget, pooled 21.9%; event_ordering
(27.8%) and multi_session_reasoning (21.1%) clear the bar individually.

## PART B — LongMemEval-S (honest pools, first run ever)

Scope finding first: with honest message-level BM25 top-20 pools, missing gold
is RARE on LME. Of 470 answerable questions (890 gold sessions), only **44
rows (9.4%) miss any gold session**, 60 missing sessions total = 6.7% of all
gold (RB types: 39/248 rows, 55/596 gold sessions = 9.2%). The published
pools' `answer_` injection was masking a retrieval problem that mostly does
not exist at this granularity — 90% of LME questions have every answer session
represented in an honest top-20.

A/B/C (abstention rows excluded; content_absent = 0 by construction):

| Budget | saturation | cohort | rows | missing | A k-reach | **B probe-excl** | both | k-excl | C unreach | **B / accessible** |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B=30 | 0/44 | POOLED | 44 | 60 | 34 | **6** | 29 | 5 | 20 | **10.0%** |
| B=30 | 0/44 | RETRIEVAL-BOUND | 39 | 55 | 31 | **6** | 26 | 5 | 18 | **10.9%** |
| B=80 | 0/44 | RETRIEVAL-BOUND | 39 | 55 | 50 | **4** | 46 | 4 | 1 | 7.3% |
| B=180 | 0/44 | RETRIEVAL-BOUND | 39 | 55 | **55** | **0** | 55 | 0 | 0 | **0.0%** |

Per type at B=30: multi-session 4/32 = 12.5%, temporal-reasoning 2/23 = 8.7%
(single-session-preference 0/4, knowledge-update 0/1, single-session-user 0/1).

Sensitivity with `_abs` rows included (6 extra rows whose "gold" sessions
contain no answer by design): B=30 RB 8/60 = 13.3%, pooled 12.1%; B=80 6.7%;
B=180 0.0%. Either way the read is the same band.

Probe-type ablation (B=30, RB, recovered-by-type): **entity 27, adjacency 5**;
chain/temporal/decomposition 0 — the same two-family structure as BEAM.
(chain/temporal only appear at B>=80 where they are k-redundant.)

The budget-scaling curve — the question yesterday's writeup called
"unanswerable locally" — is now answered on LME, on a store big enough that
NO budget saturates: the exclusive fraction **decays monotonically to exactly
zero** (10.9% -> 7.3% -> 0.0%). At B=180 the original query's own ranking
recovers **100%** (55/55) of missing gold sessions. On LME, everything a probe
can reach, a deeper k reaches too; aperture alone suffices. Reverse
exclusivity is also materially worse than BEAM: k-exclusive 5/55 = 9.1% at
B=30 (BEAM 3/167 = 1.8%) — probes do not even dominate k on what k finds.

**LME verdict vs pre-registered thresholds: JUDGMENT CALL (10.9% at matched
B=30; 13.3% with abs rows), reported faithfully — and the judgment reads
NEGATIVE.** The fraction sits in the 5-15% band, so the bet does not
mechanically die on LME, but three facts argue against the interrogator here:
(1) below the 15% confirm bar at the matched budget; (2) the exclusive residue
vanishes entirely at deeper honest apertures (0/55 at B=180, zero saturation)
— on LME the interrogator is provably "mere aperture"; (3) the addressable
surface is tiny — 6 exclusive sessions across 470 questions = the interrogator
could matter on at most ~1.3% of LME rows.

## Honest limitations

1. **Probe-arm budget accounting (frozen-protocol artifact).** As in
   yesterday's run, the text-probe channel and the adjacency channel are each
   capped at B separately, so the probe arm can acquire up to 2B units vs the
   k-arm's B. Kept byte-identical for comparability; adjacency carries 39/123
   RB probe recoveries on BEAM, so a strict shared-B variant would lower the
   probe numbers somewhat on both stores. This inflates both benchmarks
   equally and cannot explain the BEAM-vs-LME asymmetry.
2. **Small LME missing-gold population.** 39 RB rows / 55 missing sessions —
   the 10.9% is 6 sessions; +/-2 sessions moves it 3.6pp. The BEAM cohort
   (167) is the statistically meaningful one.
3. **BEAM B=80 remains uninterpretable** (99/99 saturated; store is only
   78-115 turns). The BEAM budget-scaling question is still open; only LME
   answers it, and answers it against the bet.
4. **Session-level gold accounting on LME** credits an arm for recovering ANY
   message of an answer session; a single acquired message may not contain the
   answer-bearing span. This is generous to BOTH arms symmetrically.
5. **Small per-type counts** on BEAM temporal (n=7), knowledge_update (n=5),
   contradiction (n=14) — directional only. The confirmed signal rests on
   event_ordering (108) and multi_session (38).
6. **Probe attribution on LME** credits the lexicographically-first acquired
   unit's probe type per recovered session (deterministic; matters only for
   the ablation split, not the exclusive counts).

## Verdicts

- **BEAM: CONFIRMED.** RB exclusive-probe fraction **24.0%** (40/167) at the
  valid budget B=30 on the true 128K store, >= the 15% bar; content_absent 0;
  yesterday's 22.8% was NOT reconstruction bias (true-store number is
  higher). The exclusive signal remains entirely entity-refocus +
  order-adjacency; chain/temporal/decomposition are zero on the true store
  too, so the honest v0 mechanism is still structured query expansion +
  adjacency, not broken-chain detection.
- **LongMemEval-S: JUDGMENT CALL, leaning kill.** RB exclusive fraction
  **10.9%** (6/55) at matched B=30 — between the pre-registered bounds, so
  reported as such — but the residue decays to **0.0%** at B=180 with zero
  budget saturation: on LME every missing gold session is reachable by simply
  deepening the original query's k to ~200 messages, and missing gold affects
  only 9.4% of rows to begin with.

Combined read for the bet: the interrogator has a real, confirmed niche on
BEAM-style single-chat histories (dense entities, ordered turns, gold
scattered beyond the pool's aperture), and no defensible niche on
LME-style multi-session haystacks, where honest BM25 pools rarely miss gold
and a larger k recovers all of it. Any method building that proceeds should
scope the interrogator's claims to the former regime and use LME as the
negative control, not a target.
