# Interrogator v0 conversion experiment (2026-07-14)

Does probe-exclusive reachability (confirmed offline at **24.0%** on BEAM
retrieval-bound abilities, true 128K store, frozen harness) **convert** into
reader accuracy? This is the method experiment with real reader calls that the
confirmation gate authorized. Scope is bound to the BEAM single-dense-chat
regime only (LME is the negative control and is NOT run here — see the
CONFIRMATION-GATE RESULT in CONTEXT.md).

Driver: `analysis/interrogator_conversion.py` (reuses the frozen
`interrogator_{bm25,probes,beam_store}` modules byte-for-byte; no fork). Reader
calls via `reader.client.generate_answer` (OpenRouter, fallbacks disabled,
provider recorded). Judge via `cli/judge.py` (generic 3-tier rubric,
`deepseek/deepseek-chat-v3-0324`), repo convention. Run dirs under
`results/answer_generation_runs/interrogator_v0_conversion/`.

## Cohort (pre-registered, frozen before any reader call)

- Slice: `beam_transfer_bm25_top20_v0.jsonl` (280 rows = 7 abilities x 40).
- True store: `data/beam_source/data/100K-00000-of-00001.parquet` (the 128K
  tier). Alignment verified: all slice units match store rendering verbatim,
  all gold ids present (`content_absent = 0`).
- Evaluation cohort = the four **retrieval-bound** abilities only
  (`event_ordering`, `multi_session_reasoning`, `temporal_reasoning`,
  `contradiction_resolution`) = **160 rows** total.
  - **Missing-gold cohort (primary): 83 rows** whose top-20 pool omits >=1 gold
    turn (event_ordering 38, multi_session 24, contradiction 14, temporal 7).
  - **Do-no-harm cohort: 77 rows** whose pool already contains all gold
    (temporal 33, contradiction 26, multi_session 16, event_ordering 2).
- Cap: 160 rows total, no further subsampling needed (already <= the ~160 cap).

## Conditions (all share the one reader prompt template)

Evidence rendered as raw turns via `v2/replay_io.py:SHARED_READER_TEMPLATE`.
Acquired units are appended to the pool as raw turns. Because BEAM turns are
huge (median ~1,300 reader tokens, up to ~6,000) and the top-20 pool alone is
~19K-63K tokens, a matched **total-evidence-token budget** is enforced at
**E = 24,000 tokens** (fits both readers' 128K context with wide margin and
holds cost < $10). Assembly order is identical across all four conditions:
**[acquired units first] then [pool units by BM25 rank]**, included
whole-unit until E is reached, then the final unit is truncated at E. Condition
1 has no acquired units, so it is pool-only capped at E. This makes the contrast
*what fills the evidence budget*, not *how much* — total tokens are matched at E
across conditions 2-4 and the same cap governs condition 1.

Budget for acquisition: **B = 30 candidate units** (the validated budget). Each
condition proposes up to B acquisition units; whichever survive the E-token cap
after being placed first is what the reader sees.

1. **fixed** — top-20 pool only, capped at E (status-quo baseline).
2. **adaptive_k** — pool + next-B units by **original-query BM25** over the true
   store (the aperture baseline the interrogator must beat).
3. **interrogator** — pool + up to B units from the frozen entity-refocus +
   order-adjacency probe generators (`generate_probes` + adjacency channel,
   byte-identical to `interrogator_reachability.py`; deterministic, no LLM in the
   v0 acquisition loop). Chain/temporal/decomposition probes contributed zero
   twice offline and are excluded from the mechanism framing but the frozen
   `generate_probes` is used unchanged (they simply never win a slot at B=30).
4. **ceiling** — pool + ALL missing gold turns injected (retrieval oracle upper
   bound: what perfect acquisition would buy).

## Readers (OpenRouter, fallbacks disabled, provider recorded)

- Small: `meta-llama/llama-3.1-8b-instruct`.
- Strong: `qwen/qwen-2.5-72b-instruct` (the strong reader with the most
  precedent in this repo's cross-benchmark answer-generation runs).

Parallelism 32, temperature 0, seed 42, max_tokens 64.

## Pre-registered read (committed BEFORE looking at scored results)

- **Primary metric:** judge accuracy (exact-rate, score==2) on the
  **missing-gold cohort** (83 rows).
- **Interrogator LIVES iff:** it beats `adaptive_k` on the missing-gold cohort
  by **>= 3pp on at least one reader** AND does not damage the do-no-harm cohort
  by more than **1pp** (do-no-harm guard), on the same reader.
- **Interrogator dies (kill):** if within noise of `adaptive_k` (< 3pp on both
  readers) the mechanism does not convert. Report as a kill.
- **Ceiling context:** report how much headroom even exists to fight over =
  `ceiling` minus `fixed` on the missing-gold cohort. If the ceiling itself is
  **< 5pp**, the whole acquisition axis is second-order for readers and *that*
  is the finding (regardless of the interrogator-vs-adaptive-k contrast).

## Cost budget

Pre-registered estimate: 160 rows x 4 conditions x 2 readers = 1,280 reader
calls at E~24K prompt tokens. Est. Llama $0.31 + Qwen-72B $5.60 + judge $0.33 =
**~$6.24**. Cap $10. Stratified row-cut plan if over budget: drop
do-no-harm rows first (they are the secondary cohort), keeping all 83
missing-gold rows.

---

## OFFLINE PRE-CHECK (deterministic, before reader calls)

Before any reader call, I checked whether the offline 24% *turn-level*
probe-exclusivity even reaches the reader as *row-level* answerability. It does
not, and this is the decisive mechanism finding:

- With the **full uncapped B=30 acquisition** (no token truncation at all), on
  the 83 missing-gold rows the interrogator recovers gold that adaptive-k misses
  on exactly **6 rows**, and adaptive-k recovers gold the interrogator misses on
  **6 rows** — symmetric, net zero at the row level.
- Under the matched **E=24K** cap actually sent to readers, the split is
  **7 / 7** — still a wash. The evidence *set* differs on 70/83 rows (different
  non-gold turns get injected), but the *gold* outcome is identical.
- Why the offline 24% does not convert: the offline metric counted
  probe-exclusive **gold turns** (40/167), but those turns cluster on rows where
  k already recovers *other* gold for the same question. At the row-outcome
  granularity that a reader sees, probe-exclusive turns almost never land on a
  row that k left completely goldless. Turn-level exclusivity ≠ row-level
  answerability advantage.
- Acquisition cost sanity: interrogator's full B=30 acquisition is a median
  **41.6K tokens** (p90 60K, max 74K) — infeasible in any single reader context,
  which is *why* the matched E-cap is mandatory and not a design choice that
  disfavors the interrogator.

The reader run below was executed anyway (the pre-registration commits to reader
accuracy, and the do-no-harm / ceiling numbers are required deliverables).

## RESULTS (reader accuracy, judge exact-rate score==2)

640 calls per reader (160 rows x 4 conditions). All 429 rate-limit failures
resolved by lower-parallelism resume; **1 Qwen row**
(`beam-128k-5-temporal_reasoning-001`) exceeds the DeepInfra Qwen-72B served
32,767-token context on the interrogator condition and is dropped from the Qwen
cohort in **all four** conditions to keep the paired comparison balanced (Qwen
n = 159; Llama n = 160). Protocol deviation logged below.

### meta-llama/llama-3.1-8b-instruct (small)

| condition | missing-gold acc (n=83) | do-no-harm acc (n=77) | all (n=160) |
|---|---:|---:|---:|
| fixed | 1.2% | 1.3% | 1.2% |
| adaptive_k | 3.6% | 3.9% | 3.8% |
| interrogator | 3.6% | 2.6% | 3.1% |
| ceiling | 2.4% | 5.2% | 3.8% |

- interrogator vs adaptive_k (missing-gold): **+0.0pp** (paired: 2 rescues / 2
  damages, n=83).
- do-no-harm damage vs fixed: interrogator −1.3pp (within the 1pp guard band; it
  is actually slightly *above* fixed on dnh).
- ceiling headroom (ceiling − fixed, missing-gold): **+1.2pp** (paired: 1 rescue
  / 0 damages).

### qwen/qwen-2.5-72b-instruct (strong)

| condition | missing-gold acc (n=83) | do-no-harm acc (n=76) | all (n=159) |
|---|---:|---:|---:|
| fixed | 3.6% | 9.2% | 6.3% |
| adaptive_k | 1.2% | 7.9% | 4.4% |
| interrogator | 1.2% | 5.3% | 3.1% |
| ceiling | 3.6% | 11.8% | 7.5% |

- interrogator vs adaptive_k (missing-gold): **+0.0pp** (paired: 0 rescues / 0
  damages).
- do-no-harm damage vs fixed: interrogator **−3.9pp** (VIOLATES the do-no-harm
  guard — acquisition evicts gold-bearing pool turns under the matched budget).
- ceiling headroom (ceiling − fixed, missing-gold): **+0.0pp** (paired: 2
  rescues / 2 damages — net zero even for the retrieval oracle).

### Per-ability (missing-gold cohort, exact-rate)

event_ordering (n=38) and multi_session_reasoning (n=24) — where the offline
effect concentrated — are **0% across every condition and both readers**,
including the ceiling. The tiny non-zero cells are contradiction_resolution
(n=14) and temporal_reasoning (n=7), which are directional at that sample size.
The offline signal's home abilities produce zero reader accuracy even when the
answer-bearing turns are oracle-injected: reconstructing a 5-item ordering from
scattered turns is a reader-reasoning failure, not a retrieval-completeness one.

## PRE-REGISTERED VERDICT

- **Small (Llama-3.1-8B): DOES NOT CONVERT (kill).** interrogator = adaptive_k
  on the primary metric (+0.0pp; needed ≥3pp). Do-no-harm guard technically
  passes but is moot given zero benefit.
- **Strong (Qwen-2.5-72B): DOES NOT CONVERT (kill).** interrogator = adaptive_k
  (+0.0pp) AND violates do-no-harm (−3.9pp vs fixed).
- **Both readers: kill.** The interrogator v0 mechanism does not convert its
  confirmed offline turn-level probe-exclusivity into reader accuracy.

## THE CEILING IS THE HEADLINE

Per the pre-registration, when the interrogator is within noise of adaptive-k we
report how much ceiling even exists to fight over. The retrieval-oracle ceiling
(ceiling − fixed on the missing-gold cohort) is **+1.2pp (Llama)** and
**+0.0pp (Qwen)** — both **< 5pp**. Perfect acquisition of every missing gold
turn buys essentially nothing on this cohort. This is the finding: **on BEAM's
retrieval-bound abilities the acquisition axis is second-order for readers.**
The bottleneck on event_ordering / multi_session is reader reasoning over
present evidence, not evidence presence — consistent with CONTEXT.md's prior
that all systems score 0.00–0.08 on these abilities regardless of retrieval.

The confirmation gate measured the right quantity (probe-exclusive reachability)
but at the wrong granularity for the downstream claim: turn-level exclusivity
was real (24.0%), but it neither survives the matched-budget context cap nor
sits on rows whose *answer* flips, and the underlying abilities are
reasoning-bound, not retrieval-bound, at the reader.

## COST ACCOUNTING (per query, matched E=24K)

| reader | condition | evidence tok | prompt tok | units acquired | $/query |
|---|---|---:|---:|---:|---:|
| Llama-8B | fixed | 23,635 | 20,998 | 0 | $0.00042 |
| Llama-8B | adaptive_k | 24,010 | 21,841 | 30 | $0.00044 |
| Llama-8B | interrogator | 24,010 | 21,911 | 30 | $0.00044 |
| Llama-8B | ceiling | 23,706 | 21,095 | 1 | $0.00042 |
| Qwen-72B | fixed | 23,632 | 21,404 | 0 | $0.00772 |
| Qwen-72B | adaptive_k | 24,010 | 22,313 | 30 | $0.00805 |
| Qwen-72B | interrogator | 24,010 | 22,366 | 30 | $0.00806 |
| Qwen-72B | ceiling | 23,704 | 21,508 | 1 | $0.00776 |

Read-time cost note: because total evidence tokens are matched at E, acquisition
adds only the marginal prompt-token cost of re-ranking overhead, not extra
context — the $/query delta between fixed and interrogator is <$0.0004 on both
readers. The interrogator's cost argument is therefore not "it sends more"; it
is "it substitutes *what* it sends," and that substitution did not pay.

**Total spend (both readers + deepseek judge, estimated from token usage):
~$5.57** (well under the $10 cap).

## ABSOLUTE SCORES — CAVEAT

These judge exact-rates (1–9% on the retrieval-bound cohort) are **not
comparable to the Honcho/BEAM leaderboard**: different readers (Llama-8B /
Qwen-72B vs the leaderboard's readers), different judge (deepseek-chat-v3-0324,
3-tier generic rubric), a matched-E truncated single-chat evidence format, and a
deliberately adversarial subset (only the 4 retrieval-bound abilities, over half
of them event_ordering/multi_session which are near-zero for all systems). They
are internally consistent for the paired within-cohort contrasts, which is all
the verdict rests on.

## PROTOCOL DEVIATIONS

1. **Matched budget operationalized in tokens, not units.** B=30 *units* of BEAM
   turns is a median 41.6K acquisition tokens, infeasible in one context. Per
   the pre-registration's truncation clause, evidence is capped at **E=24,000
   tokens** identically across all conditions, acquired-units-first. This is
   faithful to "matched total-evidence-token budgets" but means only the first
   ~10-11 acquired units survive on dense rows (same for adaptive_k and
   interrogator, so the contrast is fair).
2. **One Qwen row dropped** (`beam-128k-5-temporal_reasoning-001`) from all Qwen
   conditions: its interrogator rendering tokenizes to 33,137 tokens, exceeding
   the served 32,767 context. Char/4 estimate (24K) under-counted this dense
   row. Qwen n=159; other 3 Qwen conditions succeeded on it. Impact: negligible
   and symmetric.
3. **max_tokens raised 64 -> 128** (identical across all conditions/readers) so
   event_ordering list-type answers are not truncated mid-list. Qwen still hits
   finish_reason=length on some rows (verbose non-answers), but identically
   across conditions.
4. **429 rate-limits** on the first pass (allow_fallbacks=False + parallelism 32
   saturated the endpoint). Resolved by resume passes at parallelism 8/4/2; all
   but the 1 context-length row recovered. No mocked results.

## CODE

- `analysis/interrogator_conversion.py` — deterministic evidence assembly
  (reuses frozen `interrogator_{bm25,probes,beam_store,reachability}` modules).
- `cli/interrogator_conversion_run.py` — reader driver (+ `--resume`).
- `analysis/interrogator_conversion_report.py` — scoring against the
  pre-registered read.
- `tests/analysis/test_interrogator_conversion.py` — 7 tests (assembly, cap,
  matched-budget, cohort, determinism).
- Runs: `results/answer_generation_runs/interrogator_v0_conversion/<condition>__<reader>/`.
