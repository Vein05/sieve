# Proposal: from benchmark-fit compilation to utility-trained compilation (2026-07-14)

Status: proposal for discussion. Follows the reader-conditioning negative result
(`research/matrix-mining-reader-conditioning-2026-07.md`).

## The diagnosis: SIEVE v1 is an expert system for LongMemEval's taxonomy

How v1 actually works, from the code:

1. **Intent classifier** (`compiler/intent/query_features.py`, 524 lines):
   regex/marker-list query classification with hand vocabulary
   (`" altogether "`, `" on average "`, `_EXTREMUM_MARKERS`, ...).
2. **Planner** (`compiler/planner/planner.py`): a hand-built decision tree
   mapping query families to 8 hand-crafted schemas (state_update,
   temporal_interval, relative_time, current_state, aggregate, ordered_choice,
   comparison, direct_value). The families and schemas mirror LongMemEval's
   six question types nearly 1:1.
3. **Rule runtime** (`compiler/runtime/`, 3,056 lines; `bindings.py` alone is
   1,075): regex slot extraction, temporal normalization, numeric computation.
   40/500 queries are answered deterministically by the compiler with no
   reader call.
4. **Controller** (`configs/controller_v0.json`): 5 hand-weighted signals with
   magic thresholds (commit 0.55, contradiction 0.8, ...) tuned on LME.
5. **`configs/concept_specs.json`**: 29 concept specs whose patterns are
   literal LongMemEval instance content ("pride parade", "adoption agencies",
   "charity race"). Disabled in the frozen config, but it documents the
   development loop: fit the benchmark until it wins.

Total: ~21K lines of compiler/evidence/controller rules. The failure mode is
silent evidence destruction: when the classifier or schema misfires on
out-of-distribution queries, the renderer deletes answer-bearing content.

## The evidence: fixed compilation overfits its development distribution — in both directions

From `data/router_matrix/v0.parquet` (uncapped conditions):

| system | dev distribution | in-domain | transfer |
| --- | --- | --- | --- |
| SIEVE v1 (hand-crafted) | LongMemEval | +8pp avg over raw, 21 readers | **ConvoMem −25 to −35pp vs raw** (46-48% vs 71-79%); LoCoMo ~0; never runnable on HotpotQA/MuSiQue |
| RECOMP-abst (learned, fixed) | QA corpora | HotpotQA 68.6% | **LongMemEval 24.9%** |
| Generic query-focused summary | none | — | robust everywhere: HotpotQA 88% (raw 62%), MuSiQue 50% (raw 20%), ConvoMem 78-81%, LME 46-55% |

Three observations:

1. The ConvoMem collapse is the sharpest indictment: it is *also* a
   conversational-memory benchmark, so v1 is not even "conversational memory
   compilation" — it is LongMemEval compilation.
2. RECOMP is the mirror image: a *learned* but *fixed* compressor trained on
   QA collapses on LME exactly as the *hand-crafted* LME compiler collapses on
   ConvoMem. Fixed policy is the problem, not hand-craftedness per se.
3. The strongest simple baseline is generic query-focused abstractive summary.
   Any learned method must beat "always summarize + raw fallback", which is
   embarrassingly strong. Structure still pays where v1 was designed:
   on LME, structured beats summary for strong readers by 7-10pp
   (gpt-4.1-mini 53.4 vs 46.0; MiMo 55.6 vs 45.4; qwen3.6 55.0 vs 48.6).

What v1 got *right* mechanistically (worth preserving as learnable operations,
per the reinjection and strong-reader results): update/recency resolution,
temporal normalization, distractor removal, exact-quote preservation,
structure for weak readers.

## The proposal: learn the policy, keep the operations generic

Core shift: replace the hand-written pipeline with

```
pi(content, style, budget | q, C)      # query-conditioned, NOT reader-conditioned
```

trained on downstream reader utility across multiple distributions. Reader
conditioning is dropped per the matrix-mining negative; query conditioning is
where the realizable signal is (+2.5pp at question-type granularity, ~7pp
row-level headroom, +13.2pp per-query oracle on the v2 pilot surface).

Components:

1. **Utility-trained evidence scorer.** Cross-encoder over (query, candidate
   unit) predicting query-conditional utility. Labels from counterfactual
   reader outcomes (matrix rows, reinjection runs, cheap ablation runs on 1-2
   weak readers) — explicitly NOT from v1's selected/rejected labels, which
   would distill the heuristic we are trying to kill.
2. **Query-conditioned representation policy.** Small model (GBM/MLP over
   query + candidate-set features) choosing among ~4 *generic* packagers —
   filtered raw, extractive, query-focused summary, structured+quotes — plus
   budget. Trained on multi-benchmark matrix labels.
3. **Schema-free structuring.** Replace the 8 LME schemas + 3K-line rule
   runtime with one generic LLM restructuring operation (normalize dates,
   resolve updates, preserve exact quotes) that presumes no question taxonomy.
   The 21K-line v1 stack survives only as the frozen published baseline.
4. **Calibrated do-no-harm fallback.** The policy predicts when compilation is
   risky and passes raw through. Damage rate (broken raw-correct pairs)
   becomes a first-class reported metric.

## Evaluation protocol = the headline claim

**Leave-one-benchmark-out transfer.** Train on LongMemEval (+NQ), evaluate on
ConvoMem, LoCoMo, LoCoMo-Plus, HotpotQA, MuSiQue, BEAM — the exact axis where
v1 and RECOMP both collapse. Claims:

- never materially worse than raw on any held-out benchmark (damage control);
- retains most in-domain compression gains;
- beats always-summary and always-raw on the pooled accuracy-cost frontier;
- preserves reader scaling (replay the 20-reader panel; slope/ranking metrics
  from the published paper).

## Positioning and benchmark suite (2026-07-14 update)

Retire the "RAG compression" framing. The paper is **read-time memory
compilation for agents**: given a persistent conversation history and a query,
decide what evidence the reader sees and in what form. The contrast class is
the memory-systems literature (MemGPT/Letta, Mem0, SeCom, A-Mem, LIGHT), which
is almost entirely write-time (what to store/consolidate); none does
principled query-time representation choice. The compression literature
becomes baselines, not the frame.

Primary suite (5 conversational-memory benchmarks):

| Benchmark | Venue | Stresses | Local status |
| --- | --- | --- | --- |
| LongMemEval | ICLR 2025 | factual, temporal, updates, abstention | full (500-row slice, matrix) |
| LoCoMo | ACL 2024 | multi-hop, adversarial | slices in predecessor repo |
| ConvoMem | — | preference, implicit, changing facts | `convomem_bm25_top20_v1.jsonl` (500) |
| BEAM | ICLR 2026 | 100K-10M scale, 7 abilities, contradiction | `beam_adapted_bm25_top20_v2.jsonl` (280 = 7x40) + transfer file with gold labels |
| LoCoMo-Plus | ACL 2026 | implicit constraints, beyond-factual | build via github.com/xjtuleeyf/Locomo-Plus scripts (~430 cognitive instances) |

HotpotQA/MuSiQue/NQ remain as transfer controls with the pre-registered
criterion: match always-summary within CI (routing headroom there is only
2-4pp; there is nothing to win and nothing we need to win).

## Designing for BEAM

Assets: `beam_adapted_bm25_top20_v2.jsonl` (280 rows, 40 per ability:
information_extraction, knowledge_update, temporal_reasoning,
contradiction_resolution, event_ordering, multi_session_reasoning,
abstention). `beam_transfer_bm25_top20_v0.jsonl` carries
`evidence_sufficiency.answer_bearing_memory_ids` (gold evidence turns; 41
abstention rows labeled) and per-candidate `memory_usefulness_labels`.

1. **Step 0 retrieval-completeness diagnosis — DONE (2026-07-14).** Fraction
   of sufficient-labeled rows whose gold answer-bearing turns are in the BM25
   top-20 pool (n=239):

   | ability | complete | partial | fully missing |
   | --- | ---: | ---: | ---: |
   | contradiction_resolution | **65.0%** | 35.0% | 0.0% |
   | knowledge_update | 87.2% | 0.0% | 12.8% |
   | temporal_reasoning | 82.5% | 17.5% | 0.0% |
   | information_extraction | 72.5% | 0.0% | 27.5% |
   | multi_session_reasoning | 40.0% | 55.0% | 5.0% |
   | event_ordering | **5.0%** | 82.5% | 12.5% |

   Verdict: contradiction_resolution, knowledge_update, and temporal_reasoning
   are compilation-addressable (evidence is in the pool). event_ordering and
   largely multi_session_reasoning are retrieval-bound — report them as such
   (or address via aperture, not representation); do not let them dilute the
   compilation claim. This also reframes v1's reported BEAM deficit: per-
   ability retrieval bounds must be quoted alongside any compilation number.
2. **Target abilities: contradiction_resolution + knowledge_update** via a
   generic **supersession-aware packager**: cluster candidate units by
   entity/attribute, order temporally, and render explicit update chains
   ("stated X at t1; updated to Y at t2; current: Y") instead of silently
   binding one value. This is the exact inverse of the stale-value bug that
   drives 30% of ConvoMem damage, contains no benchmark vocabulary, and aims
   at BEAM's acknowledged open problem (all published systems: 0.00-0.08 on
   contradiction resolution).
3. **Abstention ability** connects to the starvation finding: when compiled
   evidence is insufficient, say what is missing (calibrated insufficiency
   note) rather than shipping a 14-token package that induces spurious
   abstention on answerable queries — and rather than answering on
   unanswerable ones.
4. The gold labels give a reader-free **selection-quality metric** (recall of
   answer-bearing units in the compiled pack) and seed labels for the utility
   scorer.

## Designing for LoCoMo-Plus

Construction detail that changes the design: cue-trigger pairs are
**semantically filtered so BM25/MPNet cannot link query to cue** (their s4.4).
Query-conditioned retrieval is broken by construction: RAG baselines score
12-16 vs 21-26 for full-context frontier models; memory systems (Mem0 15.8,
SeCom 14.9, A-Mem 17.2, all on GPT-4o) barely beat retrieval. A read-time
compiler over query-retrieved candidates inherits the miss — no packager can
render a cue that is not in the pool.

Design response: **two-channel evidence pack.**

- Channel A (existing): query-conditioned evidence from retrieval +
  representation policy.
- Channel B (new): a **standing-constraints digest** — durable user state,
  goals, values, causal constraints compiled once per user/history,
  query-independent, always included at small budget (~100-200 tokens). This
  is the read-time analogue of what Mem0-style systems store at write time,
  and it directly targets cue-trigger semantic disconnect: the constraint
  reaches the reader because it is standing, not because the query retrieved
  it.
- The policy learns the budget split between channels per query.

Evaluation adapter (M-sized): LoCoMo-Plus scores constraint consistency with
an LLM judge (1 / 0.5 / 0), response generation rather than short-answer QA —
our scoring pipeline needs their judge protocol integrated, and their data
build scripts run locally (generated files are gitignored upstream).

Honest expectation: absolute numbers will be low for everyone (ceiling 26).
The claim shape is "two-channel compilation recovers most of the
full-context cognitive-memory score at a fraction of the tokens, where both
plain RAG and write-time memory systems fail."

## Revised paper spine

1. **Impossibility (extended):** fixed compilation fails to transfer across
   readers (published paper) *and across data distributions* (new: the
   SIEVE→ConvoMem and RECOMP→LME collapses, quantified from the matrix).
2. **Construction:** a query-conditioned compiler trained on reader utility
   with generic operations, no benchmark taxonomy.
3. **Restoration:** cross-benchmark transfer + preserved reader scaling +
   Pareto dominance over fixed policies.

The reader-conditioning negative result becomes a supporting section: the
conditioning variable that matters is the query, not the reader.

## Kill criteria

- If the learned policy cannot beat "always query-focused summary with raw
  fallback" on held-out benchmarks, the method is unnecessary — publish the
  two-axis transfer failure as the finding.
- If per-unit utility labels are too noisy to train the scorer (check
  inter-reader label correlation ~0.38 first), fall back to policy-only
  (representation choice, no learned selection).

## Evidence update (2026-07-14, three parallel analyses)

**1. Pooled routing analysis (matrix, free).** Always-summary is the global
best fixed policy (63.2% row-weighted over 34,389 reader-query rows).
Domain-aware fixed adds only +0.9pp (all from structured on LME). Per-query
routing headroom over domain-aware fixed is +6.5pp pooled and concentrates in
the conversational-memory slices: LME +9.6pp, ConvoMem +11.2pp, LoCoMo +8.6pp,
vs only +1.8-3.8pp on HotpotQA/MuSiQue/NQ. On LME the headroom decomposes into
three question types preferring three different actions: temporal-reasoning ->
structured (+8.8pp), knowledge-update -> raw (+10.3pp), multi-session ->
summary. Per-(row,reader) oracle adds only +2.1pp pooled over the reader-blind
query oracle — reader heterogeneity confirmed as a non-factor. Caveat:
ConvoMem/LoCoMo cells have only 4 readers (directional).
Target restated per discussion: on open-domain QA we only need to MATCH
always-summary, not beat it; the wins live in conversational memory.

**2. ConvoMem failure forensics (predecessor-repo runs, free).** Damage set:
155/500 queries (qwen-2.5-72b: naive fully correct, SIEVE fully wrong).
Refinement of the diagnosis: schema/intent misclassification is a MINOR
contributor — the chosen schemas are broadly reasonable. Two mechanisms carry
90% of the damage, both flowing through the `confidence_gate_fallback` and
`schema_grounding_partial` routes (139/155):

- **Evidence starvation (56% of damage):** mean compiled evidence on the
  damage set is 14.4 tokens vs 617 naive prompt tokens; readers receive
  half-empty structures and abstain (SIEVE abstention 36% vs naive 20%). The
  answer was present in the raw passages.
- **Stale-value selection (30%):** slot binding latches onto the first unit
  satisfying the schema type with no supersession check; hints bake in stale
  values the reader echoes (e.g. priority 'Low' when the update says 'High').
  Check pending: ConvoMem timestamps may not parse, silently zeroing the
  freshness signal.
- Plus 2 outright-wrong deterministic no-reader answers and 4 null-context
  hallucinations induced by confident-looking hints.

Implication: a trivial guard — "if grounding is partial or the confidence gate
fires, emit filtered raw instead of sparse structure" — should recover most of
the collapse even for v1. This becomes a pilot condition and the simplest
form of the do-no-harm fallback.

**3. Pilot scoping (free).** Total cost **$3-5**: a schema-compatible
`convomem_bm25_top20_v1.jsonl` (500 rows) already exists in the predecessor
repo; HotpotQA slice needs no adapter; filtered_raw/extractive packagers are
LLM-free and domain-agnostic. Gaps: (a) `structured`/`structured_quotes`
packagers depend on the v1 LME compilation cache — the M-sized work item is a
generic structured packager, which is the scientific object anyway; (b)
compile phase has no cost guard (replay does); (c) new config + tests (S).

**Revised pilot conditions** (100 queries x ConvoMem + HotpotQA x 2-3
readers): raw uncapped, raw@400, filtered_raw@400, extractive@400,
summary@400, structured_generic@400, and **guarded-structured** (structured
with filtered-raw fallback on partial grounding). Hypotheses: (i)
guarded/generic compilation eliminates the ConvoMem collapse (>=70% vs v1's
47%); (ii) nothing needs to beat summary on HotpotQA, only match it within CI.

**Status 2026-07-14 (later): pilot RUN and analyzed** (~$1.50; full tables in
`results/v2_runs/generic_cross_bench_pilot_analysis.md`). Verdicts:

- **H-i CONFIRMED.** ConvoMem guarded_structured 66.5-70.5 vs v1's 46-48
  (+20pp, collapse eliminated, zero benchmark-specific code); guard adds
  +1-1.5pp for all readers; qwen raw cells reproduce canonical within ~1pp.
- **H-ii FAILED, informatively.** On HotpotQA at 400 tokens only summary
  survives (71 vs 25-40 for every filter/extract/structure condition; two
  gold paragraphs do not fit a 400-token truncation). The guard correctly
  rejected structuring 93/100 but its filtered_raw fallback is inadequate on
  multi-doc QA.
- **Headline fact:** opposite-signed 30-40pp representation effects across
  benchmarks under one pipeline (ConvoMem best = HotpotQA near-worst, and
  vice versa). The query-conditioned thesis in one table.

Design consequences now binding on the construction: (1) do-no-harm is a
LADDER (filtered_raw -> summary -> uncapped raw) driven by a sufficiency
signal, not a single fallback; (2) budget/aperture joins the action space
(raw@0 vs raw@400 = +21-41pp on HotpotQA, ~0 on ConvoMem); (3) the guard's
coverage signal is a free, informative policy feature (fire rate 20% vs 93%
separates the domains). Compile path fix: `COMPILE_REASONING` disables
thinking for compile models (truncated-summary bug, tested).

Original build status: pilot infrastructure BUILT (working tree, uncommitted).
`v2/packagers/structured_generic.py` (no v1 cache, no benchmark vocab,
MIN_EVIDENCE_TOKENS=120 floor) + `v2/packagers/guarded_structured.py`
(stem-overlap coverage guard >= 0.5, guard path recorded in metadata) +
`v2/configs/generic_cross_bench_v0.json` + ConvoMem slice copied. 341 tests
pass. $0 compile smoke on 10+10 rows: mean evidence ~380 tokens on ConvoMem
(v1 damage-set mean was 14.4) — starvation eliminated by construction; guard
fired 4/10 ConvoMem, 8/10 HotpotQA (consistent with structure rarely paying
on QA). Revised full-pilot cost: **~$0.76** all-OpenRouter (cheap 7-9B
readers, qwen3-8b compile, deepseek-chat-v3 judge — judge must stay
deepseek-chat-v3 for comparability with all canonical numbers; local-judge
option rejected for that reason). ConvoMem comparison cells (matrix):
structured_v1 0.46-0.48 vs raw 0.48-0.79 vs summary 0.71-0.81 across 5
readers.

## Cheap next steps, in order

1. **Offline (free):** per-query oracle and realizable routing analysis over
   {raw, summary, structured} *pooled across benchmarks* — quantify what
   query-conditioned routing buys over always-summary globally, not just LME.
2. **Qualitative (free):** read ~20 ConvoMem SIEVE outputs; confirm the
   failure is schema-misfire evidence deletion (mechanism section of paper).
3. **Small paid:** compile the 4 generic packagers on a 100-query ConvoMem +
   HotpotQA slice, replay 2-3 readers — verify generic operations close the
   transfer gap before building the learned scorer.
