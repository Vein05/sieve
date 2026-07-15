# SIEVE v2 research context

Last updated: 2026-07-14

## Read this before proposing research

This repository is for a **new SIEVE v2 paper**. It is not a workspace for
re-proving the result of the already-published SIEVE/reader-scaling paper.

The canonical predecessor repository is:

`/Users/vein/Documents/research/memory-eligibility-feasibility`

Its paper is:

`/Users/vein/Documents/research/memory-eligibility-feasibility/pdf/main.pdf`

Title: **Fixed RAG Compression Collapses Measured Reader Scaling**. The work is
already published at EMNLP Findings. Treat its findings as prior work and cite
them; do not present them as the novelty of this repository.

## What the published paper already proved

The prior paper is a diagnostic/evaluation paper. It explicitly states: “We do
not propose a new compressor.” SIEVE v1 is an experimental apparatus in that
paper, alongside compression systems from other labs.

The established findings include:

- Fixed post-retrieval compression is reader-dependent: it generally helps
  weaker readers more and can damage stronger readers.
- Compression can hide reader upgrades and corrupt reader rankings.
- On HotpotQA, a 45.4 percentage-point raw reader upgrade shrinks to 9.0 points
  through RECOMP (20% upgrade retention).
- Generic summarization flips 31% of pairwise reader rankings on LongMemEval-S.
- On LongMemEval-S, SIEVE gain correlates negatively with raw reader baseline
  (`r = -0.887`).
- The pattern was tested across 20 readers, 12 model families, ten controlled
  domain-method settings, four QA benchmarks, and QMSum summarization.
- Controlled compressors include SIEVE, generic LLM summarization,
  LLMLingua-2, SIEVE-NLP, RECOMP, EXIT, and Provence.
- The paper also audits nine published compression papers. This repository does
  not need to rediscover that modern compressors exhibit the same scaling issue.
- The effect is not merely a BM25 artifact: it replicates with dense retrieval
  (`r = -0.921`).
- Compression breaks 692 of 4,510 raw-correct LongMemEval-S reader-row pairs
  (15.3%). Reinjection recovers much of this damage, supporting information loss
  as the mechanism.

Most importantly, the prior paper already evaluates routing:

| Strategy | Average accuracy |
| --- | ---: |
| Always compress | 48.4% |
| Per-reader threshold | 49.5% |
| Learned reader-blind per-row router | 49.9% |
| Reader-blind per-row oracle | 53.1% |
| Per-(row, reader) oracle | 56.5% |

The reader-blind oracle accesses only 70% of total routing headroom because 37%
of rows have mixed labels: the same compressed evidence helps some readers and
hurts others. A learned reader-blind router adds about 1.4 points but leaves the
scaling-collapse correlation essentially unchanged (`-0.855` versus `-0.854`).

This reader-blind ceiling is the starting point for v2.

## The new paper

The v2 hypothesis is:

> The optimal representation and aperture of retrieved evidence depend on the
> downstream reader’s capability. A capability-conditioned compiler can retain
> weak-reader gains without bottlenecking stronger readers.

The intended paper structure is:

1. **Prior impossibility:** reader-blind fixed compression has a measurable
   ceiling (cite the published paper).
2. **Construction:** infer a black-box reader capability profile `z_r` and use
   it to choose evidence style and budget.
3. **Restoration:** preserve raw reader scaling, reduce raw-correct damage, and
   recover more per-(query, reader) oracle headroom than reader-blind policies.

The new contribution is **not** any of the following by itself:

- that compression helps small readers more than strong readers;
- that structured evidence can beat a raw BM25 dump for an 8B/9B reader;
- that fixed compression sometimes damages strong readers;
- a new comparison showing RECOMP, EXIT, or LLMLingua also has this problem;
- a query-only router or reader-baseline threshold;
- another presentation of SIEVE v1 as a universal compressor.

Those are either already established or insufficiently new.

## Role of SIEVE v1 in v2

SIEVE v1 is one representation expert, not the v2 headline method. The v2
response surface contains:

- uncapped and budgeted raw evidence;
- filtered raw evidence;
- extractive evidence;
- v1-selected content rendered raw;
- v1-selected content rendered extractively;
- structured SIEVE evidence;
- structured evidence plus exact quotations;
- generic abstractive summary;
- exact published SIEVE-v1 as an external fixed-system control.

The factorial styles share one reader prompt template. Exact SIEVE-v1 is the
intentional exception because it replays the published system, including its 40
deterministic no-reader routes.

The policy should ultimately condition on a capability profile `z_r`, not a
categorical model ID. Proposed profile dimensions include distractor tolerance,
multi-hop integration, temporal reasoning, paraphrase/quotation sensitivity,
context utilization depth, and abstention calibration. Router claims require
both held-out queries and held-out reader families.

## Current go/no-go experiment

Before training a router, map the reader × representation × budget response
surface. Proceed only if:

- oracle-adaptive representation is meaningfully better than every fixed
  policy;
- representation preference differs across readers, not only across queries;
- the interaction is stable under paired/bootstrap uncertainty;
- a reader-conditioned policy has headroom over reader-blind policies;
- the result transfers to unseen reader families.

If all reader response surfaces are effectively the same, do not build the
capability profiler or learned policy.

## Completed pilot and its correct interpretation

Run:

`results/v2_runs/pilot_qwen35_9b_bf16_validation_v0/`

Setup: 91 validation queries, Qwen3.5-9B BF16, 546 reader calls, parallelism 32,
DeepInfra BF16 route, fallbacks and reasoning disabled.

| Condition | Accuracy |
| --- | ---: |
| Uncapped raw | 36.3% |
| Raw at 400 | 38.5% |
| Selected raw at 400 | 37.4% |
| Selected extractive at 400 | 31.9% |
| Structured at 400 | 46.2% |
| Structured + quotes at 400 | 46.2% |

The best fixed condition is 46.2%. The per-query representation oracle is
59.3%, an oracle gap of +13.2 points (query-bootstrap 95% CI: +6.6 to +18.7).
Mixed queries are 41.8%. Structured versus uncapped raw has 17 rescues and 8
damages.

Correct interpretation:

- This is promising evidence of **query-dependent representation headroom for
  one 9B reader**.
- It does not yet demonstrate a reader × representation interaction because
  only one reader was tested.
- It does not establish that a capability-conditioned router will transfer to
  new readers.
- The 59.3% number is a hindsight representation oracle, not a retrieval oracle,
  gold-evidence accuracy, BM25 ceiling, or deployable policy result.
- The 46.2% result is not “wrong half the time” in a way that diagnoses the
  compiler alone; retrieval, evidence sufficiency, reader behavior, and judging
  remain separate factors.

## Required next evidence

The smallest high-information next test is to replay the same 91 queries and
same six pilot conditions on a genuinely strong reader. This costs 546 reader
calls and requires no new compilation. Measure:

- paired representation × reader interaction;
- the difference-in-differences between structured and raw gains;
- best fixed representation per reader;
- per-reader oracle gap;
- rows where the two readers prefer conflicting representations;
- rescue/damage transitions relative to uncapped raw;
- reader-blind versus per-(query, reader) oracle headroom.

If the strong reader has a meaningfully different response surface, add one mid
reader before expanding to the full matrix. Do not immediately run all 11,500
variants or train a router.

## Qwen3.5-397B Alibaba follow-up

The same 91-query, six-condition pilot was replayed with
`qwen/qwen3.5-397b-a17b`, locked through OpenRouter to the `alibaba` provider
with fallbacks disabled. All 546 successful records report `Alibaba`; OpenRouter
reports the endpoint quantization as `unknown`.

- Raw uncapped: 37.4%
- Raw at 400: 42.9%
- Structured at 400: 45.1%
- Structured plus quotes at 400: 42.9%
- Per-query representation oracle: 57.1%
- Oracle gap over best fixed: +12.1pp

This is not a strong capability contrast on the benchmark: uncapped raw is only
+1.1pp above the 9B reader. Both readers select `structured@400` as their best
fixed policy. The reader-blind per-query oracle gets 104/182 reader-query pairs
correct versus 106/182 for the per-(query, reader) oracle, so reader conditioning
adds only two correct pairs in this two-reader pilot. Interaction bootstrap CIs
include zero.

Do not treat this result as a rejection of v2: parameter count is not the
capability variable, and the 397B reader did not establish a materially stronger
raw baseline. It does mean that the next reader must be architecture-different
and empirically frontier on this slice. Do not expand the full surface or train
the router until that contrast is tested.

## MiMo-V2.5 Xiaomi follow-up

The same pilot was replayed with `xiaomi/mimo-v2.5`, locked through OpenRouter
to Xiaomi's FP8 endpoint with fallbacks and reasoning disabled. All 546 records
completed successfully and report `Xiaomi` as the serving provider.

- Raw uncapped: 35.2%
- Raw at 400: 37.4%
- Structured at 400: 45.1%
- Structured plus quotes at 400: 44.0%
- Per-query representation oracle: 52.7%
- Oracle gap over best fixed: +7.7pp

MiMo is architecture-different but again not an empirically stronger reader on
this slice: its raw scores are slightly below Qwen3.5-9B. Structured-minus-raw
gains are identical to the 9B reader at both raw controls, so the aggregate
reader interaction is 0.0pp.

However, the canonical `paper_mimo25_naive` run scores 52/91 (57.1%) on these
exact IDs, while the v2 shared-prompt `raw@0` condition scores 32/91 (35.2%).
The v2 condition has 23 damages versus the canonical answers and 11 additional
`Unknown` responses. The prompt/evidence format, endpoint, reasoning setting,
and abstention instruction differ, so do not interpret the 35.2% result as an
intrinsic MiMo capability estimate. Diagnose this presentation shift separately.

Across all three pilots, `structured@400` is the best shared fixed policy. The
reader-blind per-query oracle gets 152/273 reader-query pairs correct versus
154/273 for the per-(query, reader) oracle. Reader conditioning adds only two
pairs, and the reader-blind oracle captures 93.3% of available headroom.

These three pilots support query-adaptive representation choice, not yet
reader-conditioned choice. The next model must be selected by demonstrated raw
LongMemEval capability rather than parameter count or general benchmark
reputation.

For the next model, run a raw-only 91-query screen first. Expand to the other
five representation cells only if that model is materially stronger under the
current shared prompt. This is cheaper and prevents another nominally strong
model from consuming a full surface without creating the required capability
contrast.

## Offline matrix mining result (2026-07-14)

The planned offline test of reader-conditioned learnability was run against
`data/router_matrix/v0.parquet` (21 readers x 500 LongMemEval rows x 3 uncapped
styles). Full writeup: `research/matrix-mining-reader-conditioning-2026-07.md`.

- The per-(row, reader) oracle gap over the reader-blind row oracle is +3.4pp
  and survives judge-noise correction at ~2.6pp. It is real but small.
- Aggregate reader x style preferences are highly stable (split-half r = 0.845)
  and the equal-baseline/opposite-effect existence pairs are confirmed
  (Phi-4 vs Gemma-3-12B on summary; OLMo +18.2pp vs MiMo -10.0pp).
- But no realizable reader-conditioned policy captures the gap: per-reader
  fixed adds +0.2pp over always-structured; per-(reader, question-type) adds
  +0.0pp over per-question-type reader-blind; calibration-similarity-weighted
  row voting adds +0.25pp over an unweighted cross-reader majority under
  reader-family holdout. The best single similar donor is worse than the
  unweighted majority: per-row reader disagreement is noise-dominated.
- Query-side conditioning is what is learnable: +2.5pp at question-type
  granularity with ~7pp further row-level headroom, consistent with the
  pilots' per-query representation oracle gaps.

Consequence: the bet-2 kill criterion fires on the v1-era style surface. Do
not build the capability profiler or buy a larger reader panel for style
routing. The open decision is whether to (a) reposition around query-adaptive
representation plus the reader-conditioning negative result, or (b) first test
presentation/format-level reader adaptation (the MiMo 57.1 -> 35.2 shared-
prompt collapse and LongMemEval CP4's 10pp format swings suggest ~10-20pp
reader x presentation effects, an axis absent from the style taxonomy).

## Direction pivot: benchmark overfitting, not reader conditioning (2026-07-14)

SIEVE v1 is LongMemEval-specific by construction (schemas mirror the LME
question taxonomy; `concept_specs.json` contains literal LME instance
content). Matrix evidence: SIEVE +8pp on LME but -25 to -35pp vs raw on
ConvoMem; RECOMP (QA-trained) is the mirror image (68.6% HotpotQA, 24.9% LME).
Generic query-focused summary is the strongest fixed policy everywhere
(63.2% pooled). Per-query routing headroom concentrates in conversational
memory (+8-11pp) and is small on open-domain QA (+2-4pp; matching summary
there is sufficient). ConvoMem forensics: 90% of SIEVE damage is evidence
starvation (14-token packages -> spurious abstention) and stale-value slot
binding, not intent misclassification. Proposal and full evidence:
`research/v3-learned-compiler-proposal-2026-07.md`. Next concrete step: the
$3-5 generic-packager pilot on ConvoMem+HotpotQA defined there.

Framing update (same day): position as **read-time memory compilation for
agents** (contrast: write-time memory systems Mem0/SeCom/A-Mem/LIGHT), not RAG
compression. Primary suite = LongMemEval, LoCoMo, ConvoMem, BEAM (local slice
with gold answer-bearing labels: `beam_adapted_bm25_top20_v2.jsonl`, 7x40
abilities), LoCoMo-Plus (ACL 2026, build from their repo). QA benchmarks stay
as match-summary transfer controls. Key design facts: LoCoMo-Plus is
adversarially filtered against BM25/MPNet cue recovery (query-conditioned
retrieval fails by construction -> two-channel pack with standing-constraints
digest); BEAM contradiction resolution is an open problem (all systems
0.00-0.08) targeted by a generic supersession-aware packager. Free next
analysis: BEAM retrieval-completeness diagnosis via answer_bearing_memory_ids
(done: contradiction 65% complete/0% missing = compilation-addressable;
event_ordering 5% complete = retrieval-bound).

**Pilot RUN and analyzed (2026-07-14, ~$1.50):**
`results/v2_runs/generic_cross_bench_pilot_analysis.md`. H-i confirmed:
guarded generic structuring scores 66.5-70.5 on ConvoMem vs v1's 46-48
(collapse eliminated; guard +1-1.5pp, fired 20/100). H-ii failed: at 400
tokens on HotpotQA only summary survives (71 vs 25-40 for all
extract/filter/structure conditions; guard fired 93/100 but its filtered_raw
fallback is inadequate there). Headline: opposite-signed 30-40pp
representation effects across benchmarks under one pipeline — the
query-conditioned thesis in one table. Design consequences: do-no-harm must
be a LADDER (filtered_raw -> summary -> uncapped raw), and budget/aperture
belongs in the action space (raw@0 vs raw@400 = +21-41pp on HotpotQA).
Compile path now disables reasoning (COMPILE_REASONING in cli/run_v2.py) —
thinking compile models truncated summaries before the fix.

Learned-components design (L1 sufficiency estimator, L2 utility-regression
policy, L3 per-unit scorer; sequencing and kill criteria):
`research/learned-components-design-2026-07.md`. Key gate: L2 must beat the
no-learning heuristic ladder under leave-one-benchmark-out or the ladder
ships as the method.

**Heuristic ladder built, then RETRACTED by its own transfer test
(2026-07-14, $0):** `results/v2_runs/heuristic_ladder_offline_v0.md`. The
pool-fit ladder ("pool > 1.5x budget -> summary, else raw passthrough")
beat all fixed policies on the two-benchmark pilot (70.8 pooled) — but the
frozen rule FAILS on the matrix slices it never saw, and even with the
threshold tuned in-sample it never beats always-summary on any slice
(pooled 68.9 = 68.9). Post-mortem: (1) the pilot's summary comparator used
the broken one-line prompt — canonical summary scores 71-81 on ConvoMem
with the same readers, beating raw; (2) pool-fit was a proxy for benchmark
identity, not a causal signal — MuSiQue raw is 19.6 vs summary 49.0 at any
pool size, so raw passthrough is NOT universal do-no-harm; (3)
selected_memory_tokens units drift across matrix slices — re-derive pool
sizes from slice files. Binding conclusions: the no-learning baseline is
**always-summary with the canonical prompt** (68.9 pooled over 5 slices);
the ladder inverts (summary is the base action, escalation must be earned);
honest LORO headroom over always-summary is LME +8-11pp (21 readers,
reliable), ConvoMem +7.8pp (4 readers), QA +0.2-2.8pp (match-only
confirmed). LoCoMo has no summary cells in the matrix — untested gap. L2's
value case rests on LME/ConvoMem/BEAM/LoCoMo-Plus and the engineered ops,
not on QA mining. Summary packager prompt confound FIXED
(`v2/packagers/summary.py`, test-pinned canonical prompt). The summary packager prompt confound is
FIXED: `v2/packagers/summary.py` now uses the exact canonical LLM-Summarize
prompt (test-pinned), so the ladder numbers are lower bounds; summary cells
must be recompiled in the response-surface run.

Theory + novelty landscape: `research/read-time-compilation-theory-2026-07.md`
— definition of v1, the seven missing capabilities, and a 2026 literature
check. WARNING: the read-time niche is no longer empty — DeferMem (arXiv
2605.22411, query-time RL distillation) and TRACE (arXiv 2607.00339,
validity-aware temporal evidence graphs) are directly adjacent, both within
the last 8 weeks. Defensible novelty = policy over representations trained on
cross-reader/cross-distribution utility + do-no-harm + the two-axis
impossibility evidence. Move fast.

**L1 v0 sufficiency estimator: NEGATIVE (2026-07-14, $0, agent-built):**
`research/l1-sufficiency-v0-results-2026-07.md` + code in `v2/l1/` (tests
pass, 375 total). 8,167 gold-containment labels (HotpotQA/LME/BEAM). GBM
wins on transfer AUC (+18pp over the stem guard on every held-out
benchmark) but makes WORSE escalation decisions at matched fire rate and
loses to plain stem coverage on ConvoMem reader-outcome correlation (0.56
vs 0.36-0.48). Kill criterion fired: ship the stem-coverage guard as L1;
pack-statistics features are exhausted (three independent negatives now
agree). Content-level modeling only via the L3 cross-encoder rung.

**BEAM reader-free diagnostics (2026-07-14, $0):**
`results/v2_runs/beam_reader_free_diagnostics_v0.md`. Gold-unit survival at
budget 400: filtered_raw destroys 60-90% of answer-bearing units (third
strike against "filtered raw is safe"); extractive preserves 65-90% on the
five compilation-addressable abilities and is the deterministic selection
op to beat (L3's bar). event_ordering fails for both (aperture-bound).
Supersession probe: ~30% BM25 rank inversions (stale rival outranks gold);
gold is the latest version only 35% of the time on knowledge_update (81%
contradiction) -> supersession must render CHAINS, not resolve-to-latest —
the empirical wedge against the resolve-to-current cluster below.

Supersession/staleness is ALSO now a crowded subfield (checked 2026-07-14;
table in the theory doc): MemStrata 2606.26511 (write-time ledger, retires
stale values; embeddings can't separate contradiction from duplicate AUROC
0.59), Supersede 2606.27472 (LME knowledge-update gap, training-based),
Don't-Ask-the-LLM 2606.01435 (deterministic max(serial) beats LLM
judgment), TOKI 2606.06240, ConflictRAG 2605.17301. The stale-version
retrieval hazard itself is NOT ours to claim. Open differentiator our BEAM
probe supports: all of them resolve-to-current, but BEAM knowledge_update
gold is the latest version only 35% of the time (contradiction 81%) —
supersession-as-deletion destroys answer-bearing history;
read-time chain rendering preserves it. Planned experiment:
resolve-to-latest baseline vs chain rendering on those two abilities.

## Direction decision 2026-07-14 (evening): the interrogator bet

Root-cause diagnosis accepted after the day's negatives: post-retrieval
re-representation is a SECOND-ORDER intervention (correctness tracks
evidence PRESENCE — reinjection recovery 61-94%, LORO routing residue
~3pp, aperture swings +21-41pp). The z_r bet is dead, the query-routing
construction is ACL-tier at best, and the user rejected the
science-paper/utility-retriever alternatives. New core bet (option 1):
**read-time evidence completion — the compiler as interrogator.** It reads
the pool, detects structural incompleteness (broken update chains, missing
comparison sides, absent constraint classes, temporal gaps), and issues
targeted probes back into the store until predicted-sufficient, then
packs. Differs from FLARE/IRCoT-style iterative retrieval: probes are
driven by state semantics of a conversational history, not generation
uncertainty. Pre-registered kill-test (Opus agent, $0, launched): on
BEAM event_ordering/multi_session + LME multi-session/temporal rows with
missing gold, measure the fraction of missing gold turns that are
probe-reachable (probes generated ONLY from pool content + query; no gold
peeking) but NOT reachable by matched-budget k-expansion of the original
query. If that fraction is small, the interrogator dies before we build it.

**Kill-test result (same day): BET LIVES, PROVISIONALLY — 22.8% (31/136)
of accessible missing gold on retrieval-bound BEAM types is
probe-exclusive at matched budget B=30 (threshold >=15%); k-expansion
alone covers 61.8% (aperture is real but leaves a residue); reverse
exclusivity only 3/136.** Full writeup:
`research/interrogator-reachability-2026-07.md`; code
`analysis/interrogator_*.py` + tests (387 pass). THREE CAVEATS BINDING THE
NEXT STEP: (1) BEAM-only — the LME haystack
(longmemeval_s_cleaned.json) is absent locally and the sieve-repo slice
has empty gold fields, so LME multi-session/temporal went untested; (2)
the store had to be reconstructed by unioning ~14 sibling pools per chat
(~85% of true turns) and this biases TOWARD the interrogator — 22.8% is a
soft upper bound; (3) probe-type ablation: the exclusive signal is
entirely entity-refocus + order-adjacency probes (chain/temporal/
decomposition contributed ZERO) — the honest v0 mechanism is structured
query expansion + adjacency, not the grander "broken-chain detection"
framing. CONFIRMATION GATE before any method building: download true
128K BEAM source (HuggingFace-gated) + LongMemEval-S haystack, re-run the
same frozen harness; entity+adjacency exclusivity must survive >=15% on
the full noisy store or the 22.8% was reconstruction bias.

**Confirmation-gate data UNBLOCKED (2026-07-15 by content, dated 07-14 in
filenames): both true stores are now local.** (a) BEAM was never gated — the
kill-test agent simply had no network; and the HF repo's "100K" parquet IS
the 128K tier (the BEAM paper has no 100K size; 20 conversations = the 128K
count; slice memory contents verified verbatim against chat content, 19/20
sampled, chat_id = conversation_id). True store:
`data/beam_source/data/100K-00000-of-00001.parquet`. (b) The
`longmemeval_s_cleaned.json` haystack is gone from every local repo (data
dirs emptied in a Jun 3 cleanup; never in git/LFS), but the official
`xiaowu0162/longmemeval` release is public; downloaded to
`data/longmemeval_source/longmemeval_s` (500 questions, haystack_sessions +
answer_session_ids; maps 500/500 to our slice by exact question text).
CAUTION for LME: the sieve slice's pools have answer sessions INJECTED by
construction, so honest missing-gold analysis must rebuild pools with plain
BM25 over the haystack (no gold injection). Confirmation re-run launched on
the frozen harness with these true stores; same pre-registered thresholds
(>=15% live, <5% dead), BEAM and LME verdicts reported separately.

## Retrieval and BM25: keep the questions separate

LongMemEval-S BM25 top-20 averages roughly 700 reader tokens in the published
setup. It is not the full 115K-1.5M-token memory history.

- Full-history prompting, raw BM25 top-20, and gold-only evidence are different
  conditions and must not be called “raw context” interchangeably.
- A retrieved gold session ID does not guarantee that the answer-bearing span
  survived chunking or truncation.
- Multi-session questions have a documented retrieval/completeness bottleneck,
  but the published fixed-compression interaction also survives dense retrieval.
- Keep the candidate pool fixed during the initial representation response
  surface so that retrieval availability is not confounded with representation.
- After the representation interaction is established, adaptive aperture/top-k
  is a legitimate part of v2: style and evidence budget can be selected jointly.
- Diagnose BEAM retrieval completeness separately from compilation quality, as
  required by the v2 handoff.

## Terminology guardrails

Always qualify the word **oracle**:

- **Gold-evidence oracle:** the reader receives official answer-bearing evidence.
- **Retrieval oracle:** hindsight selection of evidence from the corpus/pool.
- **Representation oracle:** hindsight choice among already generated evidence
  representations.
- **Reader-blind per-query oracle:** one representation choice per query shared
  by all readers.
- **Per-(query, reader) oracle:** representation can vary for both query and
  reader.

Never describe the current 59.3% pilot oracle as an oracle retrieval ceiling.

## Source-of-truth documents

- `research/idea-v2.md` - current scientific thesis and experiment design.
- `research/v2-iclr-assessment.md` - novelty assessment, paper spine, risks, and
  kill criteria.
- `handoff/V2_RESEARCH_HANDOFF.md` - implementation status, validation, and next
  research actions. The `handoff/` directory is intentionally git-ignored.
- `v2/README.md` - response-surface execution and methodological controls.
- `v2/configs/response_surface_v0.json` - styles, budgets, splits, and reader
  panel requirements.
- `results/v2_runs/pilot_qwen35_9b_bf16_validation_v0/pilot_analysis.md` - current
  pilot result.
- `/Users/vein/Documents/research/memory-eligibility-feasibility/pdf/main.pdf` -
  published predecessor paper.
- `/Users/vein/Documents/research/memory-eligibility-feasibility/arxiv/sections/appendix.tex`
  - routing hierarchy, mixed-label result, and reader-blind ceiling.

## Default research decision

Do not ask, “Can SIEVE beat every compressor?” as the first question. The new
paper asks whether reader-conditioned selection over multiple representations
can break the reader-blind ceiling established by the prior paper. Compressor
leaderboards are supporting baselines after the core interaction is validated,
not the research destination.
