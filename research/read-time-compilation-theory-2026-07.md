# Read-time compilation: what v1 is, what is missing, who else is here (2026-07-14)

Companion to `research/v3-learned-compiler-proposal-2026-07.md`.

## What SIEVE v1 actually is

One sentence: **a deterministic, reader-blind, single-representation read-time
compiler whose query model is a hand-built classifier over LongMemEval's
question taxonomy.**

Pipeline: BM25 top-20 -> regex intent classification (524-line marker
vocabulary) -> one of 8 hand-crafted schemas mirroring LME question types ->
rule-based slot binding (3,056-line runtime; temporal/numeric normalization)
-> 5-signal hand-weighted controller (commit >= 0.55) -> structured rendering,
or a deterministic no-reader answer (40/500).

Defining properties, each of which is a design decision v2 must reverse or
justify:

| Property | v1 | Consequence |
| --- | --- | --- |
| Query understanding | benchmark-taxonomy classifier | ConvoMem/QA misfire |
| Representation | one (structured) always | leaves +6.5pp routing headroom |
| Sufficiency check | gates exist but starved packs ship anyway | 56% of ConvoMem damage |
| State/updates | recency rules keyed to LME timestamp formats | stale-value bug, 30% of damage |
| Selection labels | hand rules | encodes the heuristic, not utility |
| Reader model | blind (correct per matrix mining) | keep |
| Cost model | none | no Pareto argument |
| Failure mode | silent evidence deletion | worst possible: invisible |

## The seven missing capabilities

**1. Sufficiency estimation before emission.** The compiler must estimate
whether the pack it built can support an answer, BEFORE shipping it. v1 ships
14-token packages that induce spurious abstention on answerable queries. The
fallback ladder on predicted insufficiency: widen selection -> switch to
filtered raw -> emit an explicit insufficiency note. This alone should
recover most of the ConvoMem collapse (guard-fire analysis in the pilot).
Paper ablation: guarded vs unguarded structured.

**2. Supersession as a first-class generic operation.** Cluster units by
entity/attribute, order by time, render explicit update chains and surface
unresolved contradictions with dates, instead of silently binding one value.
Evidence it matters: 30% of ConvoMem damage is stale-value; BEAM
contradiction resolution is 0.00-0.08 for every published system and 65%
in-pool for us. TRACE (arXiv 2607.00339) validates the direction with
validity-aware evidence graphs but pays 38-45 hours of write-time graph
construction per LME-S corpus; we do it lightweight over the retrieved pool
at read time.

**3. Representation adaptivity.** Query-conditioned choice among styles x
budgets. Our matrix evidence: +6.5pp pooled realizable headroom over the best
fixed policy, concentrated in conversational memory (+8-11pp), decomposing
into distinct per-question-type preferences (temporal -> structured,
knowledge-update -> raw, multi-session/preference -> summary). No existing
system chooses among representations at read time (DeferMem distills into one
fixed form).

**4. Utility-grounded selection.** Train the selector on counterfactual
reader outcomes (did including this unit flip the answer?), not relevance
labels, not v1's selected/rejected labels. Assets nobody else has: the 191K
ragscale matrix, reinjection rows, BEAM per-candidate
`memory_usefulness_labels`, and a 500x23 replay infrastructure for cheap
counterfactual generation on weak readers.

**5. Pragmatics preservation for behavioral queries.** Preference and
implicit-connection queries are damaged MOST by structuring (46%/40% ConvoMem
damage rates): the answer-relevant signal is tone, constraints, context — not
extractable slot values. Level-2 "cognitive memory" (LoCoMo-Plus) formalizes
this class. Design: the policy must learn a raw/lightly-filtered floor for
behavioral queries; structuring is for stateful factual queries.

**6. Standing vs episodic channels.** Query-conditioned retrieval cannot
surface constraints that are semantically disconnected from the trigger query
(LoCoMo-Plus filters cue-query pairs recoverable by BM25/MPNet, by
construction). A standing-constraints digest (durable state/goals/values,
compiled once per history, always included, ~100-200 tokens) fixes what no
amount of per-query compilation can. This is the one place v2 borrows a
write-time idea, but keeps it tiny and provenance-linked.

**7. Calibrated do-no-harm with a cost model.** Every compilation decision
has bounded downside vs raw pass-through, and the policy optimizes the
accuracy-cost frontier, not accuracy alone. The always-summary baseline costs
an LLM pass over the full retrieved context per query; matching it with
mostly LLM-free packagers is a deployment win even at equal accuracy. Damage
rate (broken raw-correct pairs) is a headline metric, not an appendix.

## Literature check: the read-time niche is no longer empty

Closest works as of 2026-07 (all must be cited; two are recent enough to be
scoop-adjacent):

| Work | What it does | What it lacks vs v2 |
| --- | --- | --- |
| **DeferMem** (arXiv 2605.22411, May 2026) | Query-time evidence distillation via RL (DistillPO); raw history kept, segment-linked; LoCoMo 88.3, LME-S 70.0 | Single representation; trained against one fixed reader (GPT-4o-mini); no style/budget policy; no transfer or damage claims |
| **TRACE** (arXiv 2607.00339, Jul 2026) | Temporal evidence graphs; update/contradiction edges; validity-aware support paths at query time | 38-45h write-time graph build; no representation choice; no utility training; +4-6pp over Nemori |
| SeCom (NeurIPS 2024 / MSR) | Segment-level memory units + LLMLingua-2 denoising | Write-time; fixed compression; no query conditioning of representation |
| MemRefine, LightMem, provenance-tiered memory (2026 preprints) | Various write-time compression/tiering | Same fixed-policy pattern |
| Mem0 / A-Mem / MemGPT-Letta | Write-time extraction/consolidation | 15-17 on LoCoMo-Plus cognitive memory; no read-time repr. choice |
| RECOMP / EXIT / LLMLingua / CompAct | QA compressors | Reader-blind, distribution-fixed (RECOMP: 24.9% on LME) |
| FAVICOMP (EMNLP 2025 F.) | Reader-conditioned via ensemble decoding | White-box logits; single axis; and our matrix mining says reader conditioning is not where the gain is |

Implications:

1. **The framing is validated**: DeferMem's core argument — write-time
   compression commits under uncertainty; defer to query time — is exactly our
   premise, published independently. Good for the thesis, bad for claiming
   "first read-time" anything.
2. **Our defensible novelty** is the layer above: read-time compilation as a
   **policy over representations** (style x budget x channel) trained on
   **cross-reader, cross-distribution utility**, with do-no-harm calibration
   — backed by impossibility evidence (reader axis published; data axis new)
   explaining why every fixed variant, including DeferMem's and TRACE's,
   should fail to transfer. Testing DeferMem/TRACE-style baselines on
   ConvoMem/LoCoMo-Plus transfer is itself a paper section.
3. **Speed matters.** Two directly adjacent papers in the last 8 weeks. The
   two-axis impossibility + response surface + policy is a coherent package we
   can execute mostly on existing infrastructure; the window for the framing
   contribution is open but narrowing.

## Scoop-risk deflation after reading the papers (2026-07-14, PDFs in papers/)

**DeferMem's headline numbers are real but narrow.** (a) "0 commercial API
tokens" covers memory operations only — the distiller is a locally served
RL-trained Llama-3.1-8B on an A100; the answerer is GPT-4o-mini via API. The
true read-time cost is an 8B forward pass over ~22K retrieved tokens per
query — the always-summary cost profile, or worse. (b) The 70.0% on
LongMemEval-S sits on embedding retrieval + segment-link expansion achieving
98.2% recall at ~22K tokens (~30x our BM25 top-20 aperture); under the same
setup LightMem scores 68.64% and A-Mem 62.60%, so DeferMem's marginal
contribution is +1.36pp over the best baseline. (c) The judge is GPT-4o-mini,
not the official GPT-4o protocol or a cross-checked judge. (d) DistillPO is
trained on 329 QA pairs; contamination controlled per their reporting, but
130 pairs come from the LongMemEval history-construction corpus
(same-distribution, no overlap claimed). Net: validates the read-time thesis;
does not preempt representation policy, transfer, damage, reader panels, or
the cost frontier — and its own aperture result reinforces that multi-session
performance is retrieval-bound, consistent with our BEAM diagnosis.

**TRACE is a graph-construction paper.** The read-time component scores paths
over a graph that takes 38-45 hours to build per LME-S corpus; gains are
+4.5-5.7pp over Nemori. Cite its validity semantics as motivation for the
supersession packager; it is not a compilation-policy competitor.

## The supersession/staleness cluster (checked 2026-07-14)

The stale-version retrieval hazard is now an actively published subfield —
five directly relevant papers in May-July 2026. None of it may be claimed
as our discovery; our BEAM rank-inversion numbers are corroborating
measurements in the BM25/conversational-memory setting.

| Work | What it does | Relation to us |
| --- | --- | --- |
| MemStrata (arXiv 2606.26511) | Write-time bi-temporal ledger; deterministic (s,r,o) rule RETIRES stale values; shows embeddings cannot separate contradiction from duplicate (AUROC 0.59) | The AUROC 0.59 result is the structural version of our 30% BM25 rank-inversion finding — cite it |
| Supersede (arXiv 2606.27472, Jun 25) | Isolates the memory-update gap on LME knowledge-update (full context 92% vs bounded memory 77%, frontier model); fine-tunes a small model on supersession | Validates the gap is maintenance, not comprehension; training-based, write-time |
| Don't Ask the LLM to Track Freshness (arXiv 2606.01435) | Deterministic max(serial) conflict resolution beats LLM judgment by +10.8pp on MemoryAgentBench FactConsolidation | Independently validates deterministic ops over LLM judgment — same stance as our engineered supersession module |
| TOKI (arXiv 2606.06240) | Bitemporal operator algebra for contradiction resolution in persistent memory | Write-time formalism |
| ConflictRAG (arXiv 2605.17301), RAG w/ conflicting evidence (arXiv 2504.13079) | Detect/resolve inter-document conflicts in RAG | General-RAG framing, not conversational memory |

**The differentiating claim this cluster leaves open** (and our BEAM probe
supports empirically): every one of these systems resolves to the CURRENT
value — retire the stale fact, take max(serial), retrain toward the update.
But on BEAM knowledge_update, the gold answer-bearing unit is the LATEST
version only 35% of the time (contradiction_resolution: 81%): questions
about histories routinely need prior values or the update event itself.
**Supersession-as-deletion is unsafe for QA over histories;
supersession-as-chain-rendering ("X at t1 -> Y at t2", both preserved with
provenance) is strictly safer and is a read-time operation over the
retrieved pool — no ledger, no graph build, no training.** A
resolve-to-latest baseline vs chain rendering on BEAM knowledge_update +
contradiction_resolution is the experiment that makes this a paper section.
