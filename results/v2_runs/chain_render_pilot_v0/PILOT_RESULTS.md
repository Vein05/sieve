# Chain-rendering pilot v0 (2026-07-14, ~$1.20, EXPLORATORY — not pre-registered)

**Question:** does read-time supersession-chain rendering move readers on the two
abilities where failure was universal (knowledge_update, contradiction_resolution),
versus resolve-to-latest and raw chronological evidence?

**Verdict: half-yes, and the yes is on the ability where every published system
scores 0.00–0.08.** Chain rendering induces contradiction-flagging (50% vs 20%
resolve, 0% raw). On knowledge_update chain LOSES to resolve (40% vs 60%) due to
fixable rendering defects plus a BEAM reference-staleness problem that caps what
any history-preserving method can score.

## Setup

- 24 gold-complete BEAM 128K rows (12 knowledge_update, 12 contradiction_resolution),
  first-N by example_id from `beam_transfer_bm25_top20_v0.jsonl` (sibling repo).
- Three conditions, all consuming the SAME chronologically sorted, 24K-token-capped
  top-20 pool; only rendering differs:
  - `raw_chrono`: the capped pool verbatim (~24K tokens).
  - `resolve_latest`: DeepSeek-V3-0324 fact sheet, instructed to keep ONLY the most
    recent version of each updated fact (~110 tokens median).
  - `chain_render`: same compiler, instructed to render FULL update chains
    ("v1 (turn a) -> v2 (turn b) [most recent]") and mark `[CONTRADICTION]` pairs
    (~165 tokens median).
- Readers: Llama-3.1-8B, Qwen-2.5-72B (OpenRouter, no fallbacks, temp 0, narrative
  prompt, max_tokens 512 — NOT the known-bad SHARED_READER_TEMPLATE).
- Scoring: manual by Claude (Fable 5) against BEAM references, strict
  (KU: gold value stated; CR: BOTH sides of the contradiction surfaced). Single
  sample per cell. No LLM judge. n is tiny; treat all numbers as existence
  evidence only.

## Results (reader-pairs correct / total; 4 gold-truncated rows excluded, see caveat 1)

| Condition | knowledge_update (10 rows) | contradiction (10 rows) | pooled |
| --- | ---: | ---: | ---: |
| raw_chrono (24K tok) | 3/20 (15%) | 0/20 (0%) | 3/40 (8%) |
| resolve_latest | **12/20 (60%)** | 4/20 (20%) | 16/40 (40%) |
| chain_render | 8/20 (40%) | **10/20 (50%)** | 18/40 (45%) |

Per-row scores embedded in `chain_pilot.py` tally block of the session transcript;
raw answers in `answers.json`, compiled evidence in `compiled.json`, side-by-side
sheet in `review.md`.

## Findings

1. **Contradiction flagging is evidence-conditional, and chain rendering supplies
   it (the headline).** With resolve-to-latest one side of the contradiction is
   deleted, so flagging is impossible by construction — readers confidently pick a
   side (20% is rows where the compiler disobeyed and kept both). With chains +
   `[CONTRADICTION]` marks, BOTH readers flag both sides and ask for clarification
   on 5/10 rows — the exact behavior BEAM references demand. Identical readers,
   identical pool: the behavior the field scores 0.00–0.08 on is largely a
   *rendering* deficit, not a model deficit, for these readers. This is the wedge
   against the resolve-to-current cluster (TRACE/MemStrata/Supersede/...), and
   TRACE was independently confirmed same-day as NOT rendering history
   (adjacent-but-different; strongest baseline).

2. **BEAM knowledge_update references can be stale against their own chat
   (validity finding).** Row beam-128k-10-knowledge_update-001: reference = 1,350
   words/week (the constructed original->update pair), but the chat verifiably
   continues 1,200 -> 1,350 -> 1,500 -> 1,800 (values present in pool text; 65
   occurrences of "1,800"). Same for 13-002 ($50 vs chat continuing to $75).
   Readers reporting the true current value are scored wrong. These are exactly
   the gold≠latest rows where chain rendering should shine, so the benchmark
   construction caps the measurable chain-vs-resolve contrast on KU. This is the
   mirror image of the "gold is latest only 35%" measurement and belongs in the
   benchmark-validity thread.

3. **Chain's KU losses are rendering-quality defects, not concept failures:**
   (a) mis-ordered chain on 15-002 (compiler cited message-ids as turn order,
   put $600 after $650 -> both readers wrong; resolve got it right); (b) one
   arithmetic derailment (17-001/Qwen summed the chain to $10,700); (c) one
   hedge (15-001/Qwen refused to commit). A deterministic chain builder
   (supersession pairs are already labeled in `perturbation_provenance`
   stale/newer ids) removes (a) and shrinks (b).

4. **Raw 24K chronological evidence is catastrophic for these readers (8%
   pooled):** they derail into continuing the conversation instead of answering.
   ~150-token compiled sheets beat 24K raw by 5-7x. Consistent with the
   lost-in-conversation mechanism and the aperture findings; also means the
   honest baseline for any method claim here is a compiled baseline
   (summary/resolve), never raw context.

## Caveats

1. 4/24 rows had gold turns truncated by the 24K whole-unit cap (cap fills
   early turns first) — excluded from the primary table; including them doesn't
   change the ordering.
2. Compiler saw the question (both compiled conditions, matched) — query-focused
   compilation, not leakage of gold, but a generic-compilation variant should be
   ablated later.
3. `active_context` is empty in this adapted slice, so no time-anchored scoring
   was possible; KU rows with post-gold updates are structurally ambiguous.
4. Single compile sample, single reader sample, self-scored. Pre-registered
   re-run with a per-ability judge required before any number is quoted.

## Consequences for the plan

- The chain-rendering bet is ALIVE with a narrowed headline: **contradiction
  flagging as an evidence-side capability** (rendering restores it; resolution
  destroys it). KU joins as secondary after (a) deterministic chain ordering and
  (b) either time-anchored eval or reference-staleness accounting.
- Adds a second benchmark-validity finding (stale KU references) to the existing
  one (behavioral references under QA prompting).
- Next gates, in order: mechanism controls (CONCAT control; chain WITHOUT the
  [CONTRADICTION] mark to separate "both sides present" from "conflict marked");
  deterministic chain builder; pre-registered re-run at n=40/ability, 3
  samples/row, per-ability judge prompts; then the frontier probe on
  event_ordering stands unchanged.

## Oracle arms (same day, ~$0.20): binding was NOT the KU bottleneck — KU chain rendering is DEAD on BEAM; the contradiction mechanism ladder is confirmed and sharpened

Two deterministic conditions built from BEAM's own labels (perfect binding by
construction, no LLM compiler): `oracle_chain` = all labeled versions of the
queried fact verbatim, chronological, UNMARKED (no [CONTRADICTION] tags);
`oracle_latest` = newer/latest labeled turn only. Same reader prompt/readers.
Sources: KU versions from `perturbation_provenance` stale/newer ids; CR
versions from `answer_bearing_memory_ids` (both sides labeled on 11/12 rows;
cr-1-002 has one label — unusable for flagging, kept, scored 0).
Files: `oracle_answers.json`, `oracle_review.md`, `oracle_chain.py`.

Manual strict scores, all 24 rows (oracle selection sidesteps the cap issue):

| Condition | knowledge_update (24 pairs) | contradiction (24 pairs) |
| --- | ---: | ---: |
| oracle_latest (perfect-binding resolve) | **19/24 (79%)** | **0/24 (0%)** |
| oracle_chain (perfect-binding history) | 16/24 (67%) | 8/24 (33%; Qwen 6/12, Llama 2/12) |

Findings:

1. **KU: perfect binding does not rescue chains — resolve wins even at the
   oracle (79% vs 67%).** BEAM KU gold IS the newer value by construction
   (stale->newer perturbation pairs), so history adds only distraction. The
   chain-vs-resolve question on BEAM knowledge_update is CLOSED, negative.
2. **The KU history-hurts mechanism is prescriptive-norm override** (award-doc
   mechanism candidate 4, first live sighting): given "$35 -> $50 (January)",
   both readers on 13-002 concluded the regular budget "remains $35" and the
   update was "a one-time exception"; Llama did the same on 15-002 ("the
   original $600 is the one being referred to"). oracle_latest gets these
   right trivially. History invites explaining-away of updates in mid-tier
   readers.
3. **Contradiction mechanism ladder, now measured at three rungs on the same
   rows/readers:** one side present (oracle_latest) = 0% — flagging
   mechanically impossible; both sides co-present unmarked (oracle_chain) =
   33% (Qwen manages 50%, Llama 17%); both sides + explicit [CONTRADICTION]
   marks (pilot LLM chain) = 50% for BOTH readers. Co-presence is necessary;
   for weak readers the explicit conflict MARK is load-bearing. That is the
   method: supersession-aware rendering that marks conflicts, evaluated on
   contradiction_resolution.
4. **KU binding-error costs measured as a byproduct:** LLM resolve (60%) loses
   ~19pp to oracle resolve (79%); LLM chain (40%) loses ~27pp to oracle chain
   (67%). Slot binding is a real cost center for ANY compiled representation —
   but fixing it is the TRACE-cluster agenda, not our wedge.
5. **More reference-validity evidence:** on ku-11-001 and ku-12-002 even the
   gold-turn-only condition cannot produce the reference answer (all four
   oracle cells fail identically) — references appear unsupported by their own
   labeled evidence. And two CR pairs (12-001 read-vs-recommended,
   16-002 attend-vs-recommended) are not genuine contradictions; readers
   reasonably reconcile them and get scored wrong. The BEAM-validity thread
   now has three independent strands.

Consequence: the bet is now exactly one claim wide — **conflict-marked chain
rendering restores contradiction-flagging (0 -> 33 -> 50 ladder) where the
field scores 0.00-0.08** — with KU retired (negative, mechanism named) and the
validity findings as a supporting thread. Next: pre-registered n=40 CR re-run
with per-ability judge + the [CONTRADICTION]-mark ablation kept as an arm.

## One engineering trap (bit us again today)

`reader/pali_shims.py::clean_generated_answer` extracts the LAST LINE (span-answer
machinery). Any narrative or fact-sheet output routed through `clean_answer`
collapses to one line — the first run of this pilot silently reduced compiled
evidence to ~20 tokens. Use `raw_answer` for anything non-span. Fourth sighting
of the presentation-artifact class.
