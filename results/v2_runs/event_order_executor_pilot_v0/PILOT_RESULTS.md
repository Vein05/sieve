# Event-order executor pilot v0 — result

**Verdict: DEAD under the pre-registered gate.** Deterministically sorting
query-focused, provenance-linked records improves over leaving those records
unordered, but it does not beat simply giving the reader the gold turns in
chronological order. Do not build the proposed executor from this result.

## Setup

- 20 BEAM `event_ordering` rows, first valid rows in slice order.
- Gold-complete evidence from official `answer_bearing_memory_ids`.
- The compiler never sees the reference answer.
- Qwen-2.5-72B serves as both compiler and reader.
- DeepSeek-V3-0324 judges exact semantic item coverage and order.
- All 20 compiler outputs cover their required source IDs; all 80 reader calls
  succeeded; gold-through-judge validity is 20/20.

Conditions:

1. `gold_chrono`: complete labeled turns in chronological order.
2. `extracted_unsorted`: one query-focused record per labeled turn, shuffled.
3. `executed_sorted`: the same records sorted deterministically by source turn.
4. `executed_corrupted`: the sorted trace rotated by one position.

The frozen live criterion required sorted accuracy >=30% and a >=15-point gain
over the better of the first two controls. The dead criterion was sorted
accuracy <=15% or a gain below 5 points.

## Results

| Condition | Correct | Accuracy |
| --- | ---: | ---: |
| Gold chronological turns | **9/20** | **45%** |
| Extracted unordered records | 5/20 | 25% |
| Executed sorted trace | 7/20 | 35% |
| Executed corrupted trace | 6/20 | 30% |

Sorted versus unordered records yields 2 rescues and 0 damages, a +10-point
effect. However, sorted execution is 10 points below the strongest control,
so the pre-registered verdict is dead.

## Manual audit

One sorted answer (`beam-128k-8-event_ordering-002`) appears semantically
correct despite a negative judge label. Correcting it manually changes sorted
accuracy to 8/20 (40%), still below gold chronological evidence at 9/20. No
scientific verdict changes.

The corrupted condition remains unexpectedly strong because the records retain
their source-turn IDs. On several rows the reader reconstructs chronology from
those IDs instead of trusting the displayed candidate order. The corruption arm
therefore does not establish causal use of the executed numbering.

## Failure mechanism: the execution unit is wrong

The compiler emits one event record per labeled gold turn. That assumption is
structurally false on this cohort:

- 11/20 references contain a different number of ordered answer items than the
  number of labeled gold turns.
- Several turns contain multiple answer items; some rows require omitting or
  merging labeled turns.
- For example, `beam-128k-1-event_ordering-002` has three labeled turns but a
  five-item reference. The final turn contains multiple deployment and testing
  developments. A one-record-per-turn compiler collapses them before sorting.

Consequently, deterministic source-turn ordering executes the easy part while
query-conditioned event segmentation remains unsolved. Information lost during
record construction cannot be recovered by the executor.

## Binding conclusion

The cheap version of “compiler as executor” is not a viable paper direction.
Making it viable would require an event-level planner that can emit multiple
within-turn events, infer their local order, and select the requested count.
That is a new semantic-parsing method rather than a deterministic operator
ablation, and would need a fresh gate rather than post-hoc expansion of this
negative pilot.

Artifacts: `compiled.json`, `answers.json`, `judgments.json`, `analysis.json`,
the frozen design in `PRE_REGISTRATION.md`, implementation in
`analysis/event_order_executor.py` and `cli/event_order_executor_run.py`.
