# Event-order executor pilot v0 — pre-registration

## Question

Does deterministic execution of `ORDER` over provenance-linked event records
improve BEAM event-ordering accuracy beyond gold-complete chronological evidence
or the same extracted records left unordered?

## Frozen design

- First 20 valid `event_ordering` rows in slice order, with at least two
  answer-bearing turns present in the true store.
- Gold answer-bearing IDs are used only to construct a gold-complete evidence
  ceiling. The compiler never receives the reference answer.
- Qwen-2.5-72B is both compiler and reader, preventing a stronger hidden
  compiler from explaining gains.
- Conditions: gold chronological turns, extracted unordered records,
  deterministically sorted records, and a one-position rotated corruption.
- DeepSeek-V3-0324 judges exact semantic item coverage and order. Gold answers
  are passed through the judge as a validity check.

## Decision rule

- **Live:** sorted accuracy is at least 30% and at least 15 points above the
  better of gold-chronological and extracted-unordered controls.
- **Dead:** sorted accuracy is at most 15% or improves by less than 5 points.
- Otherwise **inconclusive**.

The corrupted trace is a causal diagnostic, not part of the primary gate.

## Operational amendment after incomplete first launch

The first launch produced only 8/20 valid compilations because the first 12
Qwen requests were sent concurrently and exhausted the endpoint rate limit;
all later compilations and all downstream calls succeeded. No prompt, cohort,
condition, model, or decision threshold changed. Compiler parallelism was
reduced from 12 to 4 before the complete rerun.

The second launch produced 17/20 valid compilations. The remaining three are
retried sequentially with an explicit reminder of the already-visible source
IDs. Valid prior compilations are cached; the scientific conditions and gate
remain unchanged.
