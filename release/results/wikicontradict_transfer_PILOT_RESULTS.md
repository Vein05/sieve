# WikiContradict transfer test v0 (2026-07-23, ~$0.05, deterministic scoring)

**Question:** does deterministic conflict-marked read-time rendering beat
conflict-aware *prompting* on a benchmark other than BEAM, and do they compose?
This is the cheap kill-test for whether the BEAM 0->33->50 chain-rendering
result transfers off its home benchmark.

**Verdict: the strong claim FAILS on WikiContradict.** Marks and prompts are
substitutes, not complements. Prompting alone >= rendering alone, and adding
marks on top of the prompt adds nothing (-0.2pp). The specific wedge -- that
evidence-side conflict *marks* are load-bearing -- does not survive contact with
real encyclopedic contradictions.

## Setup

- 253 IBM WikiContradict RAG-QA rows (all pass the deterministic validity gate).
- 2x2: render {plain, marked} x prompt {standard, aware}, 2 readers
  (Llama-3.1-8B, Qwen-2.5-7B), temp 0, max_tokens 512, `raw_answer` (not the
  last-line shim). 2,024 calls, 0 failures.
- `marked` = deterministic [CLAIM A]/[CLAIM B]/[CONTRADICTION] template (no LLM
  compiler). `aware` = WikiContradict-Template-5-style "surface every conflicting
  viewpoint, state that they disagree."
- Scoring is deterministic: both answer spans present (word-boundary) AND a
  lexical conflict flag. Judge deferred.

## Results (pooled, n=506 per condition)

| condition | flag_rate | both_present | det_correct |
| --- | ---: | ---: | ---: |
| plain+standard | 6.3% | 4.3% | 1.6% |
| plain+aware (the baseline to beat) | 100.0% | 23.7% | 23.7% |
| marked+standard (rendering alone) | 18.2% | 11.1% | 4.9% |
| marked+aware (both) | 99.8% | 23.5% | 23.5% |

- **rendering vs prompting: -18.8pp det / -12.6pp both.** Prompting wins big.
- **compose(both) vs prompting: -0.2pp.** No stacking. This is the decisive,
  metric-robust result: if marks added anything the prompt doesn't, marked+aware
  would exceed plain+aware. It does not.
- **marks do help in the no-prompt condition:** plain+standard -> marked+standard
  roughly doubles every metric. Real, but only relevant if you can't/won't set
  the prompt -- not a realistic deployment.

## Two caveats that do NOT rescue the strong claim

1. **The deterministic scorer undercounts paraphrase.** marked+standard readers
   answer tersely and paraphrase numeric/date/unit values (e.g. Q16 correctly
   says "6'8\" per Claim A, 6'9\" per Claim B, contradictory" but the spans don't
   substring-match). A judge would lift marked+standard's absolute numbers. But
   it cannot reverse the no-stacking result (marked+aware vs plain+aware), which
   uses identical scoring on both sides -- and that is the decisive comparison.
2. **`aware` flag_rate is 100% by construction** (the prompt orders the reader to
   say "disagree"), so flag_rate is not a fair cross-prompt signal; both_present
   is the honest metric, and it still favors prompting.

## Why this doesn't match BEAM (the important part)

On BEAM the 0->33->50 ladder was measured against **resolve-to-latest, which
DELETES one side** of the contradiction; co-presence was the binding constraint.
WikiContradict **always gives both passages**, so it structurally cannot test the
resolve-vs-preserve axis -- only mark-vs-no-mark *given* co-presence. On that
narrower axis, a prompt substitutes for the mark.

Consequence: the defensible cross-benchmark claim narrows from "conflict-marked
rendering is a new evidence-side capability" to "write-time resolution deletes a
side and *that* is the whole problem; any read-time preservation (marks OR a
prompt) fixes it." That is real but smaller, and WikiContradict/DRAGged already
show prompting works -- so it is contested ground, not open ground.

## The one angle still open (untested here)

WikiContradict flags every instance as a known conflict and blanket-instructs the
reader to surface disagreement. Real RAG pipelines see mostly non-conflict
queries, where always-on conflict-prompting may over-flag and hurt. Evidence-side
marking fired only when a *detector* sees a conflict could beat always-on
prompting on a mixed conflict/no-conflict set. That needs a detector + a
no-conflict control cohort -- a different experiment, not a rescue of this one.

Artifacts: `answers.json`, `cells.json`, harness in `analysis/wikicontradict.py`,
`cli/wikicontradict_transfer_run.py`, `analysis/wikicontradict_report.py`,
tests in `tests/analysis/test_wikicontradict.py`.
