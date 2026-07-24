# Format-lever probe v0 (2026-07-23, ~$0.30, direct-answer, providers locked)

**Question:** does evidence *format* alone move readers (content held
byte-identical), and is any effect a big untapped lever or a small model trait?
Motivated by the matrix-mining caveat that a ~20pp MiMo swing and 10pp LME
JSON-vs-NL swing hinted format might carry ~20pp of unexploited headroom.

**Verdict: DEAD as a paper-sized lever.** Format sensitivity is at the noise
floor (~2-6pp best-vs-worst, 3 of 4 readers' gaps include 0), and it is a weak
crossover (readers disagree on best format). The ~20pp hint was the span-prompt
artifact, not a format effect. This joins compression-style, reader-identity,
query-regime, and acquisition as the fifth axis of post-retrieval
re-representation to land in the same 2-4pp, hard-to-condition regime.

## Setup

- 100 LongMemEval rows; gold answer-bearing unit + top distractors selected as a
  fixed 5-unit set, rendered in 5 syntactic formats (prose / numbered / json /
  table / headers) that differ ONLY in wrapping -- unit strings byte-identical
  across formats (test-pinned). Order + instruction held constant.
- 4 readers, first-party providers LOCKED (allow_fallbacks=false): MiMo-v2.5 +
  MiMo-v2.5-pro (Xiaomi), DeepSeek-V4-Flash (DeepSeek), MiniMax-M3 (Minimax).
- Reasoning DISABLED (probe raw direct-answer format sensitivity), temp 0,
  max_tokens 512. 2,000 calls, 0 failures. Lenient token-bounded answer match.

## Results (accuracy per reader x format, n=100 single sample)

| reader | prose | numbered | json | table | headers | swing | best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek-v4-flash | 49 | 50 | 49 | 53 | 53 | 4.0 | table |
| minimax-m3 | 54 | 56 | 58 | 60 | 59 | 6.0 | table |
| mimo-v2.5 | 56 | 55 | 54 | 53 | 56 | 3.0 | prose |
| mimo-v2.5-pro | 56 | 57 | 56 | 57 | 58 | 2.0 | headers |

Format means (reader-avg): prose 53.8 / numbered 54.5 / json 54.2 / table 55.8 /
headers 56.5. Universal spread 2.8pp; mean within-reader swing 3.8pp.

## Significance (paired bootstrap over rows, each reader's own best-vs-worst)

This is a selection-biased UPPER bound (best/worst chosen post hoc), yet:

| reader | gap | 95% CI |
| --- | ---: | --- |
| deepseek-v4-flash | 4.0pp | [-1.0, 10.0]  (includes 0) |
| minimax-m3 | 6.0pp | [1.0, 12.0]  (marginal, inflated) |
| mimo-v2.5 | 3.0pp | [0.0, 7.0]  (touches 0) |
| mimo-v2.5-pro | 2.0pp | [0.0, 5.0]  (touches 0) |

The true (unbiased) effect is smaller than these. Format sensitivity is
statistically indistinguishable from noise for 3 of 4 readers.

## Caveats (none rescue it)

1. Single sample/cell, n=100 -- but the effect is so small it's at the noise
   floor regardless; 3-sample would not turn 3pp into a paper.
2. LongMemEval only, answerability controlled (gold present). Larger/noisier
   evidence might raise format effects, but that reintroduces the lost-in-middle
   confound -- i.e. it would no longer be a pure format effect.
3. Mid/strong reasoning-capable readers only; weak readers untested. But the
   MiMo capability pair (v2.5 vs v2.5-pro) shows no capability trend in swing.
4. Reasoning disabled. If anything, reasoning would launder format further, not
   amplify it.

## Consequence for the paper

The "format is the big untapped lever" hypothesis (Direction B) is falsified.
Combined with the prior negatives, the through-line is now overdetermined:
**post-retrieval re-representation is a second-order effect (2-4pp) on every axis
tested; the first-order levers are evidence presence (retrieval) and reader
capability.** Do not build a paper on adaptive representation. Either commit to
that unifying second-order finding as the thesis (measurement genre -- their
known ceiling), or pivot to a first-order lever.

Artifacts: `answers.json`, `cells.json`, harness in `analysis/format_lever.py`,
`cli/format_lever_run.py`, `analysis/format_lever_report.py`, tests in
`tests/analysis/test_format_lever.py`.
