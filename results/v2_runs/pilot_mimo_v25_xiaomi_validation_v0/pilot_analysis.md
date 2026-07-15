# MiMo-V2.5 Xiaomi validation pilot

- Queries: 91 validation examples
- Reader calls: 546 at parallelism 32
- Model: `xiaomi/mimo-v2.5`
- OpenRouter provider: `xiaomi` only; fallbacks disabled
- Endpoint quantization: FP8
- Serving provenance: 546/546 responses report `Xiaomi`
- Reasoning: disabled with `reasoning.enabled=false`
- Reader-call result: 546/546 successful
- Reader tokens: 538,215 prompt and 2,027 completion
- Estimated endpoint cost: approximately $0.08

## Condition accuracy

| Condition | Correct | Accuracy |
| --- | ---: | ---: |
| `raw@0` | 32/91 | 35.2% |
| `raw@400` | 34/91 | 37.4% |
| `selected_raw@400` | 26/91 | 28.6% |
| `selected_extractive@400` | 25/91 | 27.5% |
| `structured@400` | 41/91 | 45.1% |
| `structured_quotes@400` | 40/91 | 44.0% |

## Single-reader surface

- Best fixed: `structured@400` at 45.1%
- Per-query representation oracle: 52.7%
- Oracle gap: +7.7pp (query-bootstrap 95% CI +3.3pp to +13.2pp)
- Mixed queries: 28.6%
- Matched-content oracle: 48.4%
- Matched-content oracle gap: +3.3pp
- Matched-content mixed queries: 24.2%

## Structured comparisons

- Structured versus uncapped raw: 14 rescues, 5 damages, net +9.9pp
- Structured versus raw@400: 13 rescues, 6 damages, net +7.7pp

## Interaction with Qwen3.5-9B

| Condition | Qwen3.5-9B | MiMo-V2.5 |
| --- | ---: | ---: |
| `raw@0` | 36.3% | 35.2% |
| `raw@400` | 38.5% | 37.4% |
| `selected_raw@400` | 37.4% | 28.6% |
| `selected_extractive@400` | 31.9% | 27.5% |
| `structured@400` | 46.2% | 45.1% |
| `structured_quotes@400` | 46.2% | 44.0% |

- The structured-minus-raw@0 interaction is exactly 0.0pp, 95% CI -9.9pp
  to +9.9pp.
- The structured-minus-raw@400 interaction is exactly 0.0pp, 95% CI -7.7pp
  to +7.7pp.

## Three-reader routing hierarchy

Across Qwen3.5-9B, Qwen3.5-397B-A17B, and MiMo-V2.5:

- Best shared fixed policy: `structured@400`, 124/273 reader-query pairs
  correct (45.4%).
- Reader-blind per-query oracle: 152/273 (55.7%).
- Per-(query, reader) oracle: 154/273 (56.4%).
- Reader conditioning adds only two correct reader-query pairs (+0.7pp) over
  the reader-blind oracle.
- The reader-blind oracle captures 28/30 = 93.3% of the headroom above the best
  shared fixed policy.

## Interpretation

MiMo-V2.5 is architecture- and vendor-different, but it is not an empirically
strong reader on this slice: its raw baselines are slightly below Qwen3.5-9B.
Structured evidence improves it by the same aggregate margin as the 9B reader.
The third reader therefore strengthens the query-adaptive representation result
but adds no evidence for capability-conditioned representation choice.

This statement is specific to the controlled v2 setup, not a general judgment
of MiMo. The canonical published `paper_mimo25_naive` run scores 52/91 (57.1%)
on these exact validation IDs, versus 32/91 (35.2%) for the new `raw@0` cell.
The new raw prompt has more provider-counted prompt tokens on average and uses
the v2 shared prompt, a direct Xiaomi FP8 endpoint, explicit reasoning disable,
and a stricter abstention instruction. Relative to the canonical answers, the
new raw cell has 3 rescues, 23 damages, and 11 additional `Unknown` responses.
The 21.9pp difference is therefore an execution/presentation shift that must be
diagnosed; it cannot be attributed to model capability alone.

The next reader should be selected by demonstrated raw LongMemEval performance,
not advertised model scale or general capability. Screen future candidates on
the 91 raw cells first, then run the remaining five representations only if the
current controlled raw baseline is materially stronger. Do not expand the full
response surface or train the reader-conditioned router from these three pilots.
