# Qwen3.5-397B-A17B Alibaba validation pilot

- Queries: 91 validation examples
- Reader calls: 546
- Model: `qwen/qwen3.5-397b-a17b`
- OpenRouter provider: `alibaba` only; fallbacks disabled
- Serving provenance: 546/546 responses report `Alibaba`
- Quantization: OpenRouter reports `unknown`; no BF16/FP8 claim
- Reasoning: disabled
- Initial parallelism: 32; 36 Alibaba rate limits retried at 8
- Final reader-call result: 546/546 successful
- Reader tokens: 416,674 prompt and 1,217 completion

## Condition accuracy

| Condition | Correct | Accuracy |
| --- | ---: | ---: |
| `raw@0` | 34/91 | 37.4% |
| `raw@400` | 39/91 | 42.9% |
| `selected_raw@400` | 28/91 | 30.8% |
| `selected_extractive@400` | 27/91 | 29.7% |
| `structured@400` | 41/91 | 45.1% |
| `structured_quotes@400` | 39/91 | 42.9% |

## Single-reader surface

- Best fixed: `structured@400` at 45.1%
- Per-query representation oracle: 57.1%
- Oracle gap: +12.1pp (query-bootstrap 95% CI +5.5pp to +19.8pp)
- Mixed queries: 36.3%
- Matched-content oracle: 51.6%
- Matched-content oracle gap: +6.6pp
- Matched-content mixed queries: 29.7%

## Structured comparisons

- Structured versus uncapped raw: 15 rescues, 8 damages, net +7.7pp
- Structured versus raw@400: 12 rescues, 10 damages, net +2.2pp

## Interaction with the Qwen3.5-9B pilot

| Condition | Qwen3.5-9B | Qwen3.5-397B |
| --- | ---: | ---: |
| `raw@0` | 36.3% | 37.4% |
| `raw@400` | 38.5% | 42.9% |
| `selected_raw@400` | 37.4% | 30.8% |
| `selected_extractive@400` | 31.9% | 29.7% |
| `structured@400` | 46.2% | 45.1% |
| `structured_quotes@400` | 46.2% | 42.9% |

- Structured-minus-raw@0 interaction: +2.2pp in favor of 9B, 95% CI
  -6.6pp to +11.0pp.
- Structured-minus-raw@400 interaction: +5.5pp in favor of 9B, 95% CI
  -3.3pp to +14.3pp.
- Best shared fixed policy across both readers: `structured@400`, 83/182
  reader-query pairs correct (45.6%).
- Reader-blind per-query oracle: 104/182 (57.1%).
- Per-(query, reader) oracle: 106/182 (58.2%).
- Reader conditioning adds only 2 correct reader-query pairs (+1.1pp) over the
  reader-blind oracle for this two-reader pilot. The reader-blind oracle captures
  21/23 = 91.3% of the headroom above the best shared fixed policy.

## Interpretation

This run does not yet validate the v2 reader-conditioning thesis. Despite its
parameter scale, Qwen3.5-397B is not an empirically strong contrast on this
slice: its uncapped-raw baseline is only +1.1pp above Qwen3.5-9B, and both
readers choose structured@400 as the best fixed representation. The interaction
confidence intervals include zero, and a reader-blind oracle captures almost all
of the two-reader oracle headroom.

The result does validate the representation-surface premise for a second
reader: the 397B oracle gap is +12.1pp and 36.3% of queries have mixed outcomes.
The next useful test is therefore an architecture-different frontier reader with
a materially higher raw baseline, not expansion to the full surface or router
training.
