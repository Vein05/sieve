# Qwen3.5-9B BF16 validation pilot

- Queries: 91 validation examples
- Reader calls: 546 at parallelism 32
- Route: `deepinfra/bf16`, fallbacks disabled, reasoning disabled
- Estimated reader cost: $0.0420

## Condition accuracy

- `raw@0`: 36.3%
- `raw@400`: 38.5%
- `selected_raw@400`: 37.4%
- `selected_extractive@400`: 31.9%
- `structured@400`: 46.2%
- `structured_quotes@400`: 46.2%

## Full surface

- Best fixed: `structured@400` at 46.2%
- Per-query oracle: 59.3%
- Oracle gap: +13.2% (query-bootstrap 95% CI +6.6% to +18.7%)
- Mixed queries: 41.8%

## Matched-content surface

- Best fixed: `structured@400` at 46.2%
- Per-query oracle: 54.9%
- Oracle gap: +8.8% (query-bootstrap 95% CI +3.3% to +13.2%)
- Mixed queries: 30.8%

## Structured versus uncapped raw

- Rescues: 17
- Damages: 8

One truncated unique judgment was rerun at 64 tokens and confirmed the original no label.
