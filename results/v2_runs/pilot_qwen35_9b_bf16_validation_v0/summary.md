

## LLM Judge Summary (v2)

- **Judge Model**: `deepseek/deepseek-chat-v3-0324` via `openrouter`
- **Rubric**: 0 = incorrect, 2 = correct (binary, Wang et al.)

| system | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive | constrained_pairwise_vs_naive |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2_response_surface::qwen/qwen3.5-9b::raw::0` | 0.725 | 0.363 | 0.363 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-9b::raw::400` | 0.769 | 0.385 | 0.385 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-9b::selected_extractive::400` | 0.637 | 0.319 | 0.319 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-9b::selected_raw::400` | 0.747 | 0.374 | 0.374 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-9b::structured::400` | 0.923 | 0.462 | 0.462 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-9b::structured_quotes::400` | 0.923 | 0.462 | 0.462 | 0.500 | 0.500 |

### By Category

| system | category | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive |
| --- | --- | ---: | ---: | ---: | ---: |
| `v2_response_surface::qwen/qwen3.5-9b::raw::0` | lmf500 | 0.725 | 0.363 | 0.363 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-9b::raw::400` | lmf500 | 0.769 | 0.385 | 0.385 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-9b::selected_extractive::400` | lmf500 | 0.637 | 0.319 | 0.319 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-9b::selected_raw::400` | lmf500 | 0.747 | 0.374 | 0.374 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-9b::structured::400` | lmf500 | 0.923 | 0.462 | 0.462 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-9b::structured_quotes::400` | lmf500 | 0.923 | 0.462 | 0.462 | 0.500 |
