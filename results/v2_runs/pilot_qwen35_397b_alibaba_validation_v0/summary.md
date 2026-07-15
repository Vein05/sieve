

## LLM Judge Summary (v2)

- **Judge Model**: `deepseek/deepseek-chat-v3-0324` via `openrouter`
- **Rubric**: 0 = incorrect, 2 = correct (binary, Wang et al.)

| system | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive | constrained_pairwise_vs_naive |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2_response_surface::qwen/qwen3.5-397b-a17b::raw::0` | 0.747 | 0.374 | 0.374 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-397b-a17b::raw::400` | 0.857 | 0.429 | 0.429 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-397b-a17b::selected_extractive::400` | 0.593 | 0.297 | 0.297 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-397b-a17b::selected_raw::400` | 0.615 | 0.308 | 0.308 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-397b-a17b::structured::400` | 0.901 | 0.451 | 0.451 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-397b-a17b::structured_quotes::400` | 0.857 | 0.429 | 0.429 | 0.500 | 0.500 |

### By Category

| system | category | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive |
| --- | --- | ---: | ---: | ---: | ---: |
| `v2_response_surface::qwen/qwen3.5-397b-a17b::raw::0` | lmf500 | 0.747 | 0.374 | 0.374 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-397b-a17b::raw::400` | lmf500 | 0.857 | 0.429 | 0.429 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-397b-a17b::selected_extractive::400` | lmf500 | 0.593 | 0.297 | 0.297 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-397b-a17b::selected_raw::400` | lmf500 | 0.615 | 0.308 | 0.308 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-397b-a17b::structured::400` | lmf500 | 0.901 | 0.451 | 0.451 | 0.500 |
| `v2_response_surface::qwen/qwen3.5-397b-a17b::structured_quotes::400` | lmf500 | 0.857 | 0.429 | 0.429 | 0.500 |
