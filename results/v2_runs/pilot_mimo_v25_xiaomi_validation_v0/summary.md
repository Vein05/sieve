

## LLM Judge Summary (v2)

- **Judge Model**: `deepseek/deepseek-chat-v3-0324` via `openrouter`
- **Rubric**: 0 = incorrect, 2 = correct (binary, Wang et al.)

| system | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive | constrained_pairwise_vs_naive |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2_response_surface::xiaomi/mimo-v2.5::raw::0` | 0.703 | 0.352 | 0.352 | 0.500 | 0.500 |
| `v2_response_surface::xiaomi/mimo-v2.5::raw::400` | 0.747 | 0.374 | 0.374 | 0.500 | 0.500 |
| `v2_response_surface::xiaomi/mimo-v2.5::selected_extractive::400` | 0.549 | 0.275 | 0.275 | 0.500 | 0.500 |
| `v2_response_surface::xiaomi/mimo-v2.5::selected_raw::400` | 0.571 | 0.286 | 0.286 | 0.500 | 0.500 |
| `v2_response_surface::xiaomi/mimo-v2.5::structured::400` | 0.901 | 0.451 | 0.451 | 0.500 | 0.500 |
| `v2_response_surface::xiaomi/mimo-v2.5::structured_quotes::400` | 0.879 | 0.440 | 0.440 | 0.500 | 0.500 |

### By Category

| system | category | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive |
| --- | --- | ---: | ---: | ---: | ---: |
| `v2_response_surface::xiaomi/mimo-v2.5::raw::0` | lmf500 | 0.703 | 0.352 | 0.352 | 0.500 |
| `v2_response_surface::xiaomi/mimo-v2.5::raw::400` | lmf500 | 0.747 | 0.374 | 0.374 | 0.500 |
| `v2_response_surface::xiaomi/mimo-v2.5::selected_extractive::400` | lmf500 | 0.549 | 0.275 | 0.275 | 0.500 |
| `v2_response_surface::xiaomi/mimo-v2.5::selected_raw::400` | lmf500 | 0.571 | 0.286 | 0.286 | 0.500 |
| `v2_response_surface::xiaomi/mimo-v2.5::structured::400` | lmf500 | 0.901 | 0.451 | 0.451 | 0.500 |
| `v2_response_surface::xiaomi/mimo-v2.5::structured_quotes::400` | lmf500 | 0.879 | 0.440 | 0.440 | 0.500 |
