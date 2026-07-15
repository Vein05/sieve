

## LLM Judge Summary (v2)

- **Judge Model**: `deepseek/deepseek-chat-v3-0324` via `openrouter`
- **Rubric**: 0 = incorrect, 1 = partially correct, 2 = fully correct

| system | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive | constrained_pairwise_vs_naive |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::summary::400` | 1.190 | 0.540 | 0.650 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::summary::400` | 1.260 | 0.580 | 0.680 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::summary::400` | 1.240 | 0.570 | 0.670 | 0.500 | 0.500 |

### By Category

| system | category | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive |
| --- | --- | ---: | ---: | ---: | ---: |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::summary::400` | convomem | 1.190 | 0.540 | 0.650 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::summary::400` | convomem | 1.260 | 0.580 | 0.680 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::summary::400` | convomem | 1.240 | 0.570 | 0.670 | 0.500 |
