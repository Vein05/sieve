

## LLM Judge Summary (v2)

- **Judge Model**: `deepseek/deepseek-chat-v3-0324` via `openrouter`
- **Rubric**: 0 = incorrect, 1 = partially correct, 2 = fully correct

| system | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive | constrained_pairwise_vs_naive |
| --- | ---: | ---: | ---: | ---: | ---: |
| `interrogator_v0::ceiling::meta-llama/llama-3.1-8b-instruct` | 0.125 | 0.037 | 0.087 | 0.500 | 0.500 |

### By Category

| system | category | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive |
| --- | --- | ---: | ---: | ---: | ---: |
| `interrogator_v0::ceiling::meta-llama/llama-3.1-8b-instruct` | beam | 0.125 | 0.037 | 0.087 | 0.500 |
