

## LLM Judge Summary (v2)

- **Judge Model**: `deepseek/deepseek-chat-v3-0324` via `openrouter`
- **Rubric**: 0 = incorrect, 1 = partially correct, 2 = fully correct

| system | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive | constrained_pairwise_vs_naive |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::extractive::400` | 1.460 | 0.680 | 0.780 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::extractive::400` | 1.470 | 0.670 | 0.800 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::extractive::400` | 1.400 | 0.650 | 0.750 | 0.500 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::filtered_raw::400` | 1.540 | 0.720 | 0.820 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::filtered_raw::400` | 1.520 | 0.710 | 0.810 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::filtered_raw::400` | 1.360 | 0.640 | 0.720 | 0.500 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::guarded_structured::400` | 1.330 | 0.620 | 0.710 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::guarded_structured::400` | 1.410 | 0.650 | 0.760 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::guarded_structured::400` | 1.390 | 0.650 | 0.740 | 0.500 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::raw::0` | 1.520 | 0.710 | 0.810 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::raw::0` | 1.530 | 0.710 | 0.820 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::raw::0` | 1.450 | 0.680 | 0.770 | 0.500 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::raw::400` | 1.490 | 0.700 | 0.790 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::raw::400` | 1.480 | 0.690 | 0.790 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::raw::400` | 1.390 | 0.650 | 0.740 | 0.500 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::structured_generic::400` | 1.310 | 0.620 | 0.690 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::structured_generic::400` | 1.380 | 0.640 | 0.740 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::structured_generic::400` | 1.360 | 0.640 | 0.720 | 0.500 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::summary::400` | 1.190 | 0.540 | 0.650 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::summary::400` | 1.240 | 0.570 | 0.670 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::summary::400` | 1.260 | 0.580 | 0.680 | 0.500 | 0.500 |

### By Category

| system | category | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive |
| --- | --- | ---: | ---: | ---: | ---: |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::extractive::400` | convomem | 1.460 | 0.680 | 0.780 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::extractive::400` | convomem | 1.470 | 0.670 | 0.800 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::extractive::400` | convomem | 1.400 | 0.650 | 0.750 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::filtered_raw::400` | convomem | 1.540 | 0.720 | 0.820 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::filtered_raw::400` | convomem | 1.520 | 0.710 | 0.810 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::filtered_raw::400` | convomem | 1.360 | 0.640 | 0.720 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::guarded_structured::400` | convomem | 1.330 | 0.620 | 0.710 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::guarded_structured::400` | convomem | 1.410 | 0.650 | 0.760 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::guarded_structured::400` | convomem | 1.390 | 0.650 | 0.740 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::raw::0` | convomem | 1.520 | 0.710 | 0.810 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::raw::0` | convomem | 1.530 | 0.710 | 0.820 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::raw::0` | convomem | 1.450 | 0.680 | 0.770 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::raw::400` | convomem | 1.490 | 0.700 | 0.790 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::raw::400` | convomem | 1.480 | 0.690 | 0.790 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::raw::400` | convomem | 1.390 | 0.650 | 0.740 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::structured_generic::400` | convomem | 1.310 | 0.620 | 0.690 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::structured_generic::400` | convomem | 1.380 | 0.640 | 0.740 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::structured_generic::400` | convomem | 1.360 | 0.640 | 0.720 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::summary::400` | convomem | 1.190 | 0.540 | 0.650 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::summary::400` | convomem | 1.240 | 0.570 | 0.670 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::summary::400` | convomem | 1.260 | 0.580 | 0.680 | 0.500 |
