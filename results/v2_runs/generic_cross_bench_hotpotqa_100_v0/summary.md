

## LLM Judge Summary (v2)

- **Judge Model**: `deepseek/deepseek-chat-v3-0324` via `openrouter`
- **Rubric**: 0 = incorrect, 1 = partially correct, 2 = fully correct

| system | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive | constrained_pairwise_vs_naive |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::extractive::400` | 1.180 | 0.530 | 0.650 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::extractive::400` | 1.180 | 0.540 | 0.640 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::extractive::400` | 1.010 | 0.470 | 0.540 | 0.500 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::filtered_raw::400` | 0.760 | 0.330 | 0.430 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::filtered_raw::400` | 0.760 | 0.350 | 0.410 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::filtered_raw::400` | 0.610 | 0.270 | 0.340 | 0.500 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::guarded_structured::400` | 0.800 | 0.330 | 0.470 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::guarded_structured::400` | 0.810 | 0.370 | 0.440 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::guarded_structured::400` | 0.600 | 0.260 | 0.340 | 0.500 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::raw::0` | 1.150 | 0.520 | 0.630 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::raw::0` | 1.510 | 0.690 | 0.820 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::raw::0` | 1.140 | 0.530 | 0.610 | 0.500 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::raw::400` | 0.730 | 0.320 | 0.410 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::raw::400` | 0.690 | 0.310 | 0.380 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::raw::400` | 0.540 | 0.240 | 0.300 | 0.500 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::structured_generic::400` | 0.670 | 0.300 | 0.370 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::structured_generic::400` | 0.580 | 0.260 | 0.320 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::structured_generic::400` | 0.500 | 0.220 | 0.280 | 0.500 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::summary::400` | 1.420 | 0.640 | 0.780 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::summary::400` | 1.430 | 0.670 | 0.760 | 0.500 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::summary::400` | 1.430 | 0.660 | 0.770 | 0.500 | 0.500 |

### By Category

| system | category | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive |
| --- | --- | ---: | ---: | ---: | ---: |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::extractive::400` | hpqa | 1.180 | 0.530 | 0.650 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::extractive::400` | hpqa | 1.180 | 0.540 | 0.640 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::extractive::400` | hpqa | 1.010 | 0.470 | 0.540 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::filtered_raw::400` | hpqa | 0.760 | 0.330 | 0.430 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::filtered_raw::400` | hpqa | 0.760 | 0.350 | 0.410 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::filtered_raw::400` | hpqa | 0.610 | 0.270 | 0.340 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::guarded_structured::400` | hpqa | 0.800 | 0.330 | 0.470 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::guarded_structured::400` | hpqa | 0.810 | 0.370 | 0.440 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::guarded_structured::400` | hpqa | 0.600 | 0.260 | 0.340 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::raw::0` | hpqa | 1.150 | 0.520 | 0.630 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::raw::0` | hpqa | 1.510 | 0.690 | 0.820 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::raw::0` | hpqa | 1.140 | 0.530 | 0.610 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::raw::400` | hpqa | 0.730 | 0.320 | 0.410 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::raw::400` | hpqa | 0.690 | 0.310 | 0.380 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::raw::400` | hpqa | 0.540 | 0.240 | 0.300 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::structured_generic::400` | hpqa | 0.670 | 0.300 | 0.370 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::structured_generic::400` | hpqa | 0.580 | 0.260 | 0.320 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::structured_generic::400` | hpqa | 0.500 | 0.220 | 0.280 | 0.500 |
| `v2_response_surface::meta-llama/llama-3.1-8b-instruct::summary::400` | hpqa | 1.420 | 0.640 | 0.780 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-72b-instruct::summary::400` | hpqa | 1.430 | 0.670 | 0.760 | 0.500 |
| `v2_response_surface::qwen/qwen-2.5-7b-instruct::summary::400` | hpqa | 1.430 | 0.660 | 0.770 | 0.500 |
