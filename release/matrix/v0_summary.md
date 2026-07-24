# Router Matrix v0 Summary

Generated: 2026-07-14T01:19:51.910131Z


## Ingestion Stats
- Total run dirs scanned: 546
- Dirs skipped (no judge_run_v2.json): 55
- Runs loaded: 491
- Raw rows before dedup: 219965
- Rows after dedup: 191277
- Duplicates removed: 28688
- Disagreement pairs (judge noise estimate): 4789
- Disagreement rate: 0.0250

## Rows per Slice
- convomem: 7,500
- hotpotqa: 45,500
- locomo: 5,000
- longmemeval: 64,000
- longmemeval_dense: 10,296
- longmemeval_oracle: 3,000
- musique: 20,000
- nq: 25,920
- qmsum: 9,760
- reinjection_hotpotqa: 301

## Reader Models
Total distinct reader_models: 27
- meta-llama/llama-3.1-70b-instruct: 16,821
- google/gemma-3-12b-it: 15,220
- qwen/qwen-2.5-7b-instruct: 15,220
- meta-llama/llama-3.1-8b-instruct: 13,784
- qwen/qwen-2.5-72b-instruct: 12,822
- openai/gpt-4.1-mini: 10,318
- allenai/olmo-3.1-32b-instruct: 10,220
- anthropic/claude-3.5-haiku: 8,720
- x-ai/grok-4.1-fast: 8,220
- cohere/command-r-08-2024: 7,720
- google/gemma-3-27b-it: 7,720
- microsoft/phi-4: 7,220
- meta-llama/llama-3.3-70b-instruct: 6,784
- deepseek/deepseek-r1-distill-llama-70b: 6,284
- z-ai/glm-4-32b: 6,284
- qwen/qwen3-14b: 6,284
- qwen/qwen3-32b: 6,284
- qwen/qwen3-8b: 6,284
- bytedance-seed/seed-2.0-mini: 6,284
- meta-llama/llama-4-scout-17b-16e-instruct: 3,312
- meta-llama/llama-4-scout: 2,972
- xiaomi/mimo-v2.5: 1,500
- qwen/qwen3.6-27b: 1,500
- google/gemma-3-4b-it: 1,000
- openai/gpt-4o-mini: 1,000
- meta-llama/llama-3.2-3b-instruct: 1,000
- x-ai/grok-3-mini-beta: 500

## Style Coverage (rows per style)
- raw: 68,429
- summary: 45,360
- structured_v1: 28,148
- summary_trained: 17,480
- extractive_trained: 16,480
- token_pruned: 10,500
- qmsum_naive: 4,880

## Per-(Style x Slice) Run Coverage Grid
| style | convomem | hotpotqa | locomo | longmemeval | longmemeval_dense | longmemeval_oracle | musique | nq | qmsum | reinjection_hotpotqa |
|---|---|---|---|---|---|---|---|---|---|---|
| extractive_trained | 0 | 10000 | 0 | 0 | 0 | 0 | 0 | 6480 | 0 | 0 |
| qmsum_naive | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4880 | 0 |
| raw | 2500 | 15500 | 2500 | 24500 | 5148 | 1500 | 10000 | 6480 | 0 | 301 |
| structured_v1 | 2500 | 0 | 2500 | 16500 | 5148 | 1500 | 0 | 0 | 0 | 0 |
| summary | 2500 | 10000 | 0 | 11500 | 0 | 0 | 10000 | 6480 | 4880 | 0 |
| summary_trained | 0 | 10000 | 0 | 1000 | 0 | 0 | 0 | 6480 | 0 | 0 |
| token_pruned | 0 | 0 | 0 | 10500 | 0 | 0 | 0 | 0 | 0 | 0 |

## Label Balance
- Correct (1): 86,957
- Incorrect (0): 83,287
- QMSum (NaN, excluded from binary): 9,760
- Missing label: 11,273

## Unknown Systems Encountered
- (none)

## Per-(reader_model, style) Accuracy on LongMemEval
| reader_model | style | n | accuracy |
|---|---|---|---|
| google/gemma-3-12b-it | raw | 2500 | 0.371 |
| meta-llama/llama-3.1-70b-instruct | raw | 2500 | 0.387 |
| meta-llama/llama-3.1-8b-instruct | raw | 2500 | 0.230 |
| qwen/qwen-2.5-7b-instruct | raw | 2500 | 0.285 |
| google/gemma-3-12b-it | structured_v1 | 2000 | 0.503 |
| meta-llama/llama-3.1-70b-instruct | structured_v1 | 2000 | 0.529 |
| qwen/qwen-2.5-7b-instruct | structured_v1 | 2000 | 0.469 |
| allenai/olmo-3.1-32b-instruct | raw | 1500 | 0.240 |
| allenai/olmo-3.1-32b-instruct | token_pruned | 1500 | 0.253 |
| anthropic/claude-3.5-haiku | raw | 1500 | 0.311 |
| google/gemma-3-12b-it | token_pruned | 1500 | 0.354 |
| meta-llama/llama-3.1-70b-instruct | token_pruned | 1500 | 0.391 |
| meta-llama/llama-3.1-8b-instruct | token_pruned | 1500 | 0.223 |
| openai/gpt-4.1-mini | raw | 1500 | 0.301 |
| openai/gpt-4.1-mini | token_pruned | 1500 | 0.357 |
| qwen/qwen-2.5-72b-instruct | raw | 1500 | 0.344 |
| qwen/qwen-2.5-72b-instruct | token_pruned | 1500 | 0.417 |
| qwen/qwen-2.5-7b-instruct | token_pruned | 1500 | 0.291 |
| allenai/olmo-3.1-32b-instruct | structured_v1 | 500 | 0.466 |
| allenai/olmo-3.1-32b-instruct | summary | 500 | 0.528 |
| anthropic/claude-3.5-haiku | structured_v1 | 500 | 0.510 |
| anthropic/claude-3.5-haiku | summary | 500 | 0.532 |
| bytedance-seed/seed-2.0-mini | raw | 500 | 0.566 |
| bytedance-seed/seed-2.0-mini | structured_v1 | 500 | 0.568 |
| bytedance-seed/seed-2.0-mini | summary | 500 | 0.526 |
| cohere/command-r-08-2024 | raw | 500 | 0.386 |
| cohere/command-r-08-2024 | structured_v1 | 500 | 0.508 |
| cohere/command-r-08-2024 | summary | 500 | 0.460 |
| deepseek/deepseek-r1-distill-llama-70b | raw | 500 | 0.534 |
| deepseek/deepseek-r1-distill-llama-70b | structured_v1 | 500 | 0.562 |
| deepseek/deepseek-r1-distill-llama-70b | summary | 500 | 0.530 |
| google/gemma-3-12b-it | summary | 500 | 0.546 |
| google/gemma-3-27b-it | raw | 500 | 0.488 |
| google/gemma-3-27b-it | structured_v1 | 500 | 0.524 |
| google/gemma-3-27b-it | summary | 500 | 0.466 |
| google/gemma-3-4b-it | raw | 500 | 0.364 |
| google/gemma-3-4b-it | structured_v1 | 500 | 0.384 |
| meta-llama/llama-3.1-70b-instruct | summary | 500 | 0.550 |
| meta-llama/llama-3.1-70b-instruct | summary_trained | 500 | 0.262 |
| meta-llama/llama-3.1-8b-instruct | summary | 500 | 0.480 |
| meta-llama/llama-3.1-8b-instruct | summary_trained | 500 | 0.236 |
| meta-llama/llama-3.2-3b-instruct | raw | 500 | 0.212 |
| meta-llama/llama-3.2-3b-instruct | structured_v1 | 500 | 0.262 |
| meta-llama/llama-3.3-70b-instruct | raw | 500 | 0.464 |
| meta-llama/llama-3.3-70b-instruct | structured_v1 | 500 | 0.548 |
| meta-llama/llama-3.3-70b-instruct | summary | 500 | 0.544 |
| meta-llama/llama-4-scout-17b-16e-instruct | raw | 500 | 0.350 |
| meta-llama/llama-4-scout-17b-16e-instruct | structured_v1 | 500 | 0.466 |
| meta-llama/llama-4-scout-17b-16e-instruct | summary | 500 | 0.410 |
| microsoft/phi-4 | raw | 500 | 0.472 |
| microsoft/phi-4 | structured_v1 | 500 | 0.472 |
| microsoft/phi-4 | summary | 500 | 0.436 |
| openai/gpt-4.1-mini | structured_v1 | 500 | 0.534 |
| openai/gpt-4.1-mini | summary | 500 | 0.460 |
| openai/gpt-4o-mini | raw | 500 | 0.400 |
| openai/gpt-4o-mini | structured_v1 | 500 | 0.502 |
| qwen/qwen-2.5-72b-instruct | structured_v1 | 500 | 0.554 |
| qwen/qwen-2.5-72b-instruct | summary | 500 | 0.546 |
| qwen/qwen-2.5-7b-instruct | summary | 500 | 0.524 |
| qwen/qwen3-14b | raw | 500 | 0.432 |
| qwen/qwen3-14b | structured_v1 | 500 | 0.524 |
| qwen/qwen3-14b | summary | 500 | 0.496 |
| qwen/qwen3-32b | raw | 500 | 0.496 |
| qwen/qwen3-32b | structured_v1 | 500 | 0.552 |
| qwen/qwen3-32b | summary | 500 | 0.512 |
| qwen/qwen3-8b | raw | 500 | 0.556 |
| qwen/qwen3-8b | structured_v1 | 500 | 0.584 |
| qwen/qwen3-8b | summary | 500 | 0.510 |
| qwen/qwen3.6-27b | raw | 500 | 0.514 |
| qwen/qwen3.6-27b | structured_v1 | 500 | 0.550 |
| qwen/qwen3.6-27b | summary | 500 | 0.486 |
| x-ai/grok-3-mini-beta | summary | 500 | 0.576 |
| x-ai/grok-4.1-fast | raw | 500 | 0.554 |
| x-ai/grok-4.1-fast | structured_v1 | 500 | 0.562 |
| x-ai/grok-4.1-fast | summary | 500 | 0.534 |
| xiaomi/mimo-v2.5 | raw | 500 | 0.554 |
| xiaomi/mimo-v2.5 | structured_v1 | 500 | 0.556 |
| xiaomi/mimo-v2.5 | summary | 500 | 0.454 |
| z-ai/glm-4-32b | raw | 500 | 0.390 |
| z-ai/glm-4-32b | structured_v1 | 500 | 0.508 |
| z-ai/glm-4-32b | summary | 500 | 0.442 |

## Validation Gate (Computed vs Published)
| condition | metric | computed | published | delta | status |
|---|---|---|---|---|---|
| meta-llama/llama-3.1-8b-instruct / raw | accuracy@longmemeval | 0.278 | 0.278 | +0.000 | OK |
| meta-llama/llama-3.1-8b-instruct / structured_v1 | accuracy@longmemeval | nan | 0.410 | +nan | MISSING |
| openai/gpt-4.1-mini / raw | accuracy@longmemeval | 0.438 | 0.438 | +0.000 | OK |
| openai/gpt-4.1-mini / structured_v1 | accuracy@longmemeval | 0.534 | 0.534 | +0.000 | OK |
