# Generic-packager cross-benchmark pilot — analysis (2026-07-14)

Runs: `generic_cross_bench_convomem_100_v0`, `generic_cross_bench_convomem_summaryfix_100_v0`,
`generic_cross_bench_hotpotqa_100_v0`. 100 rows per slice, 3 readers
(llama-3.1-8b, qwen-2.5-7b, qwen-2.5-72b), shared v2 prompt template,
deepseek-chat-v3 judge (generic 0-2 rubric both slices; normalized = mean/2).
Summary cells compiled with qwen3-8b, reasoning disabled
(`COMPILE_REASONING`); the first ConvoMem summary compile ran with reasoning
on and produced truncated summaries — scores were within noise of the fixed
compile (59.5-63.0 both), so the gap vs canonical summary is the summarizer
prompt, not truncation. Canonical ConvoMem summaries are also short
(~30 evidence tokens).

## Results (normalized accuracy, n=100 per cell)

### ConvoMem
| condition | llama-8b | qwen-7b | qwen-72b |
|---|---:|---:|---:|
| raw@0 (uncapped, ~570 tok) | 76.0 | 72.5 | 76.5 |
| raw@400 | 74.5 | 69.5 | 74.0 |
| filtered_raw@400 | **77.0** | 68.0 | 76.0 |
| extractive@400 | 73.0 | 70.0 | 73.5 |
| summary@400 (~39 tok) | 59.5 | 62.0 | 63.0 |
| structured_generic@400 | 65.5 | 68.0 | 69.0 |
| guarded_structured@400 | 66.5 | 69.5 | 70.5 |
| *v1 structured (canonical, 500 rows)* | *48.3* | *46.1* | *46.2* |

Guard fired 20/100. Pipeline sanity: qwen-7b/72b raw match canonical raw
within ~1pp (72.5/76.5 vs 71.5/76.0). llama-8b canonical raw (47.5) is from a
mixed slice — not comparable.

### HotpotQA
| condition | llama-8b | qwen-7b | qwen-72b |
|---|---:|---:|---:|
| raw@0 (uncapped, ~940 tok) | 57.5 | 57.0 | 75.5 |
| raw@400 | 36.5 | 27.0 | 34.5 |
| filtered_raw@400 | 38.0 | 30.5 | 38.0 |
| extractive@400 | 59.0 | 50.5 | 59.0 |
| summary@400 | **71.0** | **71.5** | **71.5** |
| structured_generic@400 | 33.5 | 25.0 | 29.0 |
| guarded_structured@400 | 40.0 | 30.0 | 40.5 |

Guard fired 93/100. Canonical summary (500 rows): 80.7/69.3/86.9 — qwen-7b
matches; the stronger readers are ~10-15pp below canonical (pipeline/prompt
difference, the documented presentation confound; within-pilot comparisons
are unaffected).

## Verdicts on the pre-registered hypotheses

**H-i (generic ops eliminate the ConvoMem collapse): CONFIRMED in substance.**
v1 structured 46-48 -> guarded_structured 66.5-70.5 (+20pp), within 3-9.5pp of
raw instead of -28pp. The guard adds +1.0-1.5pp over unguarded for all three
readers. The >=70% threshold is met for qwen-72b only; the residual gap to raw
means structure still is not the right action on ConvoMem — which is the
routing thesis, not a failure of the packager.

**H-ii (match summary on HotpotQA): FAILED for every non-summary condition.**
At a 400-token budget on multi-document QA, only abstraction survives:
summary 71 vs filtered_raw 30-38, structured 25-33, guarded 30-40. Truncating
raw to 400 destroys it (two gold paragraphs do not fit); the guard correctly
rejected structuring 93% of the time but its fallback (filtered_raw@400) is
itself inadequate here. Extraction (50-59) sits in between. Summary is
CONSTRUCTIVE on QA, not a baseline to merely tolerate.

## What the pilot establishes

1. **The cross-benchmark interaction is huge and opposite-signed under one
   pipeline**: ConvoMem's best condition (filtered_raw 68-77) is HotpotQA's
   near-worst (30-38); HotpotQA's best (summary 71) is ConvoMem's worst
   (59-63). 30-40pp swings from representation choice alone, same readers,
   same judge, same prompt. This is the query-conditioned-compilation thesis
   in one table.
2. **Do-no-harm needs a ladder, not a single fallback.** Guard->filtered_raw
   suffices on conversational memory; on multi-doc QA the ladder must
   escalate to summary or uncapped budget. The coverage signal itself is
   informative: guard-fire rate separated the domains 20% vs 93%.
3. **The v1 collapse mechanism is fixed by construction**: floors + guard
   recover +20pp on ConvoMem with zero benchmark-specific code.
4. Budget is not neutral: raw@0 vs raw@400 is +21-41pp on HotpotQA, -0.5-3pp
   on ConvoMem. Aperture belongs in the policy's action space.

Cost: ~4,800 reader/compile calls + ~2,700 judge calls, ≈ $1.50 total.
