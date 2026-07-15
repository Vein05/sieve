# SIEVE v2: ICLR Assessment and Execution Plan

*Compiled 2026-07-13 from a deep read of 12 papers (research/papers/), the sieve v1 codebase, the ARR 12575 review cycle, and a fresh novelty scan.*

## Verdict

Target: **7-8 at ICLR 2027**. Achievable, not guaranteed. The decisive framing upgrade: the prior paper (arXiv 2606.21807) contains an **impossibility result** that v2 answers. That converts "we built an adaptive compressor" (6, borderline) into "we proved reader-blind compression has a ceiling, then built the first system that breaks it" (7-8 territory). ICLR is a better fit than ICML (method is deliberately simple; ICLR rewards strong empirical LLM-systems work, and LongMemEval + BEAM are both ICLR papers).

## The spine: impossibility -> construction -> restoration

1. **Impossibility (prior paper, cite):** Fixed compression collapses reader scaling (SIEVE r = -0.887 on LongMemEval-S). Table 10: a reader-blind per-row oracle captures only **70% of routing headroom** because **37% of rows are mixed-label** (same evidence helps some readers, hurts others). A learned reader-blind router gains only +1.4pp and does not change the slope (r = -0.855 vs -0.854). The remaining 30% is inaccessible without reader identity.
2. **Construction (v2):** capability profile z_r from a ~50-100 query calibration battery (no model IDs, no white-box access) drives a policy pi(style, budget | query features, z_r) over a 6-style representation taxonomy. v1 structured compiler is retained as one expert.
3. **Restoration (headline figure):** compression that preserves reader scaling. Figure 1: gain-vs-baseline slope r = -0.89 (fixed) vs ~0 (adaptive); upgrade retention rho restored 0.20 -> ~0.9; pairwise ranking flips 31% -> near zero. Only we can compute these metrics (ragscale 176,864-row matrix).

## Evidence inventory (from the July 2026 agent sweep)

### From our own paper (arXiv 2606.21807, "Fixed RAG Compression Collapses Measured Reader Scaling")
- 20 readers, 12 families; LongMemEval-S primary; HotpotQA/MuSiQue/NQ/QMSum transfer. 176,864-row interaction matrix.
- SIEVE deltas: Llama-3.1-8B +13.2pp (naive 27.8%); GPT-4.1-mini +9.6pp; Phi-4 +0.0pp; DS-R1-70B -2.0pp; Seed-2.0-Mini -4.0pp (LLM-Sum). Linear-fit crossover at ~59% naive baseline; DS-V4-Flash (held-out) crossed to net-negative as predicted.
- Rescue:damage ratio 3.1:1 weakest reader -> 1.1:1 strongest. SIEVE breaks 10-24% of strong readers' correct answers (692/4510 = 15.3% of naive-correct pairs).
- Reinjection recovers 61% of SIEVE damage, 86-94% of RECOMP damage: damage is mostly recoverable information loss, not distortion. Error taxonomy of C->W rows: missing critical detail 34%, over-compression 25%, temporal supersession 17%, factual distortion 13%, wrong evidence 11%.
- Multi-session rows are retrieval-bound (BM25 top-20 retrieves 3-4 of 5+ sessions); compilation cannot fix them. notes/multi-session-ceiling.md: ~54% hard ceiling; needs write-time fact indexing / cross-session entity linking, not better compression.
- SIEVE@200 (161 tokens) Pareto-dominates naive uncapped (717 tokens) for weak readers: +10-11pp at 78% fewer tokens.
- Judge protocol (reusable): DeepSeek V3 primary, GPT-4o cross-check kappa 0.89-0.92, human kappa 0.76.

### From the benchmarks
- **LongMemEval (ICLR 2025):** oracle-vs-full-history gap 30-55% (GPT-4o 0.870 -> 0.606; Llama-70B 0.744 -> 0.334). CP4: JSON+CoN vs NL+Direct swings up to 10pp at oracle retrieval (GPT-4o 0.924 vs 0.862). Paper explicitly notes "optimal token budget varies by model capability, no single retrieval budget works universally". Hardest: temporal reasoning, multi-session, abstention.
- **BEAM (ICLR 2026):** 100K-10M tokens, 10 abilities, nugget scoring. LIGHT (episodic KV + working memory + scratchpad) beats vanilla by +49% relative at 100K up to +156% at 10M. Contradiction resolution is near-zero for every system (0.00-0.08): open problem, quotable target. K=15 optimal retrieval budget; K=20 degrades (noise sensitivity = compilation quality matters).

### From the compression baselines
- **RECOMP (ICLR 2024):** compressors trained against a fixed reader; transfer to LLaMA-13B degrades. Per-reader oracle definition acknowledges optimal compression is reader-dependent.
- **ECoRAG (ACL Findings 2025):** appendix A.8 = clearest unframed evidence of the phenomenon: compression rescues Llama3 on NQ (0.27 -> 30.22 EM) while GPT-4o-mini does better uncompressed (50.18 raw vs 36.48 ECoRAG).
- **EXIT:** reader-agnostic training; same classifier/threshold for 8B and 70B readers.
- **LLMLingua/LongLLMLingua:** proxy-target "distribution alignment" is static; query-adaptive budgets, never reader-adaptive.
- **ACC-RAG (EMNLP Findings 2025):** sufficiency selector reads one fixed decoder's hidden states (Mistral-7B only); no cross-reader transfer tested.
- **OSCAR:** maximal reader entanglement, retrained per backbone; gains scale with reader size.
- **Nobody** varies reader capability while holding compression fixed, or takes reader identity as an inference-time input.
- Expected ICLR baseline suite: uncompressed RAG, RECOMP-Extr/Abst, LLMLingua-2, LongLLMLingua, CompAct, ECoRAG, EXIT, + closed book.

### Novelty scan (July 2026): NOT scooped
- Closest: **FAVICOMP** (arXiv 2409.12468, EMNLP 2025 Findings): conditions on target model via ensemble decoding, but needs white-box logit access at inference, single familiarity axis, standard QA only. Differentiators to foreground: calibration-set profiling (black-box), multi-style axis, budget co-optimization, conversational memory domain.
- Partial overlap: PoC (arXiv 2603.19733, budget from developer-set performance floor). Adjacent: ATACompressor, AdaComp, ACC-RAG (query/task-adaptive only), ZipRL, "LLM Scaling Paradox in Context Compression" (arXiv 2602.09789, compressor-side corroboration).
- Routing literature reviewers will cite: RouteLLM, INFERENCEDYNAMICS, RouteProfile: they route queries to models; we route representations to readers. Orthogonal intervention.

### ARR 12575 review cycle (prior paper: scores 2 / 3 / 3)
- Excitement ceiling = "diagnostic, no method". v2 is literally the reader-aware router our own TODO listed as the 4.5-soundness stretch goal.
- Sharpest technical attack: **headroom confound** (yo68 W3). Defenses that worked: headroom-normalized gain survives (r = -0.54 LME-S, r = -0.95 HotpotQA); tercile breakdown monotone (0.185/0.098/0.059, Welch t = 7.07); negative normalized gain for strongest HotpotQA tercile; raw-correct-conditioned damage rate r = -0.67. These metrics go in v2 main text from day one.
- Two-force framework attacked as post-hoc. v2 fix: measure B(x) and D(x) independently (D via reinjection, B via distractor-injection perturbations; families already specced in configs/v2_perturbation_spec_v0.json).
- Artifact visibility: foreground ragscale as a dataset release in the main text (two reviewers gave Datasets:1 despite the 176K-row release).

### Codebase inventory (what is reusable)
- **v2-ready:** cli/replay.py (compile-once/replay-many, the key asset); 500-row LongMemEval caches data/compilation_cache/paper_sieve.json + paper_naive.json (fingerprint f68251277315bc7a, 40/500 deterministic); dataset-slices/longmemeval_bm25_top20_v1.jsonl + hotpotqa_distractor_500_v1.jsonl; reader/client.py (ollama + openrouter, seed=42); reader/scoring.py (LongMemEval/LoCoMo/BEAM metrics); cli/judge.py.
- **Missing for v2 (bounded work):** 4 new packagers (filtered-raw, extractive, structured+quotes, LLM-summary); budget parameterization at 200/400/800; perturbation generation scripts; calibration battery. evidence/renderer.py has only one live mode ("structured_json"); extend, don't replace. evidence/pack_builder.py already produces the raw baseline.
- **Caution:** TODO.md P0 bug list references stale paths (assembly_methods/, scoring_judge.py) from a pre-refactor layout. Reverify each P0 against current files before fixing. results/answer_generation_runs/ is empty locally; runs live on the Windows PC (Tailscale).

## Paper design

- **C1 Response surface (phenomenon):** 6 representations x 3 budgets x 500 LongMemEval queries x ~10-12 readers (~96K reader calls; compilation cached once). Deliver: per-(reader, style, budget) accuracy; oracle-adaptive vs best-fixed gap; mixed-label structure at style granularity. Go/no-go gate for everything downstream.
- **C2 Capability profile z_r (the scientific object, main bet):** ~50-100 probe queries over 6 dimensions (distractor tolerance, multi-hop integration, temporal reasoning, paraphrase sensitivity, context depth, abstention calibration). Make-or-break main-text ablation: **z_r vs param count vs naive baseline accuracy vs model-ID one-hot** as conditioning signal. De-risk first via the existing 176K-row matrix: find two readers with equal baselines but different damage profiles (existence proof that a scalar cannot suffice).
- **C3 Router:** small policy (GBM or small MLP), deliberately simple; contribution is the conditioning. Train on N readers, hold out >= 6 across families (prior paper already validated 4 held-out models on the trend line).
- **C4 Restoration:** full 20-reader panel via replay. Report scaling slope, upgrade retention, Kendall tau ranking preservation, per-tier Pareto frontiers (weak keep +11-13pp; strong statistically tied with raw at ~5x fewer tokens). Cross-domain: HotpotQA (slice exists), NQ/MuSiQue via ragscale. Stretch: BEAM-1M subset targeting contradiction resolution (diagnose the retrieval-completeness confound first: v1's 30% vs 54% raw on BEAM may be retrieval failure, not compilation).

## Bets and kill criteria

| Bet | Odds | If it fails |
|---|---|---|
| Oracle-adaptive >> best-fixed on surface | High (37% mixed rows, 15.3% damage, 10pp format swings) | Dies cheaply at week 3; pivot to phenomenon paper |
| z_r beats scalar proxies + transfers to held-out readers | Uncertain: the high-risk/high-reward core | Fall back to baseline-conditioned routing: solid 6 / ACL-main, loses profile novelty |
| BEAM-1M lands in time | Medium (confound must be diagnosed first) | Appendix; LME + HotpotQA + NQ suffice |

Floor: C1 alone with our statistical rigor is a publishable empirical paper. The 7-8 requires bet 2.

## Timeline (ICLR 2027: abstracts ~Sept 19, papers ~Sept 26, 2026)

~9-10 weeks, feasible only because of replay infra:
1. Weeks 1-1.5: reverify + fix P0 bugs (stale paths!), build 4 packagers + budget parameterization; mine 176K matrix for equal-baseline/different-damage pairs.
2. Weeks 2-4: response surface; go/no-go gate.
3. Weeks 4-7: z_r battery, router, held-out transfer, scalar-proxy ablation.
4. Weeks 7-10: 20-reader panel, cross-domain, BEAM stretch, writing (attack-preemption metrics in main text).

## Immediate next actions

1. Reverify TODO.md P0 bugs against the current repo layout (judge dedup collision + ordering-query misclassification would silently corrupt the per-question-type surface).
2. Mine ragscale matrix (Windows PC) for the z_r existence proof: free, and de-risks the main bet before any build.
3. Spec the four packagers against evidence/renderer.py + pack_builder.py interfaces.
4. Write the response-surface driver on top of cli/replay.py.
