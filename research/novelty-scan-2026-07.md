# Novelty Scan: Reader-Conditioned Evidence Compilation (2026-07-13)

Verdict: **not scooped.** No paper combines (a) calibration-set capability profiling (model-ID-free, black-box), (b) multi-way evidence style selection, (c) joint style + budget adaptation, (d) conversational memory evaluation. Space is crowded with adjacent work; the synthesis is open.

## Closest paper (must cite and differentiate in intro)

- **FAVICOMP** — Familiarity-Aware Evidence Compression for RAG. arXiv 2409.12468, EMNLP 2025 Findings. Compresses evidence to be "familiar" to the target reader via ensemble decoding. Requires white-box token-probability access at inference; single adaptation axis; standard QA only. Our deltas: offline calibration profiling (works on black-box APIs), multi-style taxonomy, budget co-optimization, conversational memory domain.

## Partial overlap

- **PoC** — Performance-Oriented Context Compression. arXiv 2603.19733 (Mar 2026). Developer-set performance floor drives compression ratio via a lightweight predictor. Budget-only, query-level, no capability inference, no style axis.

## Adjacent (cite in related work)

- **"When Less is More: The LLM Scaling Paradox in Context Compression"** — arXiv 2602.09789 (Feb 2026). Compressor-side size-fidelity paradox (knowledge overwriting, semantic drift). Corroborates 2606.21807 from the compressor side; diagnosis only.
- **ATACompressor** — arXiv 2602.03226, SIGIR-AP 2025. Task-aware adaptive rate; not reader-aware.
- **AdaComp** — arXiv 2409.01579. Query-complexity + retrieval-quality adaptive rate.
- **ACC-RAG** — arXiv 2507.22931, EMNLP Findings 2025. Input-complexity adaptive; selector welded to one fixed decoder (Mistral-7B), no cross-reader transfer. (PDF already in papers/.)
- **ZipRL** — arXiv 2605.28069 (May 2026). RL multi-turn compression via Hindsight Response Replay; task-signal adaptive, not reader adaptive.
- **LaRA** — ICML 2025. RAG-vs-long-context routing depends on model capability; routes architecture choice, not evidence representation.

## Routing literature (reviewers will cite; orthogonal but must be addressed)

- **RouteLLM** — arXiv 2406.18665. Canonical capability-based query-to-model routing.
- **INFERENCEDYNAMICS** — arXiv 2505.16303, ACL 2026. Multi-dimensional capability/knowledge profiling for routing; closest to our profiling mechanism.
- **RouteProfile** — arXiv 2605.00180 (May 2026). Graph-based profiling for cold-start routing; structured profiles beat flat ones.
- **Capability Instruction Tuning** — arXiv 2502.17282.
- **"From Sampled Outcomes to Capability Distributions"** — arXiv 2606.06924. Capability distributions over point estimates; applicable to calibration-set profiling.

Framing answer for all routing work: they select *which model processes the query*; we select *which evidence representation a fixed reader receives*. Orthogonal intervention, composable.

## PDFs to download into papers/

- [ ] FAVICOMP: arXiv 2409.12468
- [ ] PoC: arXiv 2603.19733
- [ ] Scaling paradox: arXiv 2602.09789
- [ ] INFERENCEDYNAMICS: arXiv 2505.16303
- [ ] RouteLLM: arXiv 2406.18665
- [ ] LoCoMo (still missing per INDEX.md, wrong PDF was removed): Maharana et al., ACL 2024
- [ ] CompAct (baseline suite expects it): arXiv 2407.09014
- [ ] LLMLingua-2 (used in prior paper): arXiv 2403.12968

## Anticipated reviewer challenges (with answers)

1. "FAVICOMP already adapts to the reader" — it needs white-box access and one axis; we profile offline and navigate a style-budget surface on black-box readers.
2. "PoC already adapts budget" — its floor is a developer-specified scalar, not inferred capability; no style variation.
3. "Why not just route queries to models (RouteLLM)?" — deployment often fixes the reader (cost, privacy, product); we optimize the evidence given the reader, and the two compose.
4. "LLMLingua's distribution alignment is target-aware" — static one-time proxy alignment, model-ID-dependent, ratio-only.
