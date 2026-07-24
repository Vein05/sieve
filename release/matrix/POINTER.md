# The router matrix parquet

The interaction matrix itself is **not** copied into this artifact. It lives at:

```
data/router_matrix/v0.parquet   (10 MB, tracked by git LFS)
```

with the human-readable summary and validation gate in `v0_summary.md` (copied
alongside this file). It contains 191,277 deduped cells (built 2026-07-14).

Point every analysis script at this path (it is the default):

```bash
python3 analysis/p3_cross_benchmark.py --parquet data/router_matrix/v0.parquet
```

## Provenance: prior-paper substrate vs. this study's extension cells

The matrix ingests the **prior paper's published runs** (Panthi & Abdelfattah,
2026, arXiv:2606.21807 — the `ragscale` dataset). The validation gate in
`v0_summary.md` confirms the ingested cells match that paper's published
numbers. **These cells are the substrate; they are not re-claimed by this
artifact.**

On top of that substrate, this study contributed the following **post-freeze
extension cells** (these, plus the analysis layer, are what this artifact
claims):

- **Added slices:** ConvoMem and LoCoMo; LongMemEval dense and oracle slices.
- **Added readers:** `xiaomi/mimo-v2.5` and `qwen/qwen3.6-27b` (the reader that
  completes the 21-reader panel).
- **Added summary cells:** additional `summary`-style cells used by the
  summary-policy analyses.

The `structured_v1`, `raw`, `summary_trained`, `extractive_trained`, and
`token_pruned` style coverage inherited from the prior paper's runs is
substrate, not an extension. The format-probe cells (2,000-call format-lever
run) are a separate artifact of this study and live with the format-lever
harness, not in this parquet.
