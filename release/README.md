# Artifact: "Selecting Evidence Representations by Reader Does Not Pay"

Artifact release for the short paper *Selecting Evidence Representations by
Reader Does Not Pay: A Bounded Negative Result for Reader-Conditioned
Compression*.

## What this artifact contains (and only this)

This artifact is **the analysis layer and the extension cells** built for this
paper. It is **not** a re-release of the `ragscale` interaction matrix, which
was released with the prior paper (Panthi & Abdelfattah, 2026,
arXiv:2606.21807) and is used here only as the substrate. The dataset
contribution of that paper is not re-claimed.

Specifically, this artifact claims:

1. **Analysis layer** — offline analysis code that regenerates the paper's
   held-out mechanism results from the router matrix, with tests.
2. **Extension cells** — the additional cells this study contributes on top of
   the prior paper's runs (summary-policy cells, the reader completing the
   21-reader panel, the 2,000-call format probe with its byte-identity-tested
   harness). These are described in `matrix/POINTER.md` and are physically part
   of the shared parquet; their provenance is documented there.

## License

MIT (see `LICENSE`). Applies to the code and result documents in this artifact.
The underlying `ragscale` matrix is governed by the prior paper's release.

## Layout

```
release/
  README.md                  this file
  LICENSE                    MIT
  TODO.md                    honest list of scripts still to re-implement
  analysis/                  copies of the analysis scripts (run against the parquet)
    p3_cross_benchmark.py    P3: cross-benchmark per-row delta correlations
    format_lever.py          format-probe harness (byte-identical content across formats)
    format_lever_report.py   format-probe reporting
    wikicontradict.py        WikiContradict transfer harness
    wikicontradict_report.py WikiContradict transfer reporting
  results/                   frozen result documents
    p3_cross_benchmark_RESULTS.md
    format_lever_PILOT_RESULTS.md
    wikicontradict_transfer_PILOT_RESULTS.md
  matrix/
    v0_summary.md            human-readable summary + validation gate for the matrix
    POINTER.md               where the parquet lives and which cells are extensions
```

## Reproduction

Every analysis in this artifact runs **offline** against the router matrix
parquet at `data/router_matrix/v0.parquet` (10 MB, kept in the repo under git
LFS — see `matrix/POINTER.md`). No API calls, no network.

```bash
# P3 cross-benchmark mechanism analysis (reproduces results/p3_cross_benchmark_RESULTS.md)
python3 analysis/p3_cross_benchmark.py --parquet data/router_matrix/v0.parquet
```

The format-lever and WikiContradict harnesses (`format_lever.py`,
`wikicontradict.py`) are the probe runners for their respective pilots; their
frozen outputs are in `results/`. See each script's module docstring for usage.

All analyses regenerate their tables in ~2 minutes on a laptop.
