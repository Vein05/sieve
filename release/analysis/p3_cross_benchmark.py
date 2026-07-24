"""P3: pairwise per-row compression-delta correlations between readers on held-out benchmarks."""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

DEFAULT_PARQUET = "/Users/vein/Documents/research/sieve/data/router_matrix/v0.parquet"
MIN_COMMON_ROWS = 100
MIN_READERS = 3
CORRELATION_THRESHOLD = 0.5
REFERENCE_SLICE = "longmemeval"
STYLE_PAIRS: tuple[tuple[str, str], ...] = (
    ("summary", "raw"),
    ("structured_v1", "raw"),
    ("summary_trained", "raw"),
    ("extractive_trained", "raw"),
)
SLICES: tuple[str, ...] = ("longmemeval", "hotpotqa", "musique", "nq")


def reader_delta_vectors(df: pd.DataFrame, style_a: str, style_b: str) -> dict[str, pd.Series]:
    """Per reader, build the per-example delta vector between two representation styles."""
    out: dict[str, pd.Series] = {}
    for reader, g in df.groupby("reader_model"):
        a = g[g["style"] == style_a].groupby("example_id").correct.mean()
        b = g[g["style"] == style_b].groupby("example_id").correct.mean()
        common = a.index.intersection(b.index)
        if len(common) >= MIN_COMMON_ROWS:
            out[reader] = (a.loc[common] - b.loc[common]).sort_index()
    return out


def pairwise_corrs(vectors: dict[str, pd.Series]) -> list[float]:
    """Pearson correlation of delta vectors for every reader pair clearing the gates."""
    corrs: list[float] = []
    for r1, r2 in itertools.combinations(sorted(vectors), 2):
        common = vectors[r1].index.intersection(vectors[r2].index)
        if len(common) < MIN_COMMON_ROWS:
            continue
        v1, v2 = vectors[r1].loc[common], vectors[r2].loc[common]
        if v1.std() == 0 or v2.std() == 0:
            continue
        corrs.append(float(np.corrcoef(v1, v2)[0, 1]))
    return corrs


def build_report(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pairwise correlation summaries over every slice and style pair."""
    rows: list[dict[str, object]] = []
    for slice_name in SLICES:
        sub = df[df.slice_name == slice_name]
        for style_a, style_b in STYLE_PAIRS:
            vectors = reader_delta_vectors(sub, style_a, style_b)
            if len(vectors) < MIN_READERS:
                continue
            corrs = pairwise_corrs(vectors)
            if not corrs:
                continue
            rows.append({
                "slice": slice_name,
                "pair": f"{style_a}-{style_b}",
                "readers": len(vectors),
                "reader_pairs": len(corrs),
                "mean_r": np.mean(corrs),
                "median_r": np.median(corrs),
                "min_r": np.min(corrs),
                "max_r": np.max(corrs),
                "frac_below_0.5": np.mean(np.array(corrs) < CORRELATION_THRESHOLD),
            })
    return pd.DataFrame(rows)


def load_matrix(parquet_path: str) -> pd.DataFrame:
    """Load uncapped, labeled cells from the router matrix parquet."""
    df = pd.read_parquet(parquet_path)
    return df[df.budget.isna() & df.correct.notna()]


def main(parquet_path: str = DEFAULT_PARQUET) -> None:
    """Compute and print the P3 cross-benchmark correlation report and verdict."""
    df = load_matrix(parquet_path)
    report = build_report(df)
    pd.set_option("display.width", 200)
    print(report.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    held_out = report[report.slice != REFERENCE_SLICE]
    pooled_below = held_out.reader_pairs.mul(held_out["frac_below_0.5"]).sum() / held_out.reader_pairs.sum()
    print(f"\nP3 verdict (threshold: r < {CORRELATION_THRESHOLD}): max pairwise mean over held-out "
          f"benchmarks = {held_out.mean_r.max():.3f}; max single pair = {held_out.max_r.max():.3f}; "
          f"fraction of all held-out pairs below {CORRELATION_THRESHOLD} = {pooled_below:.3f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET, help="path to the router matrix parquet")
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().parquet)
