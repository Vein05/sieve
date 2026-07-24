from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.p3_cross_benchmark import (
    MIN_COMMON_ROWS,
    pairwise_corrs,
    reader_delta_vectors,
)


def _cells(reader: str, style: str, ids: list[int], correct: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "reader_model": reader,
        "style": style,
        "example_id": ids,
        "correct": correct,
    })


def test_delta_vector_is_style_a_minus_style_b() -> None:
    ids = list(range(MIN_COMMON_ROWS))
    a = _cells("r1", "summary", ids, [1] * MIN_COMMON_ROWS)
    b = _cells("r1", "raw", ids, [0] * MIN_COMMON_ROWS)
    vectors = reader_delta_vectors(pd.concat([a, b]), "summary", "raw")
    assert set(vectors) == {"r1"}
    assert (vectors["r1"] == 1.0).all()


def test_reader_below_min_common_rows_is_dropped() -> None:
    ids = list(range(MIN_COMMON_ROWS - 1))
    a = _cells("r1", "summary", ids, [1] * len(ids))
    b = _cells("r1", "raw", ids, [0] * len(ids))
    assert reader_delta_vectors(pd.concat([a, b]), "summary", "raw") == {}


def test_reader_at_exactly_min_common_rows_is_kept() -> None:
    ids = list(range(MIN_COMMON_ROWS))
    a = _cells("r1", "summary", ids, [1] * MIN_COMMON_ROWS)
    b = _cells("r1", "raw", ids, [0] * MIN_COMMON_ROWS)
    assert set(reader_delta_vectors(pd.concat([a, b]), "summary", "raw")) == {"r1"}


def test_common_rows_are_intersection_of_ids() -> None:
    a = _cells("r1", "summary", list(range(MIN_COMMON_ROWS + 5)), [1] * (MIN_COMMON_ROWS + 5))
    b = _cells("r1", "raw", list(range(5, MIN_COMMON_ROWS + 5)), [0] * MIN_COMMON_ROWS)
    vectors = reader_delta_vectors(pd.concat([a, b]), "summary", "raw")
    assert list(vectors["r1"].index) == list(range(5, MIN_COMMON_ROWS + 5))


def test_pairwise_corr_of_identical_vectors_is_one() -> None:
    idx = list(range(MIN_COMMON_ROWS))
    vals = np.sin(np.arange(MIN_COMMON_ROWS, dtype=float))
    vectors = {"r1": pd.Series(vals, index=idx), "r2": pd.Series(vals, index=idx)}
    corrs = pairwise_corrs(vectors)
    assert len(corrs) == 1
    assert corrs[0] == pytest.approx(1.0)


def test_pairwise_corr_of_opposite_vectors_is_minus_one() -> None:
    idx = list(range(MIN_COMMON_ROWS))
    vals = np.sin(np.arange(MIN_COMMON_ROWS, dtype=float))
    vectors = {"r1": pd.Series(vals, index=idx), "r2": pd.Series(-vals, index=idx)}
    corrs = pairwise_corrs(vectors)
    assert len(corrs) == 1
    assert abs(corrs[0] - (-1.0)) < 1e-9


def test_zero_variance_vector_is_skipped() -> None:
    idx = list(range(MIN_COMMON_ROWS))
    flat = pd.Series([0.0] * MIN_COMMON_ROWS, index=idx)
    varying = pd.Series(np.sin(np.arange(MIN_COMMON_ROWS, dtype=float)), index=idx)
    assert pairwise_corrs({"r1": flat, "r2": varying}) == []


def test_pair_below_min_common_rows_is_skipped() -> None:
    v1 = pd.Series(np.sin(np.arange(MIN_COMMON_ROWS, dtype=float)), index=range(MIN_COMMON_ROWS))
    v2 = pd.Series(np.cos(np.arange(MIN_COMMON_ROWS - 1, dtype=float)), index=range(MIN_COMMON_ROWS - 1))
    assert pairwise_corrs({"r1": v1, "r2": v2}) == []
