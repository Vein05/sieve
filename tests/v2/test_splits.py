"""Tests for held-out query assignment."""

from __future__ import annotations

import pytest

from v2.splits import SplitConfig, split_for_example, split_rows


def test_assignment_is_stable_and_seeded():
    config = SplitConfig(seed=42)
    first = split_for_example("example-1", config)
    assert first == split_for_example("example-1", config)
    assert first in {"train", "validation", "test"}


def test_row_order_does_not_change_assignment(mini_slice_rows):
    config = SplitConfig(seed=42)
    forward = split_rows(mini_slice_rows, config)
    reverse = split_rows(list(reversed(mini_slice_rows)), config)
    forward_ids = {name: {row["example_id"] for row in rows} for name, rows in forward.items()}
    reverse_ids = {name: {row["example_id"] for row in rows} for name, rows in reverse.items()}
    assert forward_ids == reverse_ids


def test_invalid_fractions_fail():
    with pytest.raises(ValueError):
        SplitConfig(train_fraction=0.7, validation_fraction=0.2, test_fraction=0.2)
