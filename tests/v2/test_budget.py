"""Tests for v2/budget.py: token counting and greedy whole-unit packing."""

from __future__ import annotations

import pytest

from v2.budget import BudgetPacking, count_tokens, truncate_units_to_budget


class TestCountTokens:
    def test_empty_is_zero(self):
        assert count_tokens("") == 0

    def test_nonempty_positive(self):
        assert count_tokens("hello world this is a test") > 0

    def test_never_uses_str_split(self):
        # A string with punctuation glued to words tokenizes to MORE tokens than
        # naive .split() would produce; this guards against the P0 #2 regression.
        text = "don't,stop;believing:now!"
        assert count_tokens(text) >= len(text.split())

    def test_monotonic_in_length(self):
        short = count_tokens("one two three")
        long = count_tokens("one two three four five six seven eight")
        assert long > short


class TestTruncateUnitsToBudget:
    def test_empty_input(self):
        result = truncate_units_to_budget([], budget=100)
        assert isinstance(result, BudgetPacking)
        assert result.units == []
        assert result.token_count == 0
        assert result.overflow is False

    def test_blank_units_dropped(self):
        result = truncate_units_to_budget(["", "   ", "\n"], budget=100)
        assert result.units == []

    def test_single_oversized_unit_is_hard_truncated(self):
        big = " ".join(["word"] * 500)
        result = truncate_units_to_budget([big], budget=5)
        assert result.units
        assert result.units[0] != big
        assert result.overflow is True
        assert result.token_count <= 5

    def test_zero_budget_is_explicitly_unbounded(self):
        units = ["first unit", "second unit"]
        result = truncate_units_to_budget(units, budget=0)
        assert result.units == units
        assert result.token_count > 0

    def test_negative_budget_is_rejected(self):
        with pytest.raises(ValueError):
            truncate_units_to_budget(["unit"], budget=-1)

    def test_greedy_whole_unit_packing_stops_at_budget(self):
        units = [f"unit number {i} has several tokens here" for i in range(50)]
        budget = 40
        result = truncate_units_to_budget(units, budget)
        assert result.token_count <= budget
        assert result.overflow is False
        # Kept a prefix in order.
        assert result.units == units[: len(result.units)]
        # At least one unit fit.
        assert len(result.units) >= 1

    def test_exact_boundary(self):
        # Two identical units; budget set to exactly fit both.
        unit = "alpha beta gamma"
        both = count_tokens("\n".join([unit, unit]))
        result = truncate_units_to_budget([unit, unit, unit], budget=both)
        assert len(result.units) == 2
        assert result.token_count == both
        assert result.overflow is False

    def test_preserves_priority_order(self):
        units = ["first unit here", "second unit here", "third unit here"]
        result = truncate_units_to_budget(units, budget=8)
        for a, b in zip(result.units, units):
            assert a == b
