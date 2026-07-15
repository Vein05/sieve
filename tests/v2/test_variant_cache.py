"""Integrity tests for the compile-once response-surface cache."""

from __future__ import annotations

from copy import deepcopy

from v2.variant_cache import (
    VariantBuildSpec,
    build_and_save,
    build_variants,
    load_variant_cache,
    slice_fingerprint,
)


def test_fingerprint_changes_when_query_changes(mini_slice_rows):
    changed = deepcopy(mini_slice_rows)
    changed[0]["query"] += " changed"
    assert slice_fingerprint(changed) != slice_fingerprint(mini_slice_rows)


def test_fingerprint_changes_when_candidate_content_changes(mini_slice_rows):
    changed = deepcopy(mini_slice_rows)
    changed[0]["candidate_memories"][0]["content"] += " changed"
    assert slice_fingerprint(changed) != slice_fingerprint(mini_slice_rows)


def test_fingerprint_changes_when_candidate_order_changes(mini_slice_rows):
    changed = deepcopy(mini_slice_rows)
    changed[0]["candidate_memories"][:2] = reversed(
        changed[0]["candidate_memories"][:2]
    )
    assert slice_fingerprint(changed) != slice_fingerprint(mini_slice_rows)


def test_style_specific_budget_grid(
    mini_slice_rows, sieve_entries_by_id, naive_entries_by_id
):
    config = {
        "style_budgets": {"raw": [0, 200], "sieve_v1": [0]},
        "packager_params": {},
    }
    variants = build_variants(
        mini_slice_rows[:1],
        VariantBuildSpec(
            styles=["raw", "sieve_v1"],
            budgets=[200, 400, 800],
            config=config,
            sieve_entries=sieve_entries_by_id,
            naive_entries=naive_entries_by_id,
        ),
    )
    assert {variant.key.budget for variant in variants.values()} == {0, 200}
    assert len(variants) == 3
    assert all(
        variant.budget == 0 or variant.token_count <= variant.budget
        for variant in variants.values()
    )


def test_prompt_override_round_trips(
    tmp_path, mini_slice_rows, sieve_entries_by_id
):
    config = {
        "style_budgets": {"sieve_v1": [0]},
        "packager_params": {},
    }
    path = tmp_path / "variants.json"
    build_and_save(
        path,
        mini_slice_rows[:1],
        VariantBuildSpec(
            styles=["sieve_v1"],
            budgets=[200],
            config=config,
            sieve_entries=sieve_entries_by_id,
        ),
    )
    loaded = load_variant_cache(path)
    variant = next(iter(loaded.entries.values()))
    assert variant.prompt_override == sieve_entries_by_id[variant.example_id]["prompt"]


def test_answer_override_round_trips(
    tmp_path, mini_slice_rows, sieve_entries_by_id
):
    row = mini_slice_rows[0]
    entries = {key: dict(value) for key, value in sieve_entries_by_id.items()}
    entries[row["example_id"]].update(
        prompt="", is_deterministic=True, deterministic_answer="cached"
    )
    config = {"style_budgets": {"sieve_v1": [0]}, "packager_params": {}}
    path = tmp_path / "variants.json"
    spec = VariantBuildSpec(["sieve_v1"], [0], config, entries)
    build_and_save(path, [row], spec)
    variant = next(iter(load_variant_cache(path).entries.values()))
    assert variant.answer_override == "cached"


def test_effective_grid_changes_cache_identity(tmp_path, mini_slice_rows):
    path = tmp_path / "variants.json"
    config = {"packager_params": {}}
    first = VariantBuildSpec(["raw"], [200], config)
    _, first_fingerprint, first_rebuilt = build_and_save(
        path, mini_slice_rows[:1], first
    )
    second = VariantBuildSpec(["raw"], [400], config)
    _, second_fingerprint, second_rebuilt = build_and_save(
        path, mini_slice_rows[:1], second
    )
    assert first_rebuilt is True
    assert second_rebuilt is True
    assert first_fingerprint != second_fingerprint
