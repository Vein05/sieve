"""Deterministic query splits for two-dimensional router evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"


@dataclass(frozen=True)
class SplitConfig:
    """Fractions for stable example-id hashing."""

    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(value <= 0.0 for value in fractions):
            raise ValueError("all split fractions must be positive")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1.0")


def split_for_example(example_id: str, config: SplitConfig) -> str:
    """Assign an example deterministically without depending on row order."""
    digest = hashlib.sha256(f"{config.seed}:{example_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < config.train_fraction:
        return TRAIN
    if value < config.train_fraction + config.validation_fraction:
        return VALIDATION
    return TEST


def split_rows(
    rows: Sequence[Mapping[str, Any]], config: SplitConfig
) -> dict[str, list[Mapping[str, Any]]]:
    """Partition rows while preserving their input order within each split."""
    output: dict[str, list[Mapping[str, Any]]] = {
        TRAIN: [],
        VALIDATION: [],
        TEST: [],
    }
    for row in rows:
        example_id = str(row.get("example_id", ""))
        if not example_id:
            raise ValueError("every row must have a non-empty example_id")
        output[split_for_example(example_id, config)].append(row)
    return output
