"""Gold-containment label extraction for L1 sufficiency estimator.

Produces (query, pack, label) triples from HotpotQA, BEAM, and LongMemEval
slices. Label = fraction of gold units present in the pack (binary: 1.0 if
fully contained, else 0.0). All processing is local / offline.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Pack variants built per row (top-k prefixes and random subsets).
_TOP_K_VALUES = (1, 3, 5, 10, 20)
_RANDOM_SUBSET_SIZES = (3, 7)
_RANDOM_SEED = 42

# BEAM: label a memory as gold if it has this label tag.
_BEAM_GOLD_LABEL = "answer_bearing"

# LME: memory_id prefix marking the answer session.
_LME_GOLD_PREFIX = "answer_"


@dataclass(frozen=True)
class LabeledExample:
    """Single (query, pack, label) training example."""

    example_id: str
    source: str  # "hotpotqa" | "beam" | "longmemeval"
    query: str
    pack_memory_ids: tuple[str, ...]
    pack_texts: tuple[str, ...]
    label: float  # 1.0 = fully sufficient, 0.0 = insufficient
    gold_memory_ids: tuple[str, ...]


def _build_pack_variants(
    candidates: list[dict[str, Any]],
    gold_ids: set[str],
    example_id: str,
    source: str,
    query: str,
) -> list[LabeledExample]:
    """Return LabeledExample list for natural pack variants of one row."""
    rng = random.Random(_RANDOM_SEED)
    variants: list[LabeledExample] = []
    all_ids = [c["memory_id"] for c in candidates]
    all_texts = [str(c["content"]) for c in candidates]
    id_to_text = dict(zip(all_ids, all_texts))

    def make_example(pack_ids: list[str]) -> LabeledExample:
        pack_texts = tuple(id_to_text[mid] for mid in pack_ids if mid in id_to_text)
        covered = sum(1 for gid in gold_ids if gid in set(pack_ids))
        label = 1.0 if (gold_ids and covered == len(gold_ids)) else 0.0
        return LabeledExample(
            example_id=example_id,
            source=source,
            query=query,
            pack_memory_ids=tuple(pack_ids),
            pack_texts=pack_texts,
            label=label,
            gold_memory_ids=tuple(sorted(gold_ids)),
        )

    # Top-k prefix packs.
    for k in _TOP_K_VALUES:
        if k <= len(candidates):
            variants.append(make_example(all_ids[:k]))

    # Full pool if not already covered by top-k.
    n = len(candidates)
    if n not in _TOP_K_VALUES:
        variants.append(make_example(all_ids))

    # Random subsets.
    for size in _RANDOM_SUBSET_SIZES:
        if size < n:
            subset = rng.sample(all_ids, size)
            variants.append(make_example(subset))

    return variants


def load_hotpotqa(slice_path: Path) -> list[LabeledExample]:
    """Load HotpotQA examples. Gold units = gold_decision.selected_memory_ids."""
    examples: list[LabeledExample] = []
    with open(slice_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            gold_ids = set(row.get("gold_decision", {}).get("selected_memory_ids", []))
            if not gold_ids:
                continue
            candidates = row.get("candidate_memories", [])
            if not candidates:
                continue
            examples.extend(
                _build_pack_variants(
                    candidates,
                    gold_ids,
                    example_id=row["example_id"],
                    source="hotpotqa",
                    query=row["query"],
                )
            )
    return examples


def _beam_gold_ids(row: dict[str, Any]) -> set[str]:
    """Return gold memory IDs for a BEAM row using answer_bearing_memory_ids."""
    es = row.get("evidence_sufficiency", {})
    ab_ids = es.get("answer_bearing_memory_ids", [])
    if ab_ids:
        return set(ab_ids)
    # Fallback: memory_usefulness_labels with answer_bearing tag.
    labels_map: dict[str, Any] = row.get("memory_usefulness_labels", {})
    return {
        mid
        for mid, info in labels_map.items()
        if _BEAM_GOLD_LABEL in info.get("labels", [])
    }


def load_beam(slice_path: Path) -> list[LabeledExample]:
    """Load BEAM examples. Gold units = answer_bearing_memory_ids."""
    examples: list[LabeledExample] = []
    with open(slice_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            gold_ids = _beam_gold_ids(row)
            if not gold_ids:
                continue
            candidates = row.get("candidate_memories", [])
            if not candidates:
                continue
            examples.extend(
                _build_pack_variants(
                    candidates,
                    gold_ids,
                    example_id=row["example_id"],
                    source="beam",
                    query=row["query"],
                )
            )
    return examples


def _lme_gold_ids(row: dict[str, Any]) -> set[str]:
    """Return gold memory IDs for a LME row.

    The 'answer_' prefix marks the session containing the reference answer.
    Only IDs present in candidate_memories are counted (BM25 may miss gold).
    """
    cand_ids = {c["memory_id"] for c in row.get("candidate_memories", [])}
    return {mid for mid in cand_ids if mid.startswith(_LME_GOLD_PREFIX)}


def load_longmemeval(slice_path: Path) -> list[LabeledExample]:
    """Load LongMemEval examples. Gold units = answer_-prefixed memory IDs."""
    examples: list[LabeledExample] = []
    with open(slice_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            gold_ids = _lme_gold_ids(row)
            candidates = row.get("candidate_memories", [])
            if not candidates:
                continue
            # Include rows even if gold is absent in BM25 (label = 0.0 for all packs).
            examples.extend(
                _build_pack_variants(
                    candidates,
                    gold_ids,
                    example_id=row["example_id"],
                    source="longmemeval",
                    query=row["query"],
                )
            )
    return examples
