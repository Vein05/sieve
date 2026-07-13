"""Project evidence units into slot-aware candidates for compilation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

from shared.nlp import cached_build_profile, get_stems_for_text
from .schema import EvidencePlan, slot_requirement_name
from .roles import infer_unit_roles
from .units import EvidenceUnit
from retrieval.query_targets import extract_query_targets

_RELATIONAL_SLOT_TYPES = {"ordering_event", "comparison_operand", "temporal_event", "temporal_time"}
_GENERIC_RELATIONAL_HINT_TOKENS = {"event", "item", "option", "post", "task", "thing"}


@dataclass(frozen=True)
class ProjectedEvidence:
    unit_id: str
    memory_id: str
    slot_names: tuple[str, ...]
    token_count: int
    slot_utility: float


def _unit_matches_slot_hint(unit: EvidenceUnit, slot: Any) -> bool:
    hint = str(getattr(slot, "entity_key", "") or "").strip()
    if not hint:
        return False
    source_text = f"{unit.source_text} {unit.render_text}".strip()
    if not source_text:
        return False
    normalized_hint = cached_build_profile(hint).normalized_text
    normalized_source = cached_build_profile(source_text).normalized_text
    if normalized_hint and re.search(rf"(?<!\w){re.escape(normalized_hint)}(?!\w)", normalized_source):
        return True

    hint_profile = cached_build_profile(hint)
    ordered_hint_tokens = [
        token
        for token in normalized_hint.split()
        if token and token not in _GENERIC_RELATIONAL_HINT_TOKENS
    ]
    if not ordered_hint_tokens:
        ordered_hint_tokens = [token for token in normalized_hint.split() if token]
    hint_stems = [stem for stem in get_stems_for_text(" ".join(ordered_hint_tokens)) if stem]
    if not hint_stems:
        return False

    source_stems = set(get_stems_for_text(source_text))
    overlap = [stem for stem in hint_stems if stem in source_stems]
    if not overlap:
        return False
    if len(hint_stems) == 1:
        return True
    head_stem = hint_stems[-1]
    return head_stem in overlap or len(overlap) >= 2


def _unit_matches_slot_attribute(unit: EvidenceUnit, slot: Any) -> bool:
    slot_attribute = str(getattr(slot, "attribute_key", "") or "").strip().lower()
    if not slot_attribute:
        return True
    provenance = getattr(unit, "provenance", {}) or {}
    if not isinstance(provenance, Mapping):
        return False
    schema_hints = provenance.get("schema_hints")
    if not isinstance(schema_hints, Mapping):
        return False
    unit_attribute = str(schema_hints.get("attribute_key") or "").strip().lower()
    if not unit_attribute:
        return False
    if unit_attribute == slot_attribute:
        return True
    if slot_attribute == "name" and unit_attribute in {"reading", "watching", "studying", "using", "employer"}:
        return True
    if slot_attribute == "brand" and unit_attribute == "using":
        return True
    return False


def project_evidence_candidates(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    units: list[EvidenceUnit],
    memory_labels_by_id: dict[str, dict[str, Any]],
) -> list[ProjectedEvidence]:
    query_targets = extract_query_targets(row, plan.family)
    required_slots = tuple(slot for slot in plan.slots if slot.required)
    slots_by_requirement: dict[str, list[Any]] = {}
    for slot in required_slots:
        slots_by_requirement.setdefault(slot_requirement_name(slot), []).append(slot)

    projected: list[ProjectedEvidence] = []
    for unit in units:
        roles = infer_unit_roles(
            row=row,
            query_family=plan.family,
            query_targets=query_targets,
            unit=unit,
            memory_labels_by_id=memory_labels_by_id,
        )
        slot_names: list[str] = []
        unit_tokens = set(unit.entity_tokens) | set(unit.subject_tokens) | set(unit.value_tokens)
        for role_name, slots in slots_by_requirement.items():
            if role_name not in roles:
                continue
            if len(slots) == 1:
                if not _unit_matches_slot_attribute(unit, slots[0]):
                    continue
                slot_names.append(slots[0].slot_name)
                continue
            relational_group = any(str(getattr(slot, "slot_type", "") or "") in _RELATIONAL_SLOT_TYPES for slot in slots)
            hinted = []
            for slot in slots:
                if not _unit_matches_slot_attribute(unit, slot):
                    continue
                hint = str(slot.entity_key or "").strip().lower()
                if not hint:
                    continue
                if relational_group:
                    if _unit_matches_slot_hint(unit, slot):
                        hinted.append(slot.slot_name)
                elif any(token == hint for token in unit_tokens):
                    hinted.append(slot.slot_name)
            if hinted:
                slot_names.extend(hinted)
            elif not relational_group:
                slot_names.extend(slot.slot_name for slot in slots)
        slot_names = tuple(dict.fromkeys(sorted(slot_names)))
        if not slot_names:
            continue

        slot_utility = float(len(slot_names)) / float(max(1, unit.token_count))
        projected.append(
            ProjectedEvidence(
                unit_id=unit.unit_id,
                memory_id=unit.memory_id,
                slot_names=slot_names,
                token_count=unit.token_count,
                slot_utility=slot_utility,
            )
        )

    projected.sort(
        key=lambda item: (len(item.slot_names), item.slot_utility, -item.token_count, item.unit_id),
        reverse=True,
    )
    return projected


def unresolved_slots(*, requirements: dict[str, int], coverage: dict[str, int]) -> dict[str, int]:
    return {
        slot_name: max(0, required - coverage.get(slot_name, 0))
        for slot_name, required in requirements.items()
        if coverage.get(slot_name, 0) < required
    }
