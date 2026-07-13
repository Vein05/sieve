"""Deterministic evidence compiler for CRISP-Evidence Compiler."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import re
from typing import Any

from ...common import (
    CandidateView,
    cached_build_profile,
    check_stem_equivalence,
    get_clean_tokens,
    get_stems_for_text,
    stem_token,
)
from ...evidence_planner import plan_evidence
from ...evidence_requirements import build_query_evidence_plan
from ...evidence_renderer import display_slot_text
from ...evidence_schema import EvidencePlan, slot_requirement_name
from ..execution.requirements import (
    aggregate_requires_two_operands as _aggregate_requires_two_operands,
    aggregate_slot as _aggregate_slot,
    coverage_dict as _coverage_dict,
    is_comparison_schema as _is_comparison_schema,
    is_current_state_schema as _is_current_state_schema,
    is_direct_lookup_schema as _is_direct_lookup_schema,
    is_state_update_schema as _is_state_update_schema,
    max_compiled_tokens as _max_compiled_tokens,
    missing_requirements as _missing_requirements,
    plan_satisfied as _plan_satisfied,
    required_binding_names as _required_binding_names,
    requirements_dict as _requirements_dict,
    selection_requirements_dict as _selection_requirements_dict,
    validated_missing_requirements as _validated_missing_requirements,
    validated_rescue_missing as _validated_rescue_missing,
)
from ..execution.router import apply_semantic_reader_downgrade
from ..execution.deterministic import build_answer_contract
from ..validation.answer_type import validate_slot_binding as _validate_slot_binding_impl
from ..validation.invariants import schema_grounding_analysis as _schema_grounding_analysis_impl
from ...evidence_roles import infer_unit_roles
from ...evidence_units import EvidenceUnit, build_evidence_units_for_views
from ...evidence_projection import project_evidence_candidates, unresolved_slots
from ...modeling_logic.features import _answer_shape_signals
from ...query_targets import QueryTargets, extract_query_targets
from .temporal import _DATE_LIKE_RE, _TIME_LIKE_RE, _binding_anchor_date, _extract_event_date
from .parsing import _NUMERIC_PARSE_RE, _DURATION_TEXT_RE
from .text import (
    _binding_source_text,
    _binding_token_sets,
    _build_text_analysis,
    _build_compiled_unit_analysis,
    _compiled_unit_body_text,
    _compiled_unit_numeric_mentions,
    _normalize_text,
    _query_normalized_text,
    _query_token_sets,
    _query_expects_time_literal,
    _requested_duration_unit,
    _split_session_header,
    _time_unit_from_query,
)
from .numeric import (
    _contains_math_expression,
    _count_numeric_spans_excluding_dates,
    _extract_numeric_mentions,
    _format_numeric_answer,
    _parse_numeric_text,
)
from .focus import (
    _COUNT_LOOKUP_OBJECT_SKIP_TOKENS,
    _COUNT_LOOKUP_PREDICATE_SKIP_TOKENS,
    _PREDICATE_FOCUS_SKIP_TOKENS,
    _QUERY_FOCUS_SKIP_TOKENS,
    _QUERY_FOCUS_SYNONYMS,
    _WEAK_QUERY_FOCUS_TOKENS,
    _count_lookup_object_tokens,
    _count_lookup_predicate_tokens,
    _predicate_focus_tokens,
    _query_focus_tokens,
    _stem_token,
    _token_stems,
)
from .roles import _PAIRED_ROLE_NAMES, _missing_role_gain, _role_priority
from ..validation.compatibility import _SOFT_INVALID_REASONS

_QUESTION_UNIT_RE = re.compile(
    r"\b(days?|weeks?|months?|years?|hours?|minutes?|mph|mbps|gb|gb|inch|inches|items?|projects?|shirts?|films?|bikes?)\b",
    re.IGNORECASE,
)
_REQUEST_LIKE_RE = re.compile(
    r"\b(?:can you|could you|do you have any tips|i(?:'m| am)? looking for|i need help|i want suggestions|recommend)\b",
    re.IGNORECASE,
)


def _binding_attribute_compatible_from_provenance(
    *,
    slot_attribute: str,
    provenance: Mapping[str, Any] | None,
    source_text: str,
) -> bool:
    slot_attribute = str(slot_attribute or "").strip().lower()
    if not slot_attribute:
        return True
    if not isinstance(provenance, Mapping):
        return False
    schema_hints = provenance.get("schema_hints")
    unit_attribute = str(schema_hints.get("attribute_key") or "").strip().lower() if isinstance(schema_hints, Mapping) else ""
    if unit_attribute:
        return _slot_attribute_compatible(slot_attribute, unit_attribute)
    if slot_attribute == "duration":
        duration_value = str(provenance.get("duration_value") or "").strip()
        return bool(duration_value or _DURATION_TEXT_RE.search(source_text) or "how long" in _normalize_text(source_text))
    if slot_attribute == "time":
        time_value = str(provenance.get("time_value") or "").strip()
        return bool(time_value or _DATE_LIKE_RE.search(source_text) or _TIME_LIKE_RE.search(source_text))
    return False


def _unit_tags(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets=None,
    unit: EvidenceUnit,
    memory_labels_by_id: dict[str, dict[str, Any]],
) -> set[str]:
    resolved_query_targets = query_targets or extract_query_targets(row, plan.query_family)
    return infer_unit_roles(
        row=row,
        query_family=plan.query_family,
        query_targets=resolved_query_targets,
        unit=unit,
        memory_labels_by_id=memory_labels_by_id,
    )


def score_evidence_unit(
    *,
    row: dict[str, Any],
    query_family: str,
    unit: EvidenceUnit,
    memory_labels_by_id: dict[str, dict[str, Any]],
    score_weights: dict[str, float] | None = None,
    plan: EvidencePlan | None = None,
) -> float:
    resolved_weights = score_weights if isinstance(score_weights, dict) else {}
    return _semantic_score_evidence_unit(
        row=row,
        query_family=query_family,
        unit=unit,
        memory_labels_by_id=memory_labels_by_id,
        score_weights=resolved_weights,
        plan=plan,
    )


def _unit_requirement_tags(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets=None,
    unit: EvidenceUnit,
    memory_labels_by_id: dict[str, dict[str, Any]],
) -> set[str]:
    return _unit_tags(
        row=row,
        plan=plan,
        query_targets=query_targets,
        unit=unit,
        memory_labels_by_id=memory_labels_by_id,
    )


def _unit_requirement_tags_cached(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets=None,
    unit: EvidenceUnit,
    memory_labels_by_id: dict[str, dict[str, Any]],
    role_cache: dict[str, set[str]],
) -> set[str]:
    cached = role_cache.get(unit.unit_id)
    if cached is None:
        cached = _unit_requirement_tags(
            row=row,
            plan=plan,
            query_targets=query_targets,
            unit=unit,
            memory_labels_by_id=memory_labels_by_id,
        )
        role_cache[unit.unit_id] = cached
    return cached


def _selected_units_with_roles(
    *,
    selected_units: list[EvidenceUnit],
    role_names: set[str],
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets | None = None,
    memory_labels_by_id: dict[str, dict[str, Any]],
    role_cache: dict[str, set[str]] | None = None,
) -> list[EvidenceUnit]:
    cached_roles: dict[str, set[str]] = {} if role_cache is None else role_cache
    role_units: list[EvidenceUnit] = []
    for unit in selected_units:
        unit_tags = _unit_requirement_tags_cached(
            row=row,
            plan=plan,
            query_targets=query_targets,
            unit=unit,
            memory_labels_by_id=memory_labels_by_id,
            role_cache=cached_roles,
        )
        if unit_tags & role_names:
            role_units.append(unit)
    return role_units


def _selected_unit_roles(
    *,
    selected_units: list[EvidenceUnit],
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets | None = None,
    memory_labels_by_id: dict[str, dict[str, Any]],
) -> tuple[set[str], dict[str, list[EvidenceUnit]], dict[str, set[str]]]:
    role_cache: dict[str, set[str]] = {}
    selected_tags: set[str] = set()
    role_units: dict[str, list[EvidenceUnit]] = {}
    memory_roles: dict[str, set[str]] = {}
    for unit in selected_units:
        unit_tags = _unit_requirement_tags_cached(
            row=row,
            plan=plan,
            query_targets=query_targets,
            unit=unit,
            memory_labels_by_id=memory_labels_by_id,
            role_cache=role_cache,
        )
        selected_tags |= unit_tags
        memory_roles.setdefault(unit.memory_id, set()).update(unit_tags)
        for tag in unit_tags:
            role_units.setdefault(tag, []).append(unit)
    return selected_tags, role_units, memory_roles


def _unit_matches_query_entity(entity: str, unit: EvidenceUnit) -> bool:
    normalized_entity = _normalize_text(entity)
    if not normalized_entity:
        return False
    normalized_source = _normalize_text(str(unit.provenance.get("source_text") or unit.render_text or ""))
    if normalized_entity in normalized_source:
        return True
    candidates = {
        _normalize_text(token)
        for token in (
            *unit.entity_tokens,
            *unit.value_tokens,
            str(unit.render_text),
            str(unit.provenance.get("source_text") or ""),
        )
        if token
    }
    return normalized_entity in candidates


def _distinct_value_signatures(units: list[EvidenceUnit]) -> set[tuple[str, ...]]:
    signatures: set[tuple[str, ...]] = set()
    for unit in units:
        parts = tuple(
            token
            for token in (
                *unit.numeric_values,
                *unit.time_markers,
                *unit.entity_tokens[:3],
                *unit.value_tokens[:3],
            )
            if token
        )
        if parts:
            signatures.add(parts)
    return signatures


def _same_memory_pair_sufficient(
    *,
    plan: EvidencePlan,
    query_targets: QueryTargets,
    role_a: str,
    role_b: str,
    pair_units: list[EvidenceUnit],
) -> bool:
    if not pair_units:
        return False

    unique_units = {unit.unit_id: unit for unit in pair_units}
    units = list(unique_units.values())
    primary_entities = tuple(entity for entity in query_targets.subject_entities if entity) or tuple(
        entity for entity in query_targets.candidate_entities if entity
    )
    matched_entities = {
        entity
        for entity in primary_entities
        if any(_unit_matches_query_entity(entity, unit) for unit in units)
    }
    distinct_numeric_values = {
        value
        for unit in units
        for value in unit.numeric_values
        if value
    }
    distinct_time_markers = {
        marker
        for unit in units
        for marker in unit.time_markers
        if marker
    }
    distinct_value_signatures = _distinct_value_signatures(units)
    has_pair_language = any(
        marker in f" {cached_build_profile(unit.render_text).normalized_text} "
        for unit in units
        for marker in (" between ", " from ", " to ", " after ", " before ", " until ", " since ")
    )

    if {role_a, role_b} == {"comparison_left", "comparison_right"}:
        if len(matched_entities) >= 2:
            return True
        if len(distinct_numeric_values) >= 2 and any(unit.has_comparison_language for unit in units):
            return True
        return False

    if {role_a, role_b} in (
        {"temporal_event_a", "temporal_event_b"},
        {"temporal_time_a", "temporal_time_b"},
    ):
        if len(matched_entities) >= 2:
            return True
        if len(units) >= 2 and len(distinct_time_markers) >= 2:
            return True
        if len(units) >= 2 and len(distinct_value_signatures) >= 2 and any(unit.time_markers for unit in units):
            return True
        if has_pair_language and len(distinct_time_markers) >= 2:
            return True
        return False

    if {role_a, role_b} == {"old_state", "new_state"}:
        if any(unit.update_markers or unit.has_current_state_language for unit in units):
            return True
        return len(units) >= 2 and len(distinct_value_signatures) >= 2

    return len(units) >= 2


def _effective_missing_roles(
    *,
    tags: set[str],
    missing: dict[str, int],
    unit: EvidenceUnit,
    selected_units: list[EvidenceUnit],
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets,
    memory_labels_by_id: dict[str, dict[str, Any]],
    role_cache: dict[str, set[str]],
) -> set[str]:
    effective_roles: set[str] = set()
    if not tags:
        return effective_roles

    selected_ordering_slots: set[str] = set()
    unit_ordering_slots: set[str] = set()
    if plan.schema_name == "OrderedChoice":
        ordering_slots = [
            slot
            for slot in plan.slots
            if slot.required and str(getattr(slot, "slot_type", "") or "") == "ordering_event"
        ]
        for selected_unit in selected_units:
            source_text = str(selected_unit.source_text or selected_unit.render_text or "").strip()
            for slot in ordering_slots:
                if _slot_entity_match(slot, source_text, query_targets):
                    selected_ordering_slots.add(slot.slot_name)
        source_text = str(unit.source_text or unit.render_text or "").strip()
        for slot in ordering_slots:
            if _slot_entity_match(slot, source_text, query_targets):
                unit_ordering_slots.add(slot.slot_name)

    selected_role_units: dict[str, list[EvidenceUnit]] = {}
    for role_name in tags & set(missing):
        counterpart = _PAIRED_ROLE_NAMES.get(role_name)
        if counterpart:
            selected_role_units[counterpart] = _selected_units_with_roles(
                selected_units=selected_units,
                role_names={counterpart},
                row=row,
                plan=plan,
                query_targets=query_targets,
                memory_labels_by_id=memory_labels_by_id,
                role_cache=role_cache,
            )

    for role_name, deficit in missing.items():
        if deficit <= 0 or role_name not in tags:
            continue
        if plan.schema_name == "OrderedChoice" and role_name == "ordering_event":
            if not unit_ordering_slots or unit_ordering_slots <= selected_ordering_slots:
                continue
            effective_roles.add(role_name)
            continue

        counterpart = _PAIRED_ROLE_NAMES.get(role_name)
        if not counterpart or missing.get(counterpart, 0) <= 0:
            effective_roles.add(role_name)
            continue

        same_unit_pair = tags & {role_name, counterpart}
        if same_unit_pair == {role_name, counterpart} and _same_memory_pair_sufficient(
            plan=plan,
            query_targets=query_targets,
            role_a=role_name,
            role_b=counterpart,
            pair_units=[unit],
        ):
            effective_roles.add(role_name)
            continue

        counterpart_units = [
            selected_unit
            for selected_unit in selected_role_units.get(counterpart, [])
            if selected_unit.memory_id == unit.memory_id
        ]
        if counterpart_units and not _same_memory_pair_sufficient(
            plan=plan,
            query_targets=query_targets,
            role_a=role_name,
            role_b=counterpart,
            pair_units=[*counterpart_units, unit],
        ):
            continue

        effective_roles.add(role_name)

    return effective_roles


def _unit_payload(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets=None,
    unit: EvidenceUnit,
    memory_labels_by_id: dict[str, dict[str, Any]],
    score_weights: dict[str, float],
) -> dict[str, Any]:
    raw_tags = _unit_requirement_tags(
        row=row,
        plan=plan,
        query_targets=query_targets,
        unit=unit,
        memory_labels_by_id=memory_labels_by_id,
    )
    tags = sorted(raw_tags)
    slot_candidates = _slot_candidates_for_unit(
        plan=plan,
        unit=unit,
        tags=raw_tags,
        query_targets=query_targets,
    )
    return {
        "unit_id": unit.unit_id,
        "memory_id": unit.memory_id,
        "parent_rank": unit.parent_rank,
        "local_order": unit.local_order,
        "render_text": unit.render_text,
        "speaker": unit.speaker,
        "date_key": unit.date_key,
        "token_count": unit.token_count,
        "unit_type": unit.unit_type,
        "entity_tokens": list(unit.entity_tokens),
        "subject_tokens": list(unit.subject_tokens),
        "value_tokens": list(unit.value_tokens),
        "numeric_values": list(unit.numeric_values),
        "time_markers": list(unit.time_markers),
        "update_markers": list(unit.update_markers),
        "typed_flags": {
            "is_direct_answer_candidate": bool(getattr(unit, "is_direct_answer_candidate", False)),
            "is_numeric_operand": bool(getattr(unit, "is_numeric_operand", False)),
            "is_temporal_anchor": bool(getattr(unit, "is_temporal_anchor", False)),
            "is_state_assertion": bool(getattr(unit, "is_state_assertion", False)),
            "is_update_assertion": bool(getattr(unit, "is_update_assertion", False)),
            "is_current_state_candidate": bool(getattr(unit, "is_current_state_candidate", False)),
            "is_attribute_value": bool(getattr(unit, "is_attribute_value", False)),
            "is_countable_item": bool(getattr(unit, "is_countable_item", False)),
            "is_generic_state_text": bool(getattr(unit, "is_generic_state_text", False)),
            "is_low_authority_text": bool(getattr(unit, "is_low_authority_text", False)),
            "is_question_or_request": bool(getattr(unit, "is_question_or_request", False)),
            "is_recommendation_or_advice": bool(getattr(unit, "is_recommendation_or_advice", False)),
            "is_instructional_quantity": bool(getattr(unit, "is_instructional_quantity", False)),
            "is_answer_like_quantity_statement": bool(getattr(unit, "is_answer_like_quantity_statement", False)),
            "has_progress_marker": bool(getattr(unit, "has_progress_marker", False)),
            "has_acquisition_marker": bool(getattr(unit, "has_acquisition_marker", False)),
            "has_consumption_or_completion_marker": bool(getattr(unit, "has_consumption_or_completion_marker", False)),
        },
        "tags": tags,
        "slot_candidates": slot_candidates,
        "score": round(
            score_evidence_unit(
                row=row,
                query_family=plan.query_family,
                unit=unit,
                memory_labels_by_id=memory_labels_by_id,
                score_weights=score_weights,
                plan=plan,
            ),
            6,
        ),
        "provenance": dict(unit.provenance),
        "analysis": _build_compiled_unit_analysis(str(unit.provenance.get("source_text") or unit.render_text or "")).to_payload(),
    }


def _slot_candidates_for_unit(
    *,
    plan: EvidencePlan,
    unit: EvidenceUnit,
    tags: set[str],
    query_targets: QueryTargets | None = None,
) -> list[str]:
    unit_tokens = set(unit.entity_tokens) | set(unit.subject_tokens) | set(unit.value_tokens)
    schema_hints = unit.provenance.get("schema_hints", {}) if isinstance(unit.provenance, Mapping) else {}
    unit_attribute = str(schema_hints.get("attribute_key") or "").strip().lower() if isinstance(schema_hints, Mapping) else ""
    preferred_attribute = str(getattr(plan, "plan_metadata", {}).get("preferred_attribute") or "").strip().lower()
    generic_state_text = getattr(unit, "is_generic_state_text", False)
    slots_by_requirement: dict[str, list[Any]] = {}
    for slot in plan.slots:
        if not slot.required:
            continue
        slots_by_requirement.setdefault(slot_requirement_name(slot), []).append(slot)

    candidates: list[str] = []
    direct_anchor_slots: list[str] = []
    for role_name, slots in slots_by_requirement.items():
        if role_name not in tags:
            # Track direct_anchor slots that were skipped due to missing role
            # tags — these are candidates for entity-match fallback below.
            if role_name == "direct_anchor":
                direct_anchor_slots.extend(slot.slot_name for slot in slots)
            continue
        if len(slots) == 1:
            only_slot = slots[0]
            slot_attribute = str(getattr(only_slot, "attribute_key", "") or "").strip().lower()
            if slot_attribute and not _binding_attribute_compatible_from_provenance(
                slot_attribute=slot_attribute,
                provenance=unit.provenance if isinstance(unit.provenance, Mapping) else None,
                source_text=str(unit.source_text or unit.render_text or "").strip(),
            ):
                continue
            if (
                preferred_attribute
                and str(getattr(only_slot, "slot_type", "") or "") in {"direct_value", "current_resolution", "state_anchor"}
                and generic_state_text
            ):
                continue
            candidates.append(slots[0].slot_name)
            continue
        relational_group = any(
            str(getattr(slot, "slot_type", "") or "") in {"comparison_operand", "ordering_event", "temporal_event", "temporal_time"}
            for slot in slots
        )
        hinted = []
        source_text = str(unit.source_text or unit.render_text or "").strip()
        for slot in slots:
            hint = str(slot.entity_key or "").strip().lower()
            slot_attribute = str(getattr(slot, "attribute_key", "") or "").strip().lower()
            if slot_attribute and not _binding_attribute_compatible_from_provenance(
                slot_attribute=slot_attribute,
                provenance=unit.provenance if isinstance(unit.provenance, Mapping) else None,
                source_text=str(unit.source_text or unit.render_text or "").strip(),
            ):
                continue
            if not hint:
                if slot_attribute and unit_attribute == slot_attribute:
                    hinted.append(slot.slot_name)
                continue
            if relational_group:
                if isinstance(query_targets, QueryTargets) and _slot_entity_match(slot, source_text, query_targets):
                    hinted.append(slot.slot_name)
            elif hint in unit_tokens:
                hinted.append(slot.slot_name)
        if hinted:
            candidates.extend(hinted)
        elif not relational_group:
            if preferred_attribute and generic_state_text:
                continue
            candidates.extend(slot.slot_name for slot in slots)

    # Entity-match fallback: if a unit was excluded from direct_anchor slots
    # because evidence_roles didn't grant the tag, but the unit has a strong
    # entity match with the query, allow it as a candidate.  This is the
    # compiler's own query-conditioned inference — it doesn't rely on the
    # pre-process tagger's metadata to decide relevance.
    if direct_anchor_slots and not any(s in candidates for s in direct_anchor_slots):
        if (
            isinstance(query_targets, QueryTargets)
            and plan.family in {"single_anchor", "information_extraction", "current_state"}
            and not generic_state_text
        ):
            source_text = str(unit.source_text or unit.render_text or "").strip()
            primary_entities = tuple(e for e in query_targets.subject_entities if e) or tuple(
                e for e in query_targets.candidate_entities if e
            )
            if primary_entities and _best_matching_entity(source_text, primary_entities):
                candidates.extend(direct_anchor_slots)

    return list(dict.fromkeys(candidates))


def _slot_bindings(
    plan: EvidencePlan,
    compiled_units: list[dict[str, Any]],
    query_targets: QueryTargets | None = None,
) -> dict[str, dict[str, Any] | None]:
    def _binding_payload_from_unit(unit: dict[str, Any], *, slot: Any | None = None, text: str | None = None) -> dict[str, Any]:
        provenance = unit.get("provenance") if isinstance(unit.get("provenance"), Mapping) else {}
        schema_hints = provenance.get("schema_hints") if isinstance(provenance, Mapping) else {}
        chosen_text = text if text is not None else unit.get("render_text")
        if slot is not None:
            chosen_text = _preferred_binding_text(plan, slot, unit, fallback_text=str(chosen_text or "").strip())
        payload = {
            "unit_id": unit.get("unit_id"),
            "memory_id": unit.get("memory_id"),
            "text": chosen_text,
            "date": unit.get("date_key"),
            "speaker": unit.get("speaker"),
            "object_id": provenance.get("object_id"),
            "object_type": provenance.get("object_type"),
            "extracted_value_text": provenance.get("extracted_value_text"),
            "place_value": provenance.get("place_value"),
            "time_value": provenance.get("time_value"),
            "duration_value": provenance.get("duration_value"),
        }
        if isinstance(schema_hints, Mapping):
            payload["schema_hints"] = dict(schema_hints)
        return payload

    if _aggregate_requires_two_operands(plan):
        aggregate_candidates: list[dict[str, Any]] = []
        for unit in compiled_units:
            source_text = _compiled_unit_body_text(unit)
            mentions = _compiled_unit_numeric_mentions(unit)
            if mentions:
                for _, _, _, span in mentions:
                    aggregate_candidates.append(
                        _binding_payload_from_unit(unit, slot=_aggregate_slot(plan) or None, text=span)
                    )
                continue
            candidates = unit.get("slot_candidates")
            if not isinstance(candidates, list) or "aggregate_items" not in candidates:
                continue
            aggregate_candidates.append(_binding_payload_from_unit(unit, slot=_aggregate_slot(plan) or None))
        bindings: dict[str, dict[str, Any] | None] = {
            "aggregate_items": aggregate_candidates[0] if aggregate_candidates else None,
            "operand_1": aggregate_candidates[0] if len(aggregate_candidates) >= 1 else None,
            "operand_2": aggregate_candidates[1] if len(aggregate_candidates) >= 2 else None,
        }
        for index, payload in enumerate(aggregate_candidates[2:], start=3):
            bindings[f"operand_{index}"] = payload
        return bindings

    slot_names = [
        slot.slot_name
        for slot in plan.slots
        if slot.required or (plan.requires_numeric_reasoning and str(slot.slot_type) == "numeric_operand")
    ]
    bindings: dict[str, dict[str, Any] | None] = {slot_name: None for slot_name in slot_names}
    used_unit_ids: set[str] = set()
    slot_by_name = {
        slot.slot_name: slot
        for slot in plan.slots
        if slot.required or (plan.requires_numeric_reasoning and str(slot.slot_type) == "numeric_operand")
    }

    def _compiled_unit_attribute_matches(unit_payload: dict[str, Any], slot: Any) -> bool:
        slot_attribute = str(getattr(slot, "attribute_key", "") or "").strip().lower()
        if not slot_attribute:
            return True
        provenance = unit_payload.get("provenance")
        source_text = str((provenance or {}).get("source_text") or unit_payload.get("render_text") or "").strip()
        return _binding_attribute_compatible_from_provenance(
            slot_attribute=slot_attribute,
            provenance=provenance if isinstance(provenance, Mapping) else None,
            source_text=source_text,
        )
    repeated_requirements = {
        requirement
        for requirement, count in _selection_requirements_dict(plan).items()
        if count > 1
    }

    ordering_slots = [slot for slot in plan.slots if slot.required and str(slot.slot_type) == "ordering_event"]
    if ordering_slots and isinstance(query_targets, QueryTargets) and query_targets.candidate_entities:
        seen_labels: set[str] = set()
        for slot in ordering_slots:
            for unit in compiled_units:
                source_text = str(((unit.get("provenance") or {}).get("source_text") or unit.get("render_text") or "")).strip()
                label = _slot_entity_match(slot, source_text, query_targets)
                if not label:
                    continue
                unit_id = str(unit.get("unit_id"))
                normalized_label = _normalize_text(label)
                if unit_id in used_unit_ids or normalized_label in seen_labels:
                    continue
                bindings[slot.slot_name] = {
                    **_binding_payload_from_unit(unit),
                }
                used_unit_ids.add(unit_id)
                seen_labels.add(normalized_label)
                break

    for slot_name in slot_names:
        if bindings[slot_name] is not None:
            continue
        slot = slot_by_name[slot_name]
        chosen = _best_slot_candidate(
            plan=plan,
            slot=slot,
            compiled_units=compiled_units,
            slot_name=slot_name,
            used_unit_ids=used_unit_ids,
            query_targets=query_targets,
            allow_reuse_for_requirement=False,
            compiled_unit_attribute_matches=_compiled_unit_attribute_matches,
        )
        if chosen is not None:
            bindings[slot_name] = chosen
            used_unit_ids.add(str(chosen.get("unit_id")))

    for slot_name in slot_names:
        if bindings[slot_name] is not None:
            continue
        slot = slot_by_name[slot_name]
        requirement_name = slot_requirement_name(slot)
        chosen = _best_slot_candidate(
            plan=plan,
            slot=slot,
            compiled_units=compiled_units,
            slot_name=slot_name,
            used_unit_ids=used_unit_ids,
            query_targets=query_targets,
            allow_reuse_for_requirement=requirement_name not in repeated_requirements,
            compiled_unit_attribute_matches=_compiled_unit_attribute_matches,
        )
        if chosen is not None:
            bindings[slot_name] = chosen
            used_unit_ids.add(str(chosen.get("unit_id")))

    return bindings


def _time_unit_from_query(row: dict[str, Any]) -> str:
    normalized = _normalize_text(str(row.get("query", "")))
    for unit in ("day", "week", "month", "year", "hour", "minute"):
        if re.search(rf"\b{unit}s?\b", normalized):
            return unit
    return "day"


def _copy_binding_with_text(binding: dict[str, Any], text: str) -> dict[str, Any]:
    updated = dict(binding)
    updated["text"] = str(text).strip()
    updated["display_text"] = str(text).strip()
    return updated


def _preferred_binding_text(
    plan: EvidencePlan,
    slot: Any,
    unit: dict[str, Any],
    *,
    fallback_text: str | None = None,
) -> str | None:
    provenance = unit.get("provenance") if isinstance(unit.get("provenance"), Mapping) else {}
    if not isinstance(provenance, Mapping):
        return fallback_text
    slot_type = str(getattr(slot, "slot_type", "") or "")
    interrogative = str(getattr(plan, "plan_metadata", {}).get("interrogative") or "").strip().lower()
    if slot_type in {"direct_value", "current_resolution", "new_state", "old_state"}:
        if interrogative == "where":
            place_value = str(provenance.get("place_value") or "").strip()
            if place_value:
                return place_value
        if interrogative == "when":
            time_value = str(provenance.get("time_value") or "").strip()
            if time_value:
                return time_value
        if interrogative == "how_long":
            duration_value = str(provenance.get("duration_value") or "").strip()
            if duration_value:
                return duration_value
        extracted_value_text = str(provenance.get("extracted_value_text") or "").strip()
        if extracted_value_text:
            return extracted_value_text
    if slot_type in {"numeric_operand", "count_item", "count_evidence"}:
        if interrogative == "how_long":
            duration_value = str(provenance.get("duration_value") or "").strip()
            if duration_value:
                return duration_value
        extracted_value_text = str(provenance.get("extracted_value_text") or "").strip()
        if extracted_value_text:
            return extracted_value_text
    return fallback_text


def _slot_candidate_rank(
    plan: EvidencePlan,
    slot: Any,
    unit: dict[str, Any],
    query_targets: QueryTargets | None,
) -> tuple[float, float]:
    base_score = float(unit.get("score") or 0.0)
    slot_type = str(getattr(slot, "slot_type", "") or "")
    tags = {str(tag).strip().lower() for tag in unit.get("tags") or [] if tag}
    typed_flags = unit.get("typed_flags") if isinstance(unit.get("typed_flags"), dict) else {}
    speaker = str(unit.get("speaker") or "").strip().lower()
    source_text = str(((unit.get("provenance") or {}).get("source_text") or unit.get("render_text") or "")).strip()
    normalized_source = _normalize_text(source_text)
    render_text = str(unit.get("render_text") or "").strip()
    provenance = unit.get("provenance") if isinstance(unit.get("provenance"), Mapping) else {}
    schema_hints = provenance.get("schema_hints") if isinstance(provenance, Mapping) else {}
    labels = {str(label).strip().lower() for label in ((schema_hints or {}).get("labels") or []) if label}
    preferred_attribute = str(getattr(slot, "attribute_key", "") or "").strip().lower()
    unit_attribute = str((schema_hints or {}).get("attribute_key") or "").strip().lower()
    interrogative = str(getattr(plan, "plan_metadata", {}).get("interrogative") or "").strip().lower()
    primary_entities = ()
    if isinstance(query_targets, QueryTargets):
        primary_entities = tuple(entity for entity in query_targets.subject_entities if entity) or tuple(
            entity for entity in query_targets.candidate_entities if entity
        )

    if slot_type not in {"count_item", "count_evidence"}:
        rank = base_score
        if speaker == "user":
            rank += 1.0
        elif speaker == "assistant":
            rank -= 0.5
        if bool(typed_flags.get("is_question_or_request")):
            rank -= 4.0
        if bool(typed_flags.get("is_recommendation_or_advice")):
            rank -= 3.5
        if bool(typed_flags.get("is_low_authority_text")):
            rank -= 4.0
        if bool(typed_flags.get("is_generic_state_text")):
            rank -= 2.0
        if "direct_answer_candidate" in labels:
            rank += 2.5
        if "attribute_value" in labels:
            rank += 2.0
        if "current_state_candidate" in labels:
            rank += 1.5
        if preferred_attribute:
            if unit_attribute and _slot_attribute_compatible(preferred_attribute, unit_attribute):
                rank += 4.0
            elif unit_attribute:
                rank -= 4.0
        if slot_type in {"direct_value", "current_resolution", "new_state", "old_state"}:
            extracted_value_text = str(provenance.get("extracted_value_text") or "").strip()
            place_value = str(provenance.get("place_value") or "").strip()
            time_value = str(provenance.get("time_value") or "").strip()
            duration_value = str(provenance.get("duration_value") or "").strip()
            if extracted_value_text:
                rank += 2.0
            if interrogative == "where":
                if place_value:
                    rank += 5.0
                if _TIME_LIKE_RE.fullmatch(render_text) or _DATE_LIKE_RE.fullmatch(render_text):
                    rank -= 6.0
            elif interrogative == "when":
                if time_value:
                    rank += 5.0
            elif interrogative == "how_long":
                if duration_value:
                    rank += 5.0
            if primary_entities:
                if any(_best_matching_entity(source_text, (entity,)) for entity in primary_entities):
                    rank += 2.5
                elif interrogative not in {"when", "where"}:
                    rank -= 1.5
        if slot_type in {"temporal_event", "temporal_time", "ordering_event"} and isinstance(query_targets, QueryTargets):
            if _slot_entity_match(slot, source_text, query_targets):
                rank += 3.5
            else:
                rank -= 2.0
        if slot_type == "numeric_operand" and interrogative == "how_long":
            duration_value = str(provenance.get("duration_value") or "").strip()
            if duration_value:
                rank += 5.0
        return (rank, -float(unit.get("parent_rank") or 0))

    rank = base_score
    entity_key = str(getattr(slot, "entity_key", "") or "").strip().lower()
    entity_stems = {
        stem
        for stem in get_stems_for_text(entity_key)
        if stem and stem not in {"item", "items", "thing", "things", "amount", "number", "count"}
    }
    source_stems = {
        stem
        for stem in get_stems_for_text(source_text)
        if stem
    }
    overlap = entity_stems & source_stems
    if slot_type in tags:
        rank += 4.0
    if "count_evidence" in tags:
        rank += 2.5
    if "count_item" in tags:
        rank += 2.0
    if entity_stems:
        if overlap:
            rank += 4.0 + float(len(overlap))
        else:
            rank -= 3.0
    if bool(typed_flags.get("is_countable_item")):
        rank += 2.0
    if bool(typed_flags.get("is_answer_like_quantity_statement")):
        rank += 3.0
    if bool(typed_flags.get("is_instructional_quantity")):
        rank += 2.5
    if bool(typed_flags.get("has_progress_marker")):
        rank += 1.5
    if bool(typed_flags.get("has_acquisition_marker")):
        rank += 1.0
    if bool(typed_flags.get("has_consumption_or_completion_marker")):
        rank += 1.0
    if speaker == "user":
        rank += 1.5
    elif speaker == "assistant":
        rank -= 1.0
    if bool(typed_flags.get("is_question_or_request")):
        rank -= 5.0
    if bool(typed_flags.get("is_recommendation_or_advice")):
        rank -= 4.0
    if bool(typed_flags.get("is_low_authority_text")):
        rank -= 4.5
    if _REQUEST_LIKE_RE.search(source_text):
        rank -= 3.5
    if "as an ai language model" in normalized_source:
        rank -= 4.0
    if "generic_state_text" in {str(label).strip().lower() for label in (((unit.get("provenance") or {}).get("schema_hints") or {}).get("labels") or []) if label}:
        rank -= 1.5
    return (rank, -float(unit.get("parent_rank") or 0))


def _best_slot_candidate(
    *,
    plan: EvidencePlan,
    slot: Any,
    compiled_units: list[dict[str, Any]],
    slot_name: str,
    used_unit_ids: set[str],
    query_targets: QueryTargets | None,
    allow_reuse_for_requirement: bool = False,
    compiled_unit_attribute_matches: Any,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    requirement_name = slot_requirement_name(slot)
    for unit in compiled_units:
        candidate_slots = unit.get("slot_candidates")
        if not isinstance(candidate_slots, list) or slot_name not in candidate_slots:
            continue
        unit_id = str(unit.get("unit_id"))
        if unit_id in used_unit_ids and not allow_reuse_for_requirement:
            continue
        render_text = str(unit.get("render_text") or "").strip()
        slot_type = str(slot.slot_type)
        if slot_type == "numeric_operand":
            provenance = unit.get("provenance") if isinstance(unit.get("provenance"), Mapping) else {}
            extracted_value_text = str((provenance or {}).get("extracted_value_text") or "").strip()
            duration_value = str((provenance or {}).get("duration_value") or "").strip()
            numeric_values = [str(value).strip() for value in unit.get("numeric_values") or [] if str(value).strip()]
            interrogative = str(getattr(plan, "plan_metadata", {}).get("interrogative") or "").strip().lower()
            extracted_is_numeric = bool(_NUMERIC_PARSE_RE.search(extracted_value_text))
            has_numeric_signal = bool(numeric_values or extracted_is_numeric)
            if interrogative == "how_long" and duration_value:
                has_numeric_signal = True
            if not has_numeric_signal:
                continue
        if slot_type == "direct_value" and (_TIME_LIKE_RE.fullmatch(render_text) or _DATE_LIKE_RE.fullmatch(render_text)):
            continue
        if not compiled_unit_attribute_matches(unit, slot):
            continue
        if (
            isinstance(query_targets, QueryTargets)
            and slot.entity_key
            and slot_type in {"comparison_operand", "ordering_event", "temporal_event", "temporal_time"}
        ):
            source_text = str(((unit.get("provenance") or {}).get("source_text") or render_text)).strip()
            if not _slot_entity_match(slot, source_text, query_targets):
                continue
        candidates.append(unit)
    if not candidates:
        return None
    candidates.sort(key=lambda unit: _slot_candidate_rank(plan, slot, unit, query_targets), reverse=True)
    chosen = candidates[0]
    provenance = chosen.get("provenance") if isinstance(chosen.get("provenance"), Mapping) else {}
    schema_hints = provenance.get("schema_hints") if isinstance(provenance, Mapping) else {}
    chosen_text = _preferred_binding_text(plan, slot, chosen, fallback_text=str(chosen.get("render_text") or "").strip())
    payload = {
        "unit_id": chosen.get("unit_id"),
        "memory_id": chosen.get("memory_id"),
        "text": chosen_text,
        "date": chosen.get("date_key"),
        "speaker": chosen.get("speaker"),
        "object_id": provenance.get("object_id"),
        "object_type": provenance.get("object_type"),
        "extracted_value_text": provenance.get("extracted_value_text"),
        "place_value": provenance.get("place_value"),
        "time_value": provenance.get("time_value"),
        "duration_value": provenance.get("duration_value"),
    }
    if isinstance(schema_hints, Mapping):
        payload["schema_hints"] = dict(schema_hints)
    return payload


def _slot_bindings_with_display_text(
    *,
    plan: EvidencePlan,
    compiled_units: list[dict[str, Any]],
    slot_bindings: dict[str, dict[str, Any] | None],
) -> dict[str, dict[str, Any] | None]:
    slot_by_name = {slot.slot_name: slot for slot in plan.slots if slot.required}
    if _aggregate_requires_two_operands(plan):
        aggregate_slot = _aggregate_slot(plan)
        if aggregate_slot is not None:
            slot_by_name.setdefault("aggregate_items", aggregate_slot)
            for slot_name in slot_bindings:
                if slot_name.startswith("operand_"):
                    slot_by_name.setdefault(slot_name, replace(aggregate_slot, slot_name=slot_name, is_variadic=False))
    units_by_id = {
        str(unit.get("unit_id")): unit
        for unit in compiled_units
        if isinstance(unit, dict) and unit.get("unit_id") is not None
    }
    rendered: dict[str, dict[str, Any] | None] = {}
    for slot_name, payload in slot_bindings.items():
        if not isinstance(payload, dict):
            rendered[slot_name] = None
            continue
        slot = slot_by_name.get(slot_name)
        if slot is None:
            rendered[slot_name] = None
            continue
        unit = units_by_id.get(str(payload.get("unit_id")))
        raw_text = str(payload.get("text") or "").strip()
        provenance_source = str(((unit or {}).get("provenance") or {}).get("source_text") or "").strip()
        source_text = provenance_source or str((unit or {}).get("render_text") or "").strip() or raw_text
        if str(slot.slot_type) in {"ordering_event", "comparison_operand", "temporal_event", "temporal_time"} and raw_text:
            source_text = raw_text
        display_text = display_slot_text(slot, payload, unit, getattr(plan, "plan_metadata", None))
        binding = dict(payload)
        binding["text"] = display_text
        binding["display_text"] = display_text
        binding["source_span_text"] = source_text
        binding["source_text"] = source_text
        rendered[slot_name] = binding
    return rendered



from .focus import *  # noqa: F401,F403
from .entities import *  # noqa: F401,F403
from .semantic import (  # noqa: F401
    _binding_semantic_support,
    _has_first_person_signal,
    _query_needs_personal_semantic_gate,
    _resolve_compiler_score_weights,
    _score_feature_bundle,
    _semantic_sufficiency_analysis,
    _strong_direct_span_source,
    _unit_has_strong_focus_alignment,
    _unit_semantic_gate_allows_anchor,
    _unit_semantic_support,
    score_evidence_unit as _semantic_score_evidence_unit,
)
from ..validation.compatibility import (  # noqa: F401
    _best_matching_entity,
    _binding_focus_overlap_tokens,
    _binding_matches_query_focus,
    _binding_quality_score,
    _binding_value_from_session_header,
    _direct_value_requires_entity_match,
    _event_matches_expected,
    _is_generic_fragment,
    _slot_allows_descriptive_fragment,
    _slot_attribute_compatible,
    _slot_entity_match,
    _slot_expected_entities,
)

__all__ = [name for name in globals() if not name.startswith("__")]
