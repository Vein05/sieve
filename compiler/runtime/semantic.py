"""Semantic scoring helpers for the compiler runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evidence.schema import EvidencePlan, slot_requirement_name
from evidence.units import EvidenceUnit
from retrieval.query_targets import QueryTargets, extract_query_targets
from ..execution.requirements import (
    is_current_state_schema,
    is_direct_lookup_schema,
    is_state_update_schema,
    required_binding_names as _required_binding_names,
)
from .entities import _best_matching_entity
from .focus import _WEAK_QUERY_FOCUS_TOKENS, _query_focus_tokens
from .roles import _missing_role_gain, _role_priority
from .text import (
    _binding_source_text,
    _binding_token_sets,
    _build_compiled_unit_analysis,
    _query_normalized_text,
)

_DEFAULT_COMPILER_SCORE_WEIGHTS: dict[str, float] = {
    "lexical_overlap": 1.5,
    "entity_overlap": 1.0,
    "numeric_bonus": 1.0,
    "temporal_bonus": 1.0,
    "update_bonus": 1.0,
    "family_bonus": 1.0,
    "answer_shape_bonus": 1.0,
    "answer_memory_bonus": 1.0,
    "semantic_support_bonus": 1.5,
    "autobiographical_bonus": 0.2,
    "typed_schema_bonus": 1.2,
    "generic_state_penalty": -1.5,
    "rank_prior": 1.0,
    "token_count": -0.02,
}


def _resolve_compiler_score_weights(context: dict[str, Any]) -> dict[str, float]:
    weights = dict(_DEFAULT_COMPILER_SCORE_WEIGHTS)
    overrides = context.get("sieve_score_weights")
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            try:
                weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return weights


def _score_feature_bundle(*, features: dict[str, float], weights: dict[str, float]) -> float:
    return sum(float(features.get(name, 0.0)) * float(weight) for name, weight in weights.items())


def _unit_tags(
    *,
    row: dict[str, Any],
    query_family: str,
    unit: EvidenceUnit,
    query_targets: QueryTargets,
) -> set[str]:
    del row, query_family
    tags: set[str] = set()
    analysis = _build_compiled_unit_analysis(str(unit.provenance.get("source_text") or unit.render_text or ""))
    source_text = analysis.body_text
    normalized = analysis.normalized_source
    if str(unit.memory_id).startswith("answer_"):
        tags.add("answer_memory")
    if unit.entity_tokens:
        tags.add("entity_overlap")
    if unit.numeric_values:
        tags.add("numeric")
    if unit.time_markers:
        tags.add("temporal")
    primary_entities = tuple(entity for entity in query_targets.subject_entities if entity) or tuple(
        entity for entity in query_targets.candidate_entities if entity
    )
    if primary_entities and _best_matching_entity(source_text, primary_entities):
        tags.add("entity_match")
    if any(marker in normalized for marker in (" current ", " now ", " switched ", " changed ")):
        tags.add("update")
    return tags


def score_evidence_unit(
    *,
    row: dict[str, Any],
    query_family: str,
    unit: EvidenceUnit,
    memory_labels_by_id: dict[str, dict[str, Any]],
    score_weights: dict[str, float],
    plan: EvidencePlan | None = None,
) -> float:
    query_targets = extract_query_targets(row, query_family)
    analysis = _build_compiled_unit_analysis(str(unit.provenance.get("source_text") or unit.render_text or ""))
    source_text = analysis.body_text
    primary_entities = tuple(entity for entity in query_targets.subject_entities if entity) or tuple(
        entity for entity in query_targets.candidate_entities if entity
    )
    unit_tokens = set(analysis.content_tokens) | set(analysis.named_tokens)
    plan_kind = str(getattr(plan, "plan_metadata", {}).get("plan_kind") or "")
    preferred_attribute = str(getattr(plan, "plan_metadata", {}).get("preferred_attribute") or "").strip().lower()
    schema_hints = unit.provenance.get("schema_hints", {}) if isinstance(unit.provenance, Mapping) else {}
    unit_plan_kinds = set(schema_hints.get("plan_kinds", [])) if isinstance(schema_hints, Mapping) else set()
    unit_attribute = str(schema_hints.get("attribute_key") or "").strip().lower() if isinstance(schema_hints, Mapping) else ""
    typed_schema_bonus = 0.0
    generic_state_penalty = 0.0

    def _attribute_compatible(expected_attribute: str, candidate_attribute: str) -> bool:
        if not expected_attribute or not candidate_attribute:
            return False
        if expected_attribute == candidate_attribute:
            return True
        if expected_attribute == "name" and candidate_attribute in {"reading", "watching", "studying", "using", "employer"}:
            return True
        if expected_attribute == "brand" and candidate_attribute == "using":
            return True
        if expected_attribute in {"duration", "time"} and candidate_attribute in {"duration", "time"}:
            return True
        return False

    if plan_kind and plan_kind in unit_plan_kinds:
        typed_schema_bonus += 1.0
    if preferred_attribute and unit_attribute and _attribute_compatible(preferred_attribute, unit_attribute):
        typed_schema_bonus += 1.25
    if plan_kind in {"attribute_lookup", "current_attribute_lookup"} and getattr(unit, "is_generic_state_text", False):
        generic_state_penalty = 1.0
    features = {
        "lexical_overlap": float(len(unit_tokens & _query_focus_tokens(row))),
        "entity_overlap": 1.0 if primary_entities and _best_matching_entity(source_text, primary_entities) else 0.0,
        "numeric_bonus": 1.0 if unit.numeric_values else 0.0,
        "temporal_bonus": 1.0 if unit.time_markers else 0.0,
        "update_bonus": 1.0 if unit.update_markers else 0.0,
        "family_bonus": 1.0,
        "answer_shape_bonus": 1.0 if str(unit.memory_id).startswith("answer_") else 0.0,
        "answer_memory_bonus": 1.0 if "answer_bearing" in set(memory_labels_by_id.get(unit.memory_id, {}).get("labels", [])) else 0.0,
        "semantic_support_bonus": 1.0,
        "autobiographical_bonus": _unit_autobiographical_score(unit, plan_kind),
        "typed_schema_bonus": float(typed_schema_bonus),
        "generic_state_penalty": float(generic_state_penalty),
        "rank_prior": 1.0 / max(1, int(unit.parent_rank)),
        "token_count": float(unit.token_count),
    }
    return _score_feature_bundle(features=features, weights=score_weights)


def _semantic_sufficiency_analysis(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets,
    validated_slot_bindings: dict[str, dict[str, Any] | None],
    slot_grounding_scores: dict[str, float],
    invalid_slots: dict[str, str],
) -> dict[str, Any]:
    required_slot_names = _required_binding_names(plan)
    semantic_support_scores: dict[str, float] = {}
    present_scores: list[float] = []
    for slot_name in required_slot_names:
        binding = validated_slot_bindings.get(slot_name)
        if not isinstance(binding, dict):
            semantic_support_scores[slot_name] = 0.0
            continue
        score = _binding_semantic_support(row=row, plan=plan, query_targets=query_targets, binding=binding)
        semantic_support_scores[slot_name] = round(float(score), 6)
        present_scores.append(float(score))
    hard_invalid = {slot_name: reason for slot_name, reason in invalid_slots.items() if reason != "generic_fragment"}
    slot_grounding_mean = 0.0
    required_grounding = [float(slot_grounding_scores.get(slot_name, 0.0)) for slot_name in required_slot_names if isinstance(validated_slot_bindings.get(slot_name), dict)]
    if required_grounding:
        slot_grounding_mean = sum(required_grounding) / float(len(required_grounding))
    semantic_support_mean = sum(present_scores) / float(len(present_scores)) if present_scores else 0.0
    needs_gate = _query_needs_personal_semantic_gate(row=row, plan=plan, query_targets=query_targets)
    gate_passed = bool((not needs_gate) or (present_scores and not hard_invalid and semantic_support_mean >= 0.5))
    return {
        "semantic_support_scores": semantic_support_scores,
        "semantic_support_mean": round(float(semantic_support_mean), 6),
        "slot_grounding_mean": round(float(slot_grounding_mean), 6),
        "semantic_sufficiency_score": round((0.7 * semantic_support_mean) + (0.3 * min(1.0, slot_grounding_mean / 4.0)), 6),
        "semantic_sufficiency_passed": gate_passed,
        "semantic_sufficiency_applied": needs_gate,
        "semantic_sufficiency_reasons": [] if gate_passed else ["weak_semantic_support"] + (["hard_invalid_binding"] if hard_invalid else []),
    }


def _binding_semantic_support(*, row: dict[str, Any], plan: EvidencePlan, query_targets: QueryTargets, binding: dict[str, Any]) -> float:
    source_text = _binding_source_text(binding)
    support = 0.0
    if str(binding.get("memory_id") or "").startswith("answer_"):
        support += 0.45
    if str(binding.get("speaker") or "").strip().lower() == "user":
        support += 0.35
    if _has_first_person_signal(source_text):
        support += 0.35
    if _strong_direct_span_source(source_text):
        support += 0.25
    primary_entities = tuple(entity for entity in query_targets.subject_entities if entity) or tuple(
        entity for entity in query_targets.candidate_entities if entity
    )
    if primary_entities and _best_matching_entity(source_text, primary_entities):
        support += 0.35
    if not primary_entities and plan.family in {"aggregation", "information_extraction", "single_anchor", "current_state", "temporal"}:
        focus_tokens = _query_focus_tokens(row)
        if _binding_focus_overlap_tokens(binding, focus_tokens) - _WEAK_QUERY_FOCUS_TOKENS:
            support += 0.2
    return min(1.0, support)


_GENERIC_ASSISTANT_RE = __import__("re").compile(
    "(?:can you|could you|do you have|for example|here are|recommend|suggest|"
    "tips?|resources?|consider|transform|try this|achieve)",
    __import__("re").IGNORECASE,
)

_PERSONAL_FACT_PLAN_KINDS = frozenset({
    "attribute_lookup",
    "entity_lookup",
    "current_attribute_lookup",
    "count_items",
    "count_events",
    "count_distinct_items",
})


def _unit_autobiographical_score(unit: EvidenceUnit, plan_kind: str) -> float:
    import re as _re
    if str(unit.memory_id).startswith("answer_"):
        return 0.0
    personal = plan_kind in _PERSONAL_FACT_PLAN_KINDS
    if not personal:
        return 0.0
    speaker = str(unit.speaker or "").strip().lower()
    source = str(
        unit.provenance.get("source_text") if isinstance(unit.provenance, Mapping) else None
    ) or str(unit.render_text or "")
    first_person = bool(_re.search(r"\b(i|i'm|i've|i’d|i'll|me|my|mine|we|we're|our|ours|us)\b", source, _re.IGNORECASE))
    reflects_user_facts = bool(_re.search(r"\b(you|your|you've|you're|you'd)\b", source, _re.IGNORECASE))
    generic_assistant = bool(_GENERIC_ASSISTANT_RE.search(source)) or (
        "?" in source and not reflects_user_facts
    )
    if speaker == "user":
        return 1.0 if first_person else 0.4
    if speaker == "assistant":
        if generic_assistant:
            return -1.0
        return 0.0
    return 0.0


def _has_first_person_signal(text: str) -> bool:
    import re
    return bool(re.search(r"\b(i|i'm|i've|i’d|i'll|me|my|mine|we|we're|our|ours|us)\b", str(text or ""), re.IGNORECASE))


def _strong_direct_span_source(source_text: str) -> bool:
    import re
    text = str(source_text or "").strip()
    if not text:
        return False
    return any(
        pattern.search(text)
        for pattern in (
            re.compile(r"\bdegree\s+in\s+[A-Z]", re.IGNORECASE),
            re.compile(r"\b(?:currently|current(?:ly)?)\s+(?:at|with)\s+[A-Z]", re.IGNORECASE),
            re.compile(r"\bworking\s+at\s+[A-Z]", re.IGNORECASE),
        )
    )


def _binding_focus_overlap_tokens(binding: dict[str, Any], focus_tokens: set[str]) -> set[str]:
    if not focus_tokens:
        return set()
    content_tokens, named_tokens = _binding_token_sets(binding)
    binding_tokens = content_tokens | named_tokens
    return set(binding_tokens & focus_tokens)


def _unit_semantic_support(*, row: dict[str, Any], query_family: str, unit: EvidenceUnit) -> float:
    query_targets = extract_query_targets(row, query_family)
    source_text = str(unit.provenance.get("source_text") or unit.render_text or "").strip()
    support = 0.0
    if str(unit.memory_id).startswith("answer_"):
        support += 0.45
    if str(unit.speaker).strip().lower() == "user":
        support += 0.35
    if _has_first_person_signal(source_text):
        support += 0.35
    if _strong_direct_span_source(source_text):
        support += 0.25
    primary_entities = tuple(entity for entity in query_targets.subject_entities if entity) or tuple(
        entity for entity in query_targets.candidate_entities if entity
    )
    if primary_entities and _best_matching_entity(source_text, primary_entities):
        support += 0.35
    return min(1.0, support)


def _query_needs_personal_semantic_gate(*, row: dict[str, Any], plan: EvidencePlan, query_targets: QueryTargets) -> bool:
    if not (is_direct_lookup_schema(plan) or is_current_state_schema(plan) or is_state_update_schema(plan)):
        return False
    normalized_query = _query_normalized_text(row)
    if _has_first_person_signal(normalized_query):
        return True
    return len(tuple(entity for entity in query_targets.subject_entities if entity)) == 0


def _unit_has_strong_focus_alignment(*, row: dict[str, Any], unit: EvidenceUnit) -> bool:
    focus_tokens = _query_focus_tokens(row)
    if not focus_tokens:
        return False
    analysis = _build_compiled_unit_analysis(str(unit.provenance.get("source_text") or unit.render_text or ""))
    overlap = (set(analysis.content_tokens) | set(analysis.named_tokens)) & focus_tokens
    return bool(overlap - _WEAK_QUERY_FOCUS_TOKENS)


def _unit_semantic_gate_allows_anchor(*, row: dict[str, Any], plan: EvidencePlan, query_targets: QueryTargets, unit: EvidenceUnit, tags: set[str], expansion: bool = False) -> bool:
    gated_roles = {"direct_anchor", "support_anchor", "state_anchor", "current_resolution", "old_state", "new_state"}
    if not tags.intersection(gated_roles):
        return True
    if not _query_needs_personal_semantic_gate(row=row, plan=plan, query_targets=query_targets):
        return True
    source_text = str(unit.provenance.get("source_text") or unit.render_text or "").strip()
    if _strong_direct_span_source(source_text):
        return True
    # During expansion, the candidate is already MiniLM-ranked as relevant.
    # Focus alignment alone is sufficient — don't also require speaker/first-
    # person support, which blocks assistant-spoken evidence that may contain
    # the answer.
    if expansion:
        return _unit_has_strong_focus_alignment(row=row, unit=unit)
    return _unit_semantic_support(row=row, query_family=plan.query_family, unit=unit) >= 0.5 and _unit_has_strong_focus_alignment(row=row, unit=unit)
