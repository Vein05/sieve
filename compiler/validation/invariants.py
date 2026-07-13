"""Schema-grounding invariants for compiler validation."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

from evidence.schema import EvidencePlan
from retrieval.query_targets import QueryTargets
from ..execution.requirements import (
    aggregate_requires_two_operands,
    aggregate_slot,
    is_comparison_schema,
    is_direct_lookup_schema,
    is_numeric_aggregate_schema,
    required_binding_names,
)

_MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_OPEN_ENDED_SUPPORT_QUERY_MARKERS = (
    " any tips",
    " any suggestions",
    " any advice",
    " suggest ",
    " suggestion",
    " suggestions",
    " recommend ",
    " recommendation",
    " do you think it might ",
    " do you think i should ",
    " what should i ",
    " help me decide ",
)


def _deterministic_slot_score_floor(plan: EvidencePlan) -> float:
    if is_numeric_aggregate_schema(plan):
        return 4.5
    if is_direct_lookup_schema(plan):
        return 4.0
    if plan.family in {"current_state", "knowledge_update", "conflict_update"}:
        return 3.5
    if plan.schema_name in {"OrderedChoice", "TemporalInterval", "RelativeTime"} or is_comparison_schema(plan):
        return 1.5
    return 2.5


def _reader_first_schema(plan: EvidencePlan) -> bool:
    """Return True when the schema requires reader-first grounding analysis.

    Exception: TemporalInterval/RelativeTime with an injected question_date have
    a precomputed hint — allow them to reach the deterministic_safe path so
    execute_direct_duration_from_compiled_units or execute_relative_time can fire.
    """
    if not (
        is_numeric_aggregate_schema(plan)
        or is_comparison_schema(plan)
        or plan.schema_name in {"TemporalInterval", "RelativeTime"}
    ):
        return False
    if plan.schema_name in {"TemporalInterval", "RelativeTime"}:
        question_date = str((getattr(plan, "plan_metadata", None) or {}).get("question_date") or "").strip()
        if question_date:
            return False
    return True


def _is_open_ended_support_query(normalized_query: str) -> bool:
    padded = f" {normalized_query.strip()} "
    if any(marker in padded for marker in _OPEN_ENDED_SUPPORT_QUERY_MARKERS):
        return True
    return False


def schema_grounding_analysis(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets,
    validated_bindings: dict[str, dict[str, Any] | None],
    displayed_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
    normalize_text: Any,
    binding_source_text: Any,
    binding_quality_score: Any,
    is_generic_fragment: Any,
    slot_allows_descriptive_fragment: Any,
    slot_expected_entities: Any,
    slot_entity_match: Any,
    parse_numeric_text: Any,
    contains_math_expression: Any,
    binding_value_from_session_header: Any,
    query_expects_time_literal: Any,
    event_matches_expected: Any,
) -> dict[str, Any]:
    slot_grounding_scores: dict[str, float] = {}
    slot_grounding_fail_reasons: dict[str, list[str]] = {}
    slot_entity_hits: dict[str, str] = {}
    required_slot_names = required_binding_names(plan)
    has_valid_binding = any(isinstance(binding, dict) for binding in validated_bindings.values())
    has_raw_binding = any(isinstance(binding, dict) for binding in displayed_bindings.values())
    required_present = all(isinstance(validated_bindings.get(slot_name), dict) for slot_name in required_slot_names)
    deterministically_safe = required_present and not invalid_slots
    if not has_valid_binding and not has_raw_binding:
        # Route to reader instead of abstaining — the reader can still use
        # compiled evidence lines even when no slot bindings survived validation.
        return {
            "answerability_level": "reader_only",
            "deterministic_allowed": False,
            "reader_route_reason": "no_usable_bindings",
            "slot_grounding_scores": slot_grounding_scores,
            "slot_grounding_fail_reasons": slot_grounding_fail_reasons,
        }

    normalized_query = normalize_text(str(row.get("query", "")))
    schema_name = str(plan.schema_name or "").strip()
    open_ended_support_query = _is_open_ended_support_query(normalized_query)
    question_subtype = "preference_summary" if (query_targets.asks_for_recall_support or open_ended_support_query) else "factual"
    if schema_name in {"DirectValue+Support", "AttributeLookup+Support", "EntityLookup+Support"} and query_targets.asks_for_recall_support:
        question_subtype = "single_anchor_preference"

    slots_to_analyze = [slot for slot in plan.slots if slot.required]
    if aggregate_requires_two_operands(plan):
        aggregate_plan_slot = aggregate_slot(plan)
        if aggregate_plan_slot is not None:
            slots_to_analyze = [
                replace(aggregate_plan_slot, slot_name="operand_1", is_variadic=False),
                replace(aggregate_plan_slot, slot_name="operand_2", is_variadic=False),
            ]

    for slot in slots_to_analyze:
        binding = validated_bindings.get(slot.slot_name)
        if not isinstance(binding, dict):
            fail_reason = invalid_slots.get(slot.slot_name) or "missing"
            slot_grounding_fail_reasons[slot.slot_name] = [fail_reason]
            slot_grounding_scores[slot.slot_name] = 0.0
            deterministically_safe = False
            continue

        source_text = binding_source_text(binding)
        display_text = str(binding.get("display_text") or binding.get("text") or "").strip()
        slot_score = binding_quality_score(binding)
        slot_reasons: list[str] = []
        normalized_display = normalize_text(display_text)
        normalized_source = normalize_text(source_text)

        if not slot_allows_descriptive_fragment(str(slot.slot_type)) and is_generic_fragment(display_text, source_text):
            slot_reasons.append("generic_fragment")
        if any(token in normalized_display for token in ("can", "what", "which", "who", "when", "where", "why")) and len(normalized_display.split()) <= 2:
            slot_reasons.append("answer_not_span")

        # General query-echo rejection: if the extracted display text is a
        # contiguous substring of the query, the compiler echoed the question
        # entity instead of extracting the answer.  Only flag multi-token spans
        # to avoid false positives on short common words.
        if (
            normalized_display
            and len(normalized_display.split()) >= 2
            and f" {normalized_display} " in f" {normalized_query} "
            and slot.slot_type in {"direct_value", "current_resolution", "new_state", "old_state", "numeric_operand"}
        ):
            slot_reasons.append("echoed_entity")

        if plan.schema_name == "OrderedChoice":
            if re.fullmatch(r"(?:\d{1,4}(?:/\d{1,2}){1,2}|[a-z]+(?:\s+[a-z]+)?)", normalized_display):
                if normalized_display in _MONTH_NAME_TO_NUMBER or any(month == normalized_display for month in _MONTH_NAME_TO_NUMBER):
                    slot_reasons.append("month_token")
            if re.fullmatch(r"(?:\d{4}/\d{1,2}/\d{1,2})", normalized_display):
                pass
            elif re.fullmatch(r"(?:\d{1,4}(?:/\d{1,2}){1,2})", normalized_display):
                slot_reasons.append("month_token")
            if slot_expected_entities(slot, query_targets) and not slot_entity_match(slot, source_text, query_targets):
                slot_reasons.append("choice_entity_mismatch")

        elif plan.schema_name == "TemporalInterval":
            if slot.slot_name in {"event_a", "event_b"} and slot_expected_entities(slot, query_targets):
                expected = slot_expected_entities(slot, query_targets)[0]
                matched_entity = slot_entity_match(slot, source_text, query_targets)
                if matched_entity:
                    slot_entity_hits[slot.slot_name] = str(matched_entity)
                if expected and not event_matches_expected(binding, expected):
                    slot_reasons.append("event_entity_mismatch")
            if slot.slot_name in {"time_a", "time_b"} and slot_expected_entities(slot, query_targets):
                matched_entity = slot_entity_match(slot, source_text, query_targets)
                if matched_entity:
                    slot_entity_hits[slot.slot_name] = str(matched_entity)
                if not matched_entity:
                    slot_reasons.append("time_entity_mismatch")
            if slot.slot_name in {"time_a", "time_b"} and any(token in normalized_display for token in ("can", "what")):
                slot_reasons.append("time_span_pollution")

        elif is_comparison_schema(plan):
            if slot_expected_entities(slot, query_targets) and not slot_entity_match(slot, source_text, query_targets):
                slot_reasons.append("comparison_entity_mismatch")
            if parse_numeric_text(display_text) is None and parse_numeric_text(source_text) is None:
                slot_reasons.append("numeric_pollution")
            if contains_math_expression(source_text) and not re.search(r"\b(more|less|higher|lower|cheaper|pricier)\b", normalized_source):
                if query_targets.candidate_entities:
                    slot_reasons.append("numeric_pollution")

        elif is_numeric_aggregate_schema(plan):
            if parse_numeric_text(display_text) is None and parse_numeric_text(source_text) is None:
                slot_reasons.append("numeric_pollution")
            if contains_math_expression(source_text) and not query_targets.candidate_entities:
                if len(normalized_display.split()) > 4:
                    slot_reasons.append("numeric_pollution")

        elif is_direct_lookup_schema(plan):
            if re.fullmatch(r"\d{1,4}(?:/\d{1,2}){1,2}", normalized_display):
                slot_reasons.append("date_literal")
            if not query_expects_time_literal(row) and binding_value_from_session_header(display_text, source_text):
                slot_reasons.append("header_hijack")
            if not query_expects_time_literal(row) and re.fullmatch(r"\d{1,2}:\d{2}", normalized_display):
                slot_reasons.append("time_literal")
            if is_generic_fragment(display_text, source_text):
                slot_reasons.append("generic_fragment")
            # EntityLookup entity-match invariant: for "which X" questions the
            # answer must mention one of the candidate entities.  If the display
            # text has no token overlap with any candidate, the compiler grabbed
            # a tangential span (e.g. "Best Buy store" for "which device first,
            # Samsung or Dell?").
            if plan.schema_name in {"EntityLookup", "EntityLookup+Support"} and query_targets.candidate_entities:
                display_tokens = set(normalized_display.split())
                entity_matched = False
                for entity in query_targets.candidate_entities:
                    entity_norm = normalize_text(str(entity or ""))
                    entity_tokens = set(entity_norm.split()) - {"the", "a", "an", "my", "first", "event", "item", "device", "gift", "project"}
                    if entity_tokens and (entity_tokens & display_tokens):
                        entity_matched = True
                        break
                if not entity_matched:
                    slot_reasons.append("entity_lookup_no_match")
            if plan.family in {"single_anchor", "information_extraction"} and query_targets.asks_for_recall_support:
                if len(normalized_source.split()) > 18 and any(marker in normalized_source for marker in ("recommend", "suggest", "advice")):
                    slot_reasons.append("preference_summary")
            if open_ended_support_query:
                slot_reasons.append("open_ended_support_query")

        slot_grounding_scores[slot.slot_name] = round(float(slot_score), 6)
        if slot_reasons:
            slot_grounding_fail_reasons[slot.slot_name] = list(dict.fromkeys(slot_reasons))
            deterministically_safe = False
        if slot.slot_name in invalid_slots:
            deterministically_safe = False

    # Cross-slot diversity check: for paired schemas, both operand slots must bind to
    # DIFFERENT memory_ids. If they share the same source, one operand is fabricated
    # (e.g. "fish in aquarium 1" and "fish in aquarium 2" both from one memory → 1+2=3
    # instead of the real 17). This is a structural invariant, not a heuristic threshold.
    _PAIRED_SCHEMAS = {"SumOperands", "TemporalInterval", "DifferenceAggregate", "PercentageAggregate"}
    if plan.schema_name in _PAIRED_SCHEMAS:
        _PAIR_SLOT_NAMES = (
            ("operand_1", "operand_2"),
            ("event_a", "event_b"),
            ("time_a", "time_b"),
        )
        for slot_a_name, slot_b_name in _PAIR_SLOT_NAMES:
            binding_a = validated_bindings.get(slot_a_name)
            binding_b = validated_bindings.get(slot_b_name)
            if not isinstance(binding_a, dict) or not isinstance(binding_b, dict):
                continue
            mid_a = str(binding_a.get("memory_id") or "").strip()
            mid_b = str(binding_b.get("memory_id") or "").strip()
            uid_a = str(binding_a.get("unit_id") or "").strip()
            uid_b = str(binding_b.get("unit_id") or "").strip()
            # Same memory_id + same or sibling unit_id → same-source binding
            if mid_a and mid_b and mid_a == mid_b:
                # Allow if units are distinct objects from different parts of the memory
                # (different object index, e.g. :o00 vs :o03), but not if trivially close
                obj_idx_a = uid_a.rsplit(":o", 1)[-1] if ":o" in uid_a else ""
                obj_idx_b = uid_b.rsplit(":o", 1)[-1] if ":o" in uid_b else ""
                if obj_idx_a == obj_idx_b or not obj_idx_a or not obj_idx_b:
                    slot_grounding_fail_reasons.setdefault(slot_b_name, []).append("same_source_operand")
                    deterministically_safe = False

    if plan.schema_name == "TemporalInterval":
        event_a_hit = normalize_text(slot_entity_hits.get("event_a", ""))
        event_b_hit = normalize_text(slot_entity_hits.get("event_b", ""))
        if event_a_hit and event_b_hit and event_a_hit == event_b_hit:
            slot_grounding_fail_reasons.setdefault("event_b", []).append("missing_counterpart")
            deterministically_safe = False

    # For numeric reasoning families, the compiler's job is to surface a small,
    # answer-sufficient package for a bounded reader rather than to prove that
    # the full computation should be emitted deterministically.
    if deterministically_safe and _reader_first_schema(plan):
        deterministically_safe = False
        slot_grounding_fail_reasons.setdefault("__route__", []).append("reader_first_schema")

    if deterministically_safe:
        # --- Multi-signal deterministic confidence gate ---
        # Deterministic extraction is only safe when multiple independent signals
        # agree. Each signal contributes a binary vote; deterministic fires only
        # when all required signals pass.
        #
        # Signal 1: Binding quality floor (schema-adaptive)
        score_floor = _deterministic_slot_score_floor(plan)
        required_scores = [
            float(slot_grounding_scores.get(slot_name, 0.0))
            for slot_name in required_slot_names
            if isinstance(validated_bindings.get(slot_name), dict)
        ]
        if not required_scores or min(required_scores) < score_floor:
            deterministically_safe = False
            for slot_name, score in (
                (slot_name, float(slot_grounding_scores.get(slot_name, 0.0)))
                for slot_name in required_slot_names
                if isinstance(validated_bindings.get(slot_name), dict)
            ):
                if score >= score_floor:
                    continue
                slot_grounding_fail_reasons.setdefault(slot_name, []).append("deterministic_confidence_low")

    if deterministically_safe:
        # Signal 2: Zero grounding failures on all required slots.
        # Any slot with a fail reason means the binding is uncertain —
        # the reader can judge better than a deterministic extraction.
        for slot_name in required_slot_names:
            if slot_grounding_fail_reasons.get(slot_name):
                deterministically_safe = False
                slot_grounding_fail_reasons.setdefault(slot_name, []).append("deterministic_fail_reason_present")
                break

    if deterministically_safe:
        # Signal 3: Schema structural complexity — multi-slot required schemas
        # are structurally unsuited for deterministic extraction because the
        # compiler must correctly bind *all* slots and compose them.
        # Only single required-slot schemas are safe for deterministic.
        required_with_bindings = [
            slot_name for slot_name in required_slot_names
            if isinstance(validated_bindings.get(slot_name), dict)
        ]
        if len(required_slot_names) > 1 and len(required_with_bindings) > 1:
            deterministically_safe = False
            slot_grounding_fail_reasons.setdefault("__route__", []).append("multi_slot_complexity")

    if deterministically_safe:
        # Signal 4: Retriever-compiler agreement — the compiled memory should
        # be the retriever's top-ranked candidate.  If a higher-ranked memory
        # exists that the compiler didn't select, the compiler may have latched
        # onto the wrong memory (e.g. old value for knowledge-update, partial
        # evidence for multi-session).  Route to reader so the LLM can
        # adjudicate across all evidence.
        candidate_memories = row.get("candidate_memories") or []
        if candidate_memories:
            pool_order = [
                str(cm.get("memory_id") or "") for cm in candidate_memories
            ]
            compiled_mem_ids = {
                str(b.get("memory_id") or "")
                for b in validated_bindings.values()
                if isinstance(b, dict) and b.get("memory_id")
            }
            compiled_ranks = [
                pool_order.index(mid)
                for mid in compiled_mem_ids
                if mid in pool_order
            ]
            best_compiled_rank = min(compiled_ranks) if compiled_ranks else len(pool_order)
            if best_compiled_rank >= 1:
                deterministically_safe = False
                slot_grounding_fail_reasons.setdefault("__route__", []).append(
                    "retriever_compiler_disagreement"
                )

    if deterministically_safe:
        return {
            "answerability_level": "deterministic_safe",
            "deterministic_allowed": True,
            "reader_route_reason": "schema_grounding_passed",
            "slot_grounding_scores": slot_grounding_scores,
            "slot_grounding_fail_reasons": slot_grounding_fail_reasons,
            "question_subtype": question_subtype,
        }

    if has_valid_binding or has_raw_binding:
        return {
            "answerability_level": "reader_only",
            "deterministic_allowed": False,
            "reader_route_reason": (
                "schema_grounding_partial"
                if slot_grounding_fail_reasons
                else "raw_slot_available"
            ),
            "slot_grounding_scores": slot_grounding_scores,
            "slot_grounding_fail_reasons": slot_grounding_fail_reasons,
            "question_subtype": question_subtype,
        }

    return {
        "answerability_level": "abstain",
        "deterministic_allowed": False,
        "reader_route_reason": "hard_validation_missing",
        "slot_grounding_scores": slot_grounding_scores,
        "slot_grounding_fail_reasons": slot_grounding_fail_reasons,
        "question_subtype": question_subtype,
    }
