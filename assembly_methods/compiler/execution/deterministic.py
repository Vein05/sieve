"""Deterministic answer-contract orchestration for compiler execution."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Callable

from ...evidence_schema import EvidencePlan
from . import requirements as req
from .requirements import (
    aggregate_requires_two_operands,
    aggregate_slot,
    is_comparison_schema,
    is_count_aggregate_schema,
    is_numeric_aggregate_schema,
    required_binding_names,
)
from .router import normalize_post_contract


def _plan_metadata_text(plan: EvidencePlan, key: str) -> str:
    return str(getattr(plan, "plan_metadata", {}).get(key) or "").strip().lower()


def _deterministic_numeric_allowed(
    *,
    plan: EvidencePlan,
    slot_name: str,
    binding: dict[str, Any],
    parse_numeric_text: Callable[[str], tuple[float, str, str] | None],
) -> bool:
    answer_shape = _plan_metadata_text(plan, "answer_shape")
    display_text = str(binding.get("display_text") or binding.get("text") or "").strip()
    source_text = str(binding.get("text") or "").strip()
    parsed = parse_numeric_text(display_text or source_text)
    if parsed is None and not re.search(r"\d", f"{display_text} {source_text}"):
        return False

    numeric_plan = (
        plan.requires_numeric_reasoning
        or answer_shape in {"numeric", "numeric_span"}
        or req.is_numeric_aggregate_schema(plan)
        or req.is_comparison_schema(plan)
        or plan.schema_name in {"TemporalInterval", "RelativeTime"}
    )
    if slot_name == "numeric_value":
        if not numeric_plan:
            return False
        if answer_shape in {"entity_span", "attribute_span", "resolved_state"}:
            return False
    return True


def _reader_first_schema(plan: EvidencePlan, *, compiled_units: list | None = None) -> bool:
    """Return True when the schema requires reader-first handling.

    Exception: if the compiled package already contains a precomputed answer
    (computed_date_hint, operand_summary, or a clean aggregate), allow the
    deterministic path to emit it directly instead of re-routing to the reader.
    These rows already have the answer — the reader is redundant.
    """
    if not (
        req.is_numeric_aggregate_schema(plan)
        or req.is_comparison_schema(plan)
        or req.is_count_aggregate_schema(plan)
        or plan.schema_name in {"TemporalInterval", "RelativeTime"}
    ):
        return False
    # If there's a precomputed hint we'll let the deterministic branch handle it.
    # The hint is attached to compiled_units via plan_metadata at render time;
    # we detect it through the plan_metadata key set at pipeline injection.
    if compiled_units is not None:
        if plan.schema_name in {"TemporalInterval", "RelativeTime"}:
            question_date = str((getattr(plan, "plan_metadata", None) or {}).get("question_date") or "").strip()
            if question_date:
                # Pipeline injected a question_date → computed_date_hint will be rendered.
                # Allow the deterministic aggregate executor a chance to fire.
                return False
    return True


def _canonicalize_count_answer_text(
    *,
    answer_text: str | None,
    parse_numeric_text: Callable[[str], tuple[float, str, str] | None],
) -> str | None:
    text = str(answer_text or "").strip()
    if not text:
        return None
    parsed = parse_numeric_text(text)
    if parsed is None:
        leading_match = re.match(
            r"^\s*((?:\d[\d,]*(?:\.\d+)?)|(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
            r"|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty))\b",
            text,
            re.IGNORECASE,
        )
        if leading_match:
            parsed = parse_numeric_text(str(leading_match.group(1) or "").strip())
        if parsed is None:
            leading_token = str((text.split() or [""])[0]).strip().lower()
            from ..runtime.parsing import _parse_number_word
            num = _parse_number_word(leading_token)
            if num is not None:
                parsed = (float(num), "", "")
    if parsed is None:
        return text
    value, prefix, unit = parsed
    if prefix or unit:
        return text
    if float(value).is_integer():
        return str(int(value))
    return str(value).rstrip("0").rstrip(".")


def _rescue_numeric_value_from_direct_binding(
    *,
    plan: EvidencePlan,
    validated_bindings: dict[str, dict[str, Any] | None],
    displayed_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
    binding_source_text: Callable[[dict[str, Any]], str],
    parse_numeric_text: Callable[[str], tuple[float, str, str] | None],
) -> None:
    if not plan.requires_numeric_reasoning:
        return
    if isinstance(validated_bindings.get("numeric_value"), dict):
        return
    source_binding = validated_bindings.get("direct_value") or displayed_bindings.get("direct_value")
    if not isinstance(source_binding, dict):
        return
    source_text = binding_source_text(source_binding)
    if not source_text:
        return
    numeric_match = re.search(
        r"(?:[$€£]\s*\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent(?:age)?|mph|km/h|km|kilometers?|kilometres?"
        r"|miles?|hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|years?|yrs?"
        r"|bucks?|dollars?|pounds?|lbs?|kilograms?|kgs?|grams?|liters?|litres?|gallons?"
        r"|calories?|kcal|feet|foot|ft|inches?|in|meters?|metres?|centimeters?|cm))\b",
        source_text,
        re.IGNORECASE,
    )
    if not numeric_match:
        return
    numeric_text = numeric_match.group(0).strip().rstrip(".,;:")
    if parse_numeric_text(numeric_text) is None:
        return
    rescued = dict(source_binding)
    rescued["text"] = numeric_text
    rescued["display_text"] = numeric_text
    validated_bindings["numeric_value"] = rescued
    invalid_slots.pop("numeric_value", None)


def _rescue_duration_bindings(
    *,
    row: dict[str, Any],
    displayed_bindings: dict[str, dict[str, Any] | None],
    validated_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
) -> None:
    def _allow_duration_rescue(slot_name: str) -> bool:
        reason = str(invalid_slots.get(slot_name) or "").strip().lower()
        return reason in {"", "missing", "numeric_pollution", "wrong_attribute"}

    normalized_query = str(row.get("query") or "").strip().lower()
    if not (
        normalized_query.startswith("how long")
        or re.search(r"\bhow\s+many\s+(days?|weeks?|months?|years?|hours?|minutes?)\b", normalized_query)
    ):
        return

    duration_source = None
    for slot_name in ("direct_value", "numeric_value", "event"):
        candidate = validated_bindings.get(slot_name) or displayed_bindings.get(slot_name)
        if isinstance(candidate, dict) and str(candidate.get("duration_value") or "").strip():
            duration_source = candidate
            break
    if duration_source is None:
        return

    duration_text = str(duration_source.get("duration_value") or "").strip()
    if not duration_text:
        return

    rescued = dict(duration_source)
    rescued["text"] = duration_text
    rescued["display_text"] = duration_text
    if not isinstance(validated_bindings.get("direct_value"), dict) and _allow_duration_rescue("direct_value"):
        validated_bindings["direct_value"] = rescued
        invalid_slots.pop("direct_value", None)
    if not isinstance(validated_bindings.get("numeric_value"), dict) and _allow_duration_rescue("numeric_value"):
        validated_bindings["numeric_value"] = rescued
        invalid_slots.pop("numeric_value", None)


def _rescue_temporal_interval_events_from_time_bindings(
    *,
    plan: EvidencePlan,
    validated_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
) -> None:
    if str(plan.schema_name or "") != "TemporalInterval":
        return
    for event_slot, time_slot in (("event_a", "time_a"), ("event_b", "time_b")):
        if isinstance(validated_bindings.get(event_slot), dict):
            continue
        time_binding = validated_bindings.get(time_slot)
        if not isinstance(time_binding, dict):
            continue
        rescued = dict(time_binding)
        validated_bindings[event_slot] = rescued
        invalid_slots.pop(event_slot, None)


def build_answer_contract(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: Any,
    compiled_units: list[dict[str, Any]],
    raw_slot_bindings: dict[str, dict[str, Any] | None],
    slot_bindings_with_display_text: Callable[..., dict[str, dict[str, Any] | None]],
    query_focus_tokens: Callable[[dict[str, Any]], set[str]],
    validate_slot_binding: Callable[..., tuple[dict[str, Any] | None, str | None]],
    infer_missing_slot_invalid_reason: Callable[..., str | None],
    promote_support_grounding: Callable[..., None],
    apply_reference_time_fallback: Callable[..., None],
    rescue_current_state_where_binding: Callable[..., None],
    rescue_information_extraction_where_binding: Callable[..., None],
    rescue_temporal_when_event_binding: Callable[..., None],
    binding_source_text: Callable[[dict[str, Any]], str],
    extract_numeric_mentions: Callable[[str], list[tuple[float, str, str, str]]],
    count_numeric_spans_excluding_dates: Callable[[str], int],
    schema_grounding_analysis: Callable[..., dict[str, Any]],
    execute_distinct_count_from_compiled_units: Callable[..., tuple[str | None, list[str]]],
    execute_percentage_aggregate_from_bindings: Callable[..., tuple[str | None, list[str]]],
    execute_aggregate_from_compiled_units: Callable[..., tuple[str | None, list[str]]],
    execute_direct_duration_from_compiled_units: Callable[..., tuple[str | None, list[str]]],
    execute_relative_time: Callable[..., tuple[str | None, list[str]]],
    execute_ordered_choice: Callable[..., tuple[str | None, list[str]]],
    execute_comparison: Callable[..., tuple[str | None, list[str]]],
    execute_temporal_interval: Callable[..., tuple[str | None, list[str]]],
    deterministic_span_allowed: Callable[..., bool],
    parse_numeric_text: Callable[[str], tuple[float, str, str] | None],
    format_numeric_answer: Callable[[float, str, str], str],
    soft_invalid_reasons: set[str],
) -> dict[str, Any]:
    def _append_route_reason(current: str, extra: str) -> str:
        current = str(current or "").strip()
        extra = str(extra or "").strip()
        if not current:
            return extra
        if not extra or extra in current.split("+"):
            return current
        return f"{current}+{extra}"

    displayed_bindings = slot_bindings_with_display_text(
        plan=plan,
        compiled_units=compiled_units,
        slot_bindings=raw_slot_bindings,
    )
    focus_tokens = query_focus_tokens(row)
    slot_by_name = {slot.slot_name: slot for slot in plan.slots}
    required_slot_names = {slot.slot_name for slot in plan.slots if slot.required}
    if aggregate_requires_two_operands(plan):
        plan_aggregate_slot = aggregate_slot(plan)
        if plan_aggregate_slot is not None:
            slot_by_name.setdefault("aggregate_items", plan_aggregate_slot)
            slot_by_name["operand_1"] = replace(plan_aggregate_slot, slot_name="operand_1", is_variadic=False)
            slot_by_name["operand_2"] = replace(plan_aggregate_slot, slot_name="operand_2", is_variadic=False)
    validated_bindings: dict[str, dict[str, Any] | None] = {}
    invalid_slots: dict[str, str] = {}

    for slot_name, slot in slot_by_name.items():
        validated, reason = validate_slot_binding(
            row=row,
            plan=plan,
            query_targets=query_targets,
            slot=slot,
            binding=displayed_bindings.get(slot_name),
            focus_tokens=focus_tokens,
        )
        validated_bindings[slot_name] = validated
        if slot_name in required_slot_names and reason and reason != "missing":
            invalid_slots[slot_name] = reason

    canonical_display_slot_types = {
        "direct_value",
        "current_resolution",
        "new_state",
        "old_state",
        "numeric_value",
        "numeric_operand",
        "state_anchor",
        "entity",
    }
    for slot_name, slot in slot_by_name.items():
        validated = validated_bindings.get(slot_name)
        displayed = displayed_bindings.get(slot_name)
        if not isinstance(validated, dict) or not isinstance(displayed, dict):
            continue
        if str(getattr(slot, "slot_type", "") or "") not in canonical_display_slot_types:
            continue
        if str(validated.get("unit_id") or "").strip() != str(displayed.get("unit_id") or "").strip():
            continue
        merged = dict(validated)
        for key in ("text", "display_text", "source_span_text", "source_text"):
            if str(displayed.get(key) or "").strip():
                merged[key] = displayed.get(key)
        validated_bindings[slot_name] = merged

    for slot_name, slot in slot_by_name.items():
        if slot_name in invalid_slots or isinstance(validated_bindings.get(slot_name), dict):
            continue
        inferred_reason = infer_missing_slot_invalid_reason(
            row=row,
            plan=plan,
            query_targets=query_targets,
            slot=slot,
            compiled_units=compiled_units,
        )
        if slot_name in required_slot_names and inferred_reason:
            invalid_slots[slot_name] = inferred_reason

    promote_support_grounding(
        plan=plan,
        validated_bindings=validated_bindings,
        displayed_bindings=displayed_bindings,
        invalid_slots=invalid_slots,
    )
    apply_reference_time_fallback(
        row=row,
        plan=plan,
        validated_bindings=validated_bindings,
        invalid_slots=invalid_slots,
    )
    rescue_current_state_where_binding(
        row=row,
        plan=plan,
        displayed_bindings=displayed_bindings,
        validated_bindings=validated_bindings,
        invalid_slots=invalid_slots,
    )
    rescue_information_extraction_where_binding(
        row=row,
        plan=plan,
        displayed_bindings=displayed_bindings,
        validated_bindings=validated_bindings,
        invalid_slots=invalid_slots,
    )
    rescue_temporal_when_event_binding(
        row=row,
        plan=plan,
        compiled_units=compiled_units,
        displayed_bindings=displayed_bindings,
        validated_bindings=validated_bindings,
        invalid_slots=invalid_slots,
    )
    _rescue_temporal_interval_events_from_time_bindings(
        plan=plan,
        validated_bindings=validated_bindings,
        invalid_slots=invalid_slots,
    )
    _rescue_duration_bindings(
        row=row,
        displayed_bindings=displayed_bindings,
        validated_bindings=validated_bindings,
        invalid_slots=invalid_slots,
    )
    _rescue_numeric_value_from_direct_binding(
        plan=plan,
        validated_bindings=validated_bindings,
        displayed_bindings=displayed_bindings,
        invalid_slots=invalid_slots,
        binding_source_text=binding_source_text,
        parse_numeric_text=parse_numeric_text,
    )
    if aggregate_requires_two_operands(plan):
        left = validated_bindings.get("operand_1")
        right = validated_bindings.get("operand_2")
        if isinstance(left, dict) and isinstance(right, dict):
            same_unit = str(left.get("unit_id") or "") == str(right.get("unit_id") or "")
            same_text = str(left.get("display_text") or left.get("text") or "").strip() == str(
                right.get("display_text") or right.get("text") or ""
            ).strip()
            same_source = binding_source_text(left) == binding_source_text(right)
            source_text = binding_source_text(left)
            mention_count = len(extract_numeric_mentions(source_text))
            if (same_unit or same_source) and same_text and mention_count < 2:
                validated_bindings["operand_2"] = None
                invalid_slots["operand_2"] = "missing_counterpart"

    for left_name, right_name in (
        ("left_operand", "right_operand"),
        ("choice_a", "choice_b"),
        ("event_a", "event_b"),
        ("time_a", "time_b"),
    ):
        left = validated_bindings.get(left_name)
        right = validated_bindings.get(right_name)
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        left_text = str(left.get("display_text") or left.get("text") or "").strip()
        right_text = str(right.get("display_text") or right.get("text") or "").strip()
        same_unit = str(left.get("unit_id") or "").strip() == str(right.get("unit_id") or "").strip()
        same_source = binding_source_text(left) == binding_source_text(right)
        same_text = left_text and right_text and left_text == right_text
        if (same_unit or same_source) and same_text:
            validated_bindings[right_name] = None
            invalid_slots[right_name] = "missing_counterpart"

    if is_numeric_aggregate_schema(plan) or is_comparison_schema(plan) or plan.schema_name in {"OrderedChoice", "TemporalInterval"}:
        unit_to_entities: dict[str, set[str]] = {}
        for slot in plan.slots:
            if not slot.required:
                continue
            binding = validated_bindings.get(slot.slot_name)
            if not isinstance(binding, dict):
                continue
            unit_id = str(binding.get("unit_id") or "").strip()
            if not unit_id:
                continue
            entity_key = slot.entity_key or re.sub(r"_[ab12]$", "", slot.slot_name)
            unit_to_entities.setdefault(unit_id, set()).add(entity_key)

        for unit_id, entities in unit_to_entities.items():
            if len(entities) <= 1:
                continue
            shared_binding = next(
                binding
                for binding in validated_bindings.values()
                if isinstance(binding, dict) and str(binding.get("unit_id")) == unit_id
            )
            shared_source = binding_source_text(shared_binding)
            temporal_pattern = r"\d{4}/\d{2}/\d{2}|\b\d{1,2}:\d{2}\b|\b(?:today|yesterday|ago|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december)\b"
            has_multi_temporal = len(list(re.finditer(temporal_pattern, shared_source.lower()))) >= 2
            has_multi_numeric = count_numeric_spans_excluding_dates(shared_source) >= 2
            word_count = len(shared_source.split())
            has_ordering_connective = bool(
                re.search(
                    r"\b(before|after|then|later|earlier|next|subsequently|afterward|afterwards|prior(?:\s+to)?|followed by|first|second|third)\b",
                    shared_source.lower(),
                )
            )

            allow_shared = False
            if plan.schema_name == "OrderedChoice" and word_count >= 5 and (has_multi_temporal or has_ordering_connective):
                allow_shared = True
            elif plan.schema_name == "TemporalInterval" and has_multi_temporal:
                allow_shared = True
            elif (is_numeric_aggregate_schema(plan) or is_comparison_schema(plan)) and (
                has_multi_numeric or re.search(r"\b(more|less|higher|lower|than)\b", shared_source.lower())
            ):
                allow_shared = True

            if allow_shared:
                continue
            for slot in plan.slots:
                binding = validated_bindings.get(slot.slot_name)
                if slot.required and isinstance(binding, dict) and str(binding.get("unit_id")) == unit_id:
                    validated_bindings[slot.slot_name] = None
                    invalid_slots.setdefault(slot.slot_name, "missing_counterpart")

    required_slot_names = required_binding_names(plan)
    compiler_answerable = all(isinstance(validated_bindings.get(slot_name), dict) for slot_name in required_slot_names)
    supporting_unit_ids = list(
        dict.fromkeys(
            str(binding.get("unit_id"))
            for binding in validated_bindings.values()
            if isinstance(binding, dict) and str(binding.get("unit_id")).strip()
        )
    )

    grounding = schema_grounding_analysis(
        row=row,
        plan=plan,
        query_targets=query_targets,
        validated_bindings=validated_bindings,
        displayed_bindings=displayed_bindings,
        invalid_slots=invalid_slots,
    )
    answerability_level = str(grounding.get("answerability_level") or "abstain")
    deterministic_allowed = bool(grounding.get("deterministic_allowed"))
    reader_route_reason = str(grounding.get("reader_route_reason") or "")
    slot_grounding_scores = dict(grounding.get("slot_grounding_scores") or {})
    slot_grounding_fail_reasons = dict(grounding.get("slot_grounding_fail_reasons") or {})
    question_subtype = str(grounding.get("question_subtype") or "")
    flat_grounding_fail_reasons = {
        str(reason).strip().lower()
        for reasons in slot_grounding_fail_reasons.values()
        if isinstance(reasons, list)
        for reason in reasons
        if str(reason).strip()
    }

    answer_mode = "reader_from_slots"
    answer_text: str | None = None
    answer_units = list(supporting_unit_ids)
    has_any_validated_slot = any(isinstance(binding, dict) for binding in validated_bindings.values())
    has_any_raw_slot = any(isinstance(binding, dict) for binding in displayed_bindings.values())
    severe_invalid = any(reason not in soft_invalid_reasons for reason in invalid_slots.values())

    deterministic_count_answer, deterministic_count_units = execute_distinct_count_from_compiled_units(
        row=row,
        plan=plan,
        compiled_units=compiled_units,
    )
    aggregate_answer: str | None = None
    aggregate_units: list[str] = []
    direct_duration_answer: str | None = None
    direct_duration_units: list[str] = []
    relative_time_answer: str | None = None
    relative_time_units: list[str] = []
    compact_direct_answer: str | None = None
    compact_direct_units: list[str] = []

    if is_numeric_aggregate_schema(plan) or is_count_aggregate_schema(plan):
        aggregate_answer, aggregate_units = execute_percentage_aggregate_from_bindings(
            row=row,
            validated_bindings=validated_bindings,
        )
        if not aggregate_answer:
            aggregate_answer, aggregate_units = execute_aggregate_from_compiled_units(
                row=row,
                plan=plan,
                compiled_units=compiled_units,
            )
    elif plan.schema_name in {"TemporalInterval", "EventDuration"}:
        direct_duration_answer, direct_duration_units = execute_direct_duration_from_compiled_units(
            row=row,
            compiled_units=compiled_units,
        )
    elif plan.schema_name == "RelativeTime":
        relative_time_answer, relative_time_units = execute_relative_time(
            row=row,
            validated_bindings=validated_bindings,
            displayed_bindings=displayed_bindings,
        )

    if plan.family in {"information_extraction", "single_anchor", "current_state", "knowledge_update", "conflict_update"}:
        from .evaluation import _execute_compact_direct_answer_from_binding

        for slot_name in ("direct_value", "current_value", "new_value", "old_state", "current_resolution"):
            binding = validated_bindings.get(slot_name) or displayed_bindings.get(slot_name)
            compact_direct_answer, compact_direct_units = _execute_compact_direct_answer_from_binding(
                row=row,
                binding=binding,
            )
            if compact_direct_answer:
                break

    if answerability_level == "abstain" and not (has_any_validated_slot or has_any_raw_slot):
        if relative_time_answer:
            answer_text = relative_time_answer
            answer_mode = "deterministic_numeric"
            answer_units = relative_time_units
        else:
            answer_mode = "abstain"
    elif answerability_level == "deterministic_safe" and compiler_answerable and deterministic_allowed:
        # Yes/no questions should go to reader, not deterministic span.
        # Deterministic span extraction picks entity phrases, not "Yes"/"No".
        _q_lower = str(row.get("query") or row.get("question") or "").strip().lower()
        _is_yes_no_question = any(_q_lower.startswith(p) for p in (
            "is ", "do ", "does ", "did ", "was ", "were ", "has ", "have ",
            "can ", "are ", "will ", "could ", "should ", "would ",
        ))
        if _is_yes_no_question and plan.schema_name not in {
            "OrderedChoice", "CountLookup", "CountEvents", "CountDistinctItems",
            "SumOperands", "TemporalInterval", "RelativeTime", "EventDuration",
        }:
            answer_mode = "reader_from_slots"
        elif plan.schema_name == "OrderedChoice":
            ordered_answer, ordered_units = execute_ordered_choice(
                row=row,
                query_targets=query_targets,
                validated_bindings=validated_bindings,
            )
            if ordered_answer:
                answer_text = ordered_answer
                answer_mode = "deterministic_span"
                answer_units = ordered_units
        elif is_count_aggregate_schema(plan) and aggregate_answer and not severe_invalid:
            answer_text = aggregate_answer
            answer_mode = "deterministic_numeric"
            answer_units = aggregate_units
        elif _reader_first_schema(plan, compiled_units=compiled_units):
            answer_mode = "reader_from_slots"
        elif plan.schema_name in {"TemporalInterval", "RelativeTime", "EventDuration"}:
            # Temporal schemas bypass _reader_first_schema when question_date is set.
            # Try the direct-duration executor first; the result is emitted as a
            # deterministic_numeric answer so the reader is not called at all.
            if plan.schema_name in {"TemporalInterval", "EventDuration"} and direct_duration_answer:
                answer_text = direct_duration_answer
                answer_mode = "deterministic_numeric"
                answer_units = direct_duration_units
            elif plan.schema_name == "RelativeTime" and relative_time_answer:
                answer_text = relative_time_answer
                answer_mode = "deterministic_numeric"
                answer_units = relative_time_units
            else:
                # No clean numeric result — fall to reader but with the pre-computed hint.
                answer_mode = "reader_from_slots"
        elif deterministic_count_answer:
            answer_text = deterministic_count_answer
            answer_mode = "deterministic_numeric"
            answer_units = deterministic_count_units
        elif plan.family in {"information_extraction", "single_anchor", "aggregation"}:
            primary = validated_bindings.get("numeric_value") if plan.requires_numeric_reasoning else None
            if not isinstance(primary, dict):
                primary = validated_bindings.get("direct_value") or validated_bindings.get("numeric_value")
            if isinstance(primary, dict):
                if primary is validated_bindings.get("numeric_value"):
                    if _deterministic_numeric_allowed(
                        plan=plan,
                        slot_name="numeric_value",
                        binding=primary,
                        parse_numeric_text=parse_numeric_text,
                    ):
                        answer_text = str(primary.get("display_text") or primary.get("text") or "").strip()
                        answer_mode = "deterministic_numeric"
                        answer_units = [str(primary.get("unit_id"))]
                elif deterministic_span_allowed(row=row, plan=plan, slot_name="direct_value", binding=primary):
                    answer_text = str(primary.get("display_text") or primary.get("text") or "").strip()
                    answer_mode = "deterministic_span"
                    answer_units = [str(primary.get("unit_id"))]
        elif plan.family in {"current_state", "knowledge_update", "conflict_update"}:
            numeric_primary = validated_bindings.get("numeric_value")
            if isinstance(numeric_primary, dict) and _deterministic_numeric_allowed(
                plan=plan,
                slot_name="numeric_value",
                binding=numeric_primary,
                parse_numeric_text=parse_numeric_text,
            ):
                answer_text = str(numeric_primary.get("display_text") or numeric_primary.get("text") or "").strip()
                answer_mode = "deterministic_numeric"
                answer_units = [str(numeric_primary.get("unit_id"))]
            primary = (
                validated_bindings.get("current_value")
                or validated_bindings.get("new_value")
                or validated_bindings.get("direct_value")
            )
            primary_slot = (
                "current_value"
                if isinstance(validated_bindings.get("current_value"), dict)
                else "new_value"
                if isinstance(validated_bindings.get("new_value"), dict)
                else "direct_value"
            )
            if answer_text is None and isinstance(primary, dict) and deterministic_span_allowed(
                row=row,
                plan=plan,
                slot_name=primary_slot,
                binding=primary,
            ):
                answer_text = str(primary.get("display_text") or primary.get("text") or "").strip()
                answer_mode = "deterministic_span"
                answer_units = [str(primary.get("unit_id"))]
        elif plan.family == "temporal" and plan.schema_name in {"DirectValue", "EventDuration"}:
            numeric_primary = validated_bindings.get("numeric_value")
            if isinstance(numeric_primary, dict) and _deterministic_numeric_allowed(
                plan=plan,
                slot_name="numeric_value",
                binding=numeric_primary,
                parse_numeric_text=parse_numeric_text,
            ):
                answer_text = str(numeric_primary.get("display_text") or numeric_primary.get("text") or "").strip()
                answer_mode = "deterministic_numeric"
                answer_units = [str(numeric_primary.get("unit_id"))]
            primary_slot = "direct_value" if isinstance(validated_bindings.get("direct_value"), dict) else "event"
            primary = validated_bindings.get(primary_slot)
            if answer_text is None and isinstance(primary, dict) and deterministic_span_allowed(
                row=row,
                plan=plan,
                slot_name=primary_slot,
                binding=primary,
            ):
                answer_text = str(primary.get("display_text") or primary.get("text") or "").strip()
                answer_mode = "deterministic_span"
                answer_units = [str(primary.get("unit_id"))]

        if compiler_answerable and is_numeric_aggregate_schema(plan) and not answer_text and not _reader_first_schema(plan):
            left = validated_bindings.get("operand_1")
            right = validated_bindings.get("operand_2")
            if isinstance(left, dict) and isinstance(right, dict):
                parsed_left = parse_numeric_text(str(left.get("display_text") or left.get("text") or ""))
                parsed_right = parse_numeric_text(str(right.get("display_text") or right.get("text") or ""))
                if parsed_left and parsed_right and parsed_left[1:] == parsed_right[1:]:
                    answer_text = format_numeric_answer(parsed_left[0] + parsed_right[0], parsed_left[1], parsed_left[2])
                    answer_mode = "deterministic_numeric"
                    answer_units = [str(left.get("unit_id")), str(right.get("unit_id"))]
    elif (
        is_count_aggregate_schema(plan)
        and aggregate_answer
        and not severe_invalid
        and not invalid_slots
    ):
        answer_text = aggregate_answer
        answer_mode = "deterministic_numeric"
        answer_units = aggregate_units
    else:
        if has_any_validated_slot:
            answer_mode = "reader_from_slots"
        elif severe_invalid:
            answer_mode = "abstain"
        elif has_any_raw_slot:
            answer_mode = "reader_from_slots"
        else:
            answer_mode = "reader_from_slots"

    compact_direct_reasons_ok = flat_grounding_fail_reasons <= {
        "deterministic_confidence_low",
        "generic_fragment",
    }
    if (
        not answer_text
        and compact_direct_answer
        and not severe_invalid
        and has_any_validated_slot
        and answerability_level in {"reader_only", "deterministic_safe"}
        and compact_direct_reasons_ok
    ):
        answer_text = compact_direct_answer
        answer_mode = "deterministic_span"
        answer_units = compact_direct_units

    # MiniLM span verification: for deterministic answers, verify that a
    # lightweight span extractor (22M param BERT, not an LLM) can also find
    # an answer in the source text.  This catches two failure modes:
    # (1) retrieval miss — the compiled evidence doesn't contain the answer,
    #     so MiniLM confidence is low → block deterministic
    # (2) extraction error — the compiler grabbed the wrong span, MiniLM
    #     found the right one → use MiniLM's span
    _MINILM_CONFIDENCE_GATE = 0.7
    _MINILM_OVERRIDE_THRESHOLD = 0.85
    if answer_mode in {"deterministic_span", "deterministic_numeric"} and answer_text:
        if req.is_count_aggregate_schema(plan) and answer_mode == "deterministic_numeric":
            answer_text = _canonicalize_count_answer_text(
                answer_text=answer_text,
                parse_numeric_text=parse_numeric_text,
            )
        skip_minilm_override = str(answer_text).strip().lower() in {"yes", "no"}
        try:
            from ..adapters.minilm_qa_adapter import extract_span

            question = str(row.get("query") or "").strip()
            # Get the source text from the primary binding
            primary_binding = None
            for slot_name in ("direct_value", "numeric_value", "current_value", "new_value", "event"):
                candidate = validated_bindings.get(slot_name)
                if isinstance(candidate, dict):
                    primary_binding = candidate
                    break
            source_text = binding_source_text(primary_binding) if primary_binding else ""
            if (
                answer_mode == "deterministic_span"
                and isinstance(primary_binding, dict)
                and str(primary_binding.get("display_text") or primary_binding.get("text") or "").strip() == str(answer_text).strip()
                and len(str(answer_text).split()) >= 2
            ):
                skip_minilm_override = True
            if question and source_text:
                span_result = extract_span(question, source_text)
                if span_result.confidence < _MINILM_CONFIDENCE_GATE:
                    # Low confidence → evidence probably doesn't contain the answer
                    answer_text = None
                    answer_mode = "reader_from_slots"
                    answer_units = []
                    slot_grounding_fail_reasons.setdefault("__route__", []).append("minilm_low_confidence")
                elif not skip_minilm_override and span_result.answer and span_result.confidence >= _MINILM_OVERRIDE_THRESHOLD:
                    # MiniLM found a confident span — check if it differs from compiler
                    compiler_normalized = re.sub(r"\s+", " ", answer_text.strip().lower())
                    minilm_normalized = re.sub(r"\s+", " ", span_result.answer.strip().lower())
                    if compiler_normalized != minilm_normalized and minilm_normalized:
                        # MiniLM disagrees with compiler — prefer MiniLM span
                        answer_text = span_result.answer
        except Exception:
            pass  # If MiniLM unavailable, fall through to original answer

    if answer_mode == "reader_from_slots" and answerability_level == "deterministic_safe":
        answerability_level = "reader_only"
        deterministic_allowed = False
        reader_route_reason = _append_route_reason(reader_route_reason, "deterministic_not_emitted")

    if answer_mode == "abstain" and has_any_validated_slot:
        answer_mode = "reader_from_slots"
        answerability_level = "reader_only"
        deterministic_allowed = False
        reader_route_reason = _append_route_reason(reader_route_reason, "reader_from_validated_slots")

    answer_contract = normalize_post_contract(
        {
            "validated_slot_bindings": validated_bindings,
            "invalid_slots": invalid_slots,
            "compiler_answerable": compiler_answerable,
            "answer_mode": answer_mode,
            "answer_text": answer_text,
            "supporting_unit_ids": answer_units,
            "displayed_slot_bindings": displayed_bindings,
            "answerability_level": answerability_level,
            "deterministic_allowed": deterministic_allowed,
            "reader_route_reason": reader_route_reason,
            "slot_grounding_scores": slot_grounding_scores,
            "slot_grounding_fail_reasons": slot_grounding_fail_reasons,
            "schema_name": plan.schema_name,
            "question_subtype": question_subtype,
        }
    )
    return {
        "validated_slot_bindings": validated_bindings,
        "invalid_slots": invalid_slots,
        "compiler_answerable": compiler_answerable,
        "answer_mode": answer_contract["answer_mode"],
        "answer_text": answer_contract["answer_text"],
        "supporting_unit_ids": answer_contract["supporting_unit_ids"],
        "displayed_slot_bindings": displayed_bindings,
        "answerability_level": answer_contract["answerability_level"],
        "deterministic_allowed": answer_contract["deterministic_allowed"],
        "reader_route_reason": answer_contract["reader_route_reason"],
        "slot_grounding_scores": slot_grounding_scores,
        "slot_grounding_fail_reasons": slot_grounding_fail_reasons,
        "schema_name": plan.schema_name,
        "question_subtype": question_subtype,
    }


def validated_slot_contract(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: Any,
    compiled_units: list[dict[str, Any]],
    raw_slot_bindings: dict[str, dict[str, Any] | None],
    slot_bindings_with_display_text: Callable[..., dict[str, dict[str, Any] | None]],
    query_focus_tokens: Callable[[dict[str, Any]], set[str]],
    validate_slot_binding: Callable[..., tuple[dict[str, Any] | None, str | None]],
    infer_missing_slot_invalid_reason: Callable[..., str | None],
    promote_support_grounding: Callable[..., None],
    apply_reference_time_fallback: Callable[..., None],
    rescue_current_state_where_binding: Callable[..., None],
    rescue_information_extraction_where_binding: Callable[..., None],
    rescue_temporal_when_event_binding: Callable[..., None],
    binding_source_text: Callable[[dict[str, Any]], str],
    extract_numeric_mentions: Callable[[str], list[tuple[float, str, str, str]]],
    count_numeric_spans_excluding_dates: Callable[[str], int],
    schema_grounding_analysis: Callable[..., dict[str, Any]],
    execute_distinct_count_from_compiled_units: Callable[..., tuple[str | None, list[str]]],
    execute_percentage_aggregate_from_bindings: Callable[..., tuple[str | None, list[str]]],
    execute_aggregate_from_compiled_units: Callable[..., tuple[str | None, list[str]]],
    execute_direct_duration_from_compiled_units: Callable[..., tuple[str | None, list[str]]],
    execute_relative_time: Callable[..., tuple[str | None, list[str]]],
    execute_ordered_choice: Callable[..., tuple[str | None, list[str]]],
    execute_comparison: Callable[..., tuple[str | None, list[str]]],
    execute_temporal_interval: Callable[..., tuple[str | None, list[str]]],
    deterministic_span_allowed: Callable[..., bool],
    parse_numeric_text: Callable[[str], tuple[float, str, str] | None],
    format_numeric_answer: Callable[[float, str, str], str],
    soft_invalid_reasons: set[str],
) -> dict[str, Any]:
    return build_answer_contract(
        row=row,
        plan=plan,
        query_targets=query_targets,
        compiled_units=compiled_units,
        raw_slot_bindings=raw_slot_bindings,
        slot_bindings_with_display_text=slot_bindings_with_display_text,
        query_focus_tokens=query_focus_tokens,
        validate_slot_binding=validate_slot_binding,
        infer_missing_slot_invalid_reason=infer_missing_slot_invalid_reason,
        promote_support_grounding=promote_support_grounding,
        apply_reference_time_fallback=apply_reference_time_fallback,
        rescue_current_state_where_binding=rescue_current_state_where_binding,
        rescue_information_extraction_where_binding=rescue_information_extraction_where_binding,
        rescue_temporal_when_event_binding=rescue_temporal_when_event_binding,
        binding_source_text=binding_source_text,
        extract_numeric_mentions=extract_numeric_mentions,
        count_numeric_spans_excluding_dates=count_numeric_spans_excluding_dates,
        schema_grounding_analysis=schema_grounding_analysis,
        execute_distinct_count_from_compiled_units=execute_distinct_count_from_compiled_units,
        execute_percentage_aggregate_from_bindings=execute_percentage_aggregate_from_bindings,
        execute_aggregate_from_compiled_units=execute_aggregate_from_compiled_units,
        execute_direct_duration_from_compiled_units=execute_direct_duration_from_compiled_units,
        execute_relative_time=execute_relative_time,
        execute_ordered_choice=execute_ordered_choice,
        execute_comparison=execute_comparison,
        execute_temporal_interval=execute_temporal_interval,
        deterministic_span_allowed=deterministic_span_allowed,
        parse_numeric_text=parse_numeric_text,
        format_numeric_answer=format_numeric_answer,
        soft_invalid_reasons=soft_invalid_reasons,
    )
