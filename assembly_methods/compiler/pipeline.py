"""Top-level SIEVE compiler pipeline orchestration."""

from __future__ import annotations

from typing import Any

from ..common import CandidateView, shared_result
from ..evidence_compiler import compile_evidence
from ..evidence_planner import plan_evidence
from ..evidence_renderer import render_evidence_package, render_multi_session_fact_list
from ..proposal_retrieval import retrieve_proposal
from ..query_targets import extract_query_targets
from .intent.answer_intent import infer_query_family, parse_answer_intent
from .intent.query_features import (
    has_event_duration_anchor,
    has_explicit_temporal_reference,
    has_latest_resolution_marker,
    interrogative_for_query,
    is_duration_query,
    is_multi_session_query,
    is_temporally_scoped_numeric_aggregate_query,
    normalized_query,
)
from .types import BoundEvidencePackage, CandidateBinding, InvariantFailure


import re as _re


_FILL_HAS_NUMBER_RE = _re.compile(r"\d")
_FILL_GENERIC_ASSISTANT_RE = _re.compile(
    r"\b(?:can you|could you|do you have|here are|recommend|suggest|tips?|consider|for example|try to|make sure|remember to)\b",
    _re.IGNORECASE,
)


def _score_fill_unit(unit, target_entities, subject_entities, preferred_attrs, schema):
    """Score an evidence unit for schema-conditioned budget fill.

    Assistant-spoken lines are gated on factual content: they must carry a
    number/date AND not be generic advice to score above zero.  This mirrors
    the aggregation filter in evidence_renderer.py and avoids spending token
    budget on assistant noise (yoga tips, general recommendations) that
    entity overlap alone would let through.
    """
    # --- Assistant factual gate (applied before entity scoring) ---
    # Assistant lines that are generic advice score 0 regardless of entity
    # overlap.  Only assistant lines carrying concrete facts (numbers, dates)
    # survive into the scoring pipeline.
    if str(unit.speaker or "").lower() == "assistant":
        text = unit.render_text or ""
        has_number = bool(_FILL_HAS_NUMBER_RE.search(text))
        is_generic = bool(_FILL_GENERIC_ASSISTANT_RE.search(text))
        if not has_number or is_generic:
            return 0.0

    s = 0.0
    text_lower = unit.render_text.lower()

    for te in target_entities:
        if te in text_lower:
            s += 2.0
    for se in subject_entities:
        if se in text_lower:
            s += 2.0
    for attr in preferred_attrs:
        if attr in text_lower:
            s += 2.0

    if "Count" in schema or "Sum" in schema:
        if unit.is_countable_item or unit.is_numeric_operand:
            s += 3.0
        if unit.has_acquisition_marker or unit.has_progress_marker:
            s += 2.0
        if unit.speaker == "user":
            s += 2.0
    elif "Temporal" in schema or "RelativeTime" in schema:
        if unit.is_temporal_anchor or unit.time_markers:
            s += 3.0
        if unit.date_key and unit.date_key != (0, 0, 0):
            s += 1.0
        if unit.speaker == "user":
            s += 1.0
    elif "CurrentState" in schema:
        if unit.is_update_assertion or unit.is_current_state_candidate:
            s += 3.0
        if unit.has_current_state_language:
            s += 2.0
    else:
        if unit.is_attribute_value or unit.is_direct_answer_candidate:
            s += 3.0
        if unit.speaker == "user":
            s += 1.0

    if unit.is_generic_state_text or unit.is_low_authority_text:
        s -= 3.0
    if unit.is_question_or_request:
        s -= 2.0
    if unit.is_recommendation_or_advice:
        s -= 2.0
    if unit.is_instructional_quantity:
        s -= 1.0
    return s


def _dedupe_texts(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for text in texts:
        normalized = str(text).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _binding_from_payload(
    *,
    slot_name: str,
    payload: dict[str, Any],
    invalid_reason: str | None = None,
) -> CandidateBinding:
    date_key = payload.get("date_key")
    normalized_date_key = tuple(date_key) if isinstance(date_key, (list, tuple)) and len(date_key) == 3 else None
    return CandidateBinding(
        slot_name=slot_name,
        slot_type=str(payload.get("slot_type") or payload.get("role") or ""),
        answer_type=str(payload.get("answer_type") or payload.get("expected_unit") or "text"),
        memory_id=str(payload.get("memory_id") or ""),
        object_id=(str(payload.get("object_id")) if payload.get("object_id") is not None else None),
        text=str(payload.get("text") or ""),
        source_text=str(payload.get("source_text") or payload.get("text") or ""),
        date_key=normalized_date_key,
        value_number=(float(payload["value_number"]) if payload.get("value_number") is not None else None),
        value_unit=(str(payload.get("value_unit")) if payload.get("value_unit") is not None else None),
        entity_match=float(payload.get("entity_match") or 0.0),
        attribute_match=float(payload.get("attribute_match") or 0.0),
        unit_match=float(payload.get("unit_match") or 0.0),
        focus_match=float(payload.get("focus_match") or 0.0),
        confidence=float(payload.get("confidence") or 0.0),
        support_ids=tuple(str(item) for item in (payload.get("support_ids") or ())),
        rejection_reasons=tuple(
            str(item)
            for item in ((payload.get("rejection_reasons") or ()) if invalid_reason is None else [invalid_reason, *(payload.get("rejection_reasons") or ())])
        ),
    )


def _build_bound_evidence_package(
    *,
    intent,
    plan,
    compiler_output: dict[str, Any],
) -> BoundEvidencePackage:
    validated_slot_bindings = dict(compiler_output.get("validated_slot_bindings") or {})
    invalid_slots = {
        str(name): str(reason)
        for name, reason in dict(compiler_output.get("invalid_slots") or {}).items()
    }
    validated_bindings = {
        slot_name: (_binding_from_payload(slot_name=slot_name, payload=payload) if isinstance(payload, dict) else None)
        for slot_name, payload in validated_slot_bindings.items()
    }
    missing_failures = tuple(
        InvariantFailure(code=reason, slot_name=slot_name, severity="hard", detail=reason.replace("_", " "))
        for slot_name, reason in invalid_slots.items()
    )
    return BoundEvidencePackage(
        intent=intent,
        plan=plan,
        validated_bindings=validated_bindings,
        invalid_bindings=invalid_slots,
        missing_invariants=missing_failures,
        compiled_memory_ids=tuple(str(item) for item in (compiler_output.get("compiled_memory_ids") or ())),
        compiled_unit_ids=tuple(
            str(item)
            for item in (
                compiler_output.get("supporting_unit_ids")
                or [unit.get("unit_id") for unit in compiler_output.get("compiled_units", []) if isinstance(unit, dict) and unit.get("unit_id")]
            )
        ),
        token_count=int(compiler_output.get("compiled_token_count") or 0),
    )


def _fallback_memory_usefulness_labels(row: dict[str, Any], context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = context.get("memory_usefulness_labels")
    if isinstance(payload, dict):
        return payload
    payload = row.get("memory_usefulness_labels")
    if isinstance(payload, dict):
        return payload
    return {}


def _resolve_proxy_runtime_labels(
    *,
    row: dict[str, Any],
    context: dict[str, Any],
    use_proxy_runtime_labels: bool,
) -> dict[str, dict[str, Any]]:
    if not use_proxy_runtime_labels:
        return {}
    try:
        from ..modeling import resolve_memory_usefulness_labels

        return resolve_memory_usefulness_labels(row, context)
    except Exception:
        return _fallback_memory_usefulness_labels(row, context)


def _candidate_query_families(row: dict[str, Any], inferred_family: str) -> list[str]:
    normalized = normalized_query(row)
    padded = f" {normalized} "
    interrogative = interrogative_for_query(normalized)
    families: list[str] = [str(inferred_family or "").strip() or "information_extraction"]

    if has_latest_resolution_marker(normalized) and interrogative in {"who", "what", "which", "where"}:
        families.extend(["current_state", "information_extraction", "temporal"])
    elif has_explicit_temporal_reference(normalized) and interrogative in {"who", "what", "which", "where"}:
        families.extend(["temporal", "information_extraction"])

    if is_duration_query(normalized) and (" when i " in padded or has_event_duration_anchor(normalized)):
        families.extend(["temporal", "information_extraction"])

    if is_temporally_scoped_numeric_aggregate_query(normalized):
        families.extend(["aggregation", "information_extraction", "temporal"])

    deduped: list[str] = []
    seen: set[str] = set()
    for family in families:
        normalized_family = str(family or "").strip()
        if not normalized_family or normalized_family in seen:
            continue
        seen.add(normalized_family)
        deduped.append(normalized_family)
    return deduped


def _compiler_output_selection_score(compiler_output: dict[str, Any]) -> float:
    answerability = str(compiler_output.get("answerability_level") or "").strip()
    route_reason = str(compiler_output.get("reader_route_reason") or "").strip()
    invalid_slots = dict(compiler_output.get("invalid_slots") or {})
    semantic_score = float(compiler_output.get("semantic_sufficiency_score") or 0.0)
    slot_completion = float(compiler_output.get("slot_completion_ratio") or 0.0)

    score = {
        "deterministic_safe": 4.0,
        "reader_only": 2.5,
        "abstain": 0.0,
    }.get(answerability, 0.0)
    score += 1.0 if bool(compiler_output.get("sufficient")) else 0.0
    score += 0.75 if bool(compiler_output.get("compiler_answerable")) else 0.0
    score += 1.25 * semantic_score
    score += 0.75 * slot_completion

    for reason in invalid_slots.values():
        normalized_reason = str(reason or "").strip()
        if normalized_reason in {
            "wrong_entity",
            "missing_counterpart",
            "same_source_operand",
            "event_entity_mismatch",
            "comparison_entity_mismatch",
            "entity_lookup_no_match",
        }:
            score -= 0.35
        elif normalized_reason in {"generic_fragment", "deterministic_confidence_low", "reader_first_schema"}:
            score -= 0.15
        else:
            score -= 0.25

    if route_reason == "no_usable_bindings":
        score -= 1.0
    elif route_reason == "hard_validation_missing":
        score -= 1.25
    elif route_reason == "schema_grounding_partial":
        score -= 0.1

    return round(score, 6)


def _select_compiler_plan(
    *,
    row: dict[str, Any],
    context: dict[str, Any],
    inferred_family: str,
    ranked_views: list[CandidateView],
    selected_views: list[CandidateView],
    target_budget: int,
    memory_labels_by_id: dict[str, dict[str, Any]],
    prebuilt_objects: list | None = None,
) -> tuple[str, str, dict[str, Any], list[dict[str, Any]]]:
    candidate_families = _candidate_query_families(row, inferred_family)
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    lattice_meta: list[dict[str, Any]] = []

    for family in candidate_families:
        compiler_output = compile_evidence(
            row=row,
            context=context,
            query_family=family,
            ranked_views=ranked_views,
            selected_views=selected_views,
            target_budget=target_budget,
            memory_labels_by_id=memory_labels_by_id,
            prebuilt_objects=prebuilt_objects,
        )
        score = _compiler_output_selection_score(compiler_output)
        candidates.append((score, family, compiler_output))
        lattice_meta.append(
            {
                "query_family": family,
                "score": score,
                "answerability_level": compiler_output.get("answerability_level"),
                "schema_name": compiler_output.get("schema_name"),
                "reader_route_reason": compiler_output.get("reader_route_reason"),
                "compiler_sufficient": bool(compiler_output.get("sufficient")),
                "compiler_answerable": bool(compiler_output.get("compiler_answerable")),
                "semantic_sufficiency_score": compiler_output.get("semantic_sufficiency_score"),
                "slot_completion_ratio": compiler_output.get("slot_completion_ratio"),
                "invalid_slot_count": len(dict(compiler_output.get("invalid_slots") or {})),
            }
        )

    base_score, base_output = next(
        (score, output)
        for score, family, output in candidates
        if family == inferred_family
    )
    best_score, best_family, best_output = max(candidates, key=lambda item: (item[0], item[1]))

    if best_family != inferred_family and best_score > (base_score + 0.35):
        return best_family, "compiler_plan_lattice", best_output, lattice_meta
    return inferred_family, "compiler_intent_query_only", base_output, lattice_meta


def _run_multi_session_pipeline(
    *,
    row: dict[str, Any],
    context: dict[str, Any],
    ranked_views: list[CandidateView],
    selected_views: list[CandidateView],
    selected_ids: list[str],
    proposal: dict[str, Any],
    prebuilt_objects: list | None = None,
    _pc_before: Any = None,
    _row_start: float = 0.0,
) -> dict[str, Any]:
    """Multi-session compilation: extract facts from ALL memories, aggregate programmatically."""
    import time as _time
    from ..perf_counters import get_counters as _get_pc
    from ..evidence_units import build_evidence_units_for_views

    target_budget = int(proposal.get("target_token_budget") or 600)
    query_targets = extract_query_targets(row, "multi_session")

    # --- Memory-level pruning with ONNX MiniLM QA model ---
    # Score each memory by QA confidence (does this memory likely contain
    # an answer to the query?).  Reuses the existing ONNX adapter (~12ms
    # per call) instead of loading a separate sentence-transformers model.
    query_text = str(row.get("query") or row.get("question") or "").strip()
    pruned_views = ranked_views
    try:
        from .adapters.minilm_qa_adapter import extract_span
        _scored = []
        for v in ranked_views:
            result = extract_span(query_text, v.text[:500])
            _scored.append((result.confidence, v))
        _scored.sort(key=lambda x: -x[0])
        _top_k = int(context.get("multi_session_top_k", 20))
        pruned_views = [v for _, v in _scored[:_top_k]]
    except Exception:
        pass  # Fall back to all views if unavailable

    # Build evidence units from pruned views (the key difference: go wide but focused)
    all_units = build_evidence_units_for_views(pruned_views, prebuilt_objects)

    # Render the fact list
    rendered_evidence_package = render_multi_session_fact_list(
        row=row,
        all_units=all_units,
        query_targets=query_targets,
        target_budget=target_budget,
    )

    ms_meta = rendered_evidence_package.get("multi_session_meta") or {}
    compiled_memory_ids = ms_meta.get("memory_ids") or []
    final_views = [
        v for v in ranked_views
        if v.memory_id in set(compiled_memory_ids)
    ]

    selection_meta = {
        "query_family": "multi_session",
        "query_family_source": "multi_session_detector",
        "legacy_selector_query_family": None,
        "compiler_inferred_query_family": "multi_session",
        "compiler_plan_lattice_candidates": [],
        "query_target_entities": list(getattr(query_targets, "candidate_entities", ())),
        "query_target_subject_entities": list(getattr(query_targets, "subject_entities", ())) if hasattr(query_targets, "subject_entities") else [],
        "query_target_alternatives": list(getattr(query_targets, "alternative_entities", ())),
        "query_target_attributes": list(getattr(query_targets, "preferred_attributes", ())),
        "answer_intent": {},
        "schema_name": "MultiSessionFactList",
        "planner_schema_name": "MultiSessionFactList",
        "question_subtype": "multi_session_fact_list",
        "planner_slots": [],
        "selection_mode": "multi_session_fact_list",
        "compiler_authoritative": True,
        "proposal_mode": "proposal_retrieval",
        "proposal_score_model": None,
        "compiler_score_model": None,
        "use_proxy_runtime_labels": False,
        "compiler_seed_memory_ids": selected_ids,
        "proposal_selected_memory_ids": selected_ids,
        "compiled_memory_ids": compiled_memory_ids,
        "compiled_memory_count": len(compiled_memory_ids),
        "compiled_unit_count": ms_meta.get("total_facts", 0),
        "compiled_token_count": ms_meta.get("token_total", 0),
        "proposal_budget_window": proposal.get("proposal_budget_window"),
        "proposal_budget_overshoot_reasons": [],
        "effective_target_budget": target_budget,
        "budget_hard_cap": None,
        "budget_window": None,
        "budget_overshoot_reasons": [],
        "compiled_texts": [line["text"] for line in rendered_evidence_package.get("evidence_lines", [])],
        "requirements": {},
        "coverage": rendered_evidence_package.get("coverage", {}),
        "requirement_coverage": rendered_evidence_package.get("coverage", {}),
        "slot_bindings": rendered_evidence_package.get("slots", {}),
        "raw_slot_bindings": {},
        "compiler_sufficient": False,
        "answer_mode": "reader_from_slots",
        "compiler_answerable": False,
        "validated_slot_bindings": rendered_evidence_package.get("slots", {}),
        "invalid_slots": {},
        "answer_text": None,
        "supporting_unit_ids": rendered_evidence_package.get("answer_units", []),
        "answerability_level": "reader_only",
        "deterministic_allowed": False,
        "reader_route_reason": "multi_session_reader",
        "reader_prompt_variant": "crisp_compact_v1",
        "slot_grounding_scores": {},
        "slot_grounding_fail_reasons": {},
        "semantic_support_scores": {},
        "semantic_support_mean": None,
        "slot_grounding_mean": None,
        "semantic_sufficiency_score": 0.5,
        "semantic_sufficiency_passed": True,
        "semantic_sufficiency_applied": True,
        "semantic_sufficiency_reasons": [],
        "compiler_expansion_steps": [],
        "compiler_used_fallback": False,
        "compiler_rescue_used": False,
        "compiler_semantic_downgraded": False,
        "compiler_compiled_texts": [],
        "compiler_missing_slots": {},
        "compiler_projected_candidates": [],
        "compiler_slot_completion_ratio": 0.0,
        "compiler_token_budget_ratio": ms_meta.get("token_total", 0) / max(target_budget, 1),
        "compiler_minimality_ratio": None,
        "budget_throttle_reason": None,
        "budget_throttle_ratio": None,
        "rendered_evidence_package": rendered_evidence_package,
        "bound_evidence_package": {},
        "proposal_debug": dict(proposal.get("selection_debug") or {}),
        "proposal_pool_size": proposal.get("selection_debug", {}).get("proposal_pool_size"),
        "proposal_rank_cutoff": proposal.get("selection_debug", {}).get("proposal_rank_cutoff"),
        "candidate_pool_augmented": proposal.get("selection_debug", {}).get("candidate_pool_augmented"),
        "answer_memory_ids": list(proposal.get("selection_debug", {}).get("answer_memory_ids") or []),
        "answer_memory_in_proposal_pool": bool(proposal.get("selection_debug", {}).get("answer_memory_in_proposal_pool")),
        "answer_memory_selected": bool(proposal.get("selection_debug", {}).get("answer_memory_selected")),
        "answer_memory_compiled": bool(
            set(proposal.get("selection_debug", {}).get("answer_memory_ids") or [])
            & set(compiled_memory_ids)
        ),
        "multi_session_meta": ms_meta,
        "_perf": {
            **_get_pc().delta_since(_pc_before),
            "row_wall_ms": round((_time.perf_counter() - _row_start) * 1000, 1),
            "candidate_pool_size": len(row.get("candidate_memories") or []),
            "selected_memory_count": len(selected_ids),
        },
    }

    return shared_result(
        final_views,
        budget=target_budget,
        budget_mode="multi_session_fact_list",
        selection_meta=selection_meta,
    )


def run_sieve_pipeline(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Run the compiler-first SIEVE pipeline and return a shared method result."""
    import time as _time
    from ..perf_counters import get_counters as _get_pc
    _pc_before = _get_pc().snapshot()
    _row_start = _time.perf_counter()

    compiler_query_family = infer_query_family(row)
    legacy_query_family = None
    proposal = retrieve_proposal(row=row, context=context, evidence_plan=plan_evidence(row, compiler_query_family))
    ranked_views: list[CandidateView] = list(proposal["ranked_views"])
    selected_views: list[CandidateView] = list(proposal["selected_views"])
    selected_ids = list(proposal["selected_memory_ids"])
    views_by_id = {view.memory_id: view for view in ranked_views}

    use_proxy_runtime_labels = bool(context.get("use_proxy_runtime_labels", False))
    memory_labels_by_id = _resolve_proxy_runtime_labels(
        row=row,
        context=context,
        use_proxy_runtime_labels=use_proxy_runtime_labels,
    )

    # Reuse the memory objects already built and cached on the row by retrieve_proposal.
    # This avoids a redundant re-parse of selected memories in compile_evidence.
    from ..proposal_retrieval import _cache_key as _proposal_cache_key
    _object_store = row.get(_proposal_cache_key(str(context.get("sieve_object_store_version") or "v2")))
    _prebuilt_objects = list(_object_store["objects"]) if isinstance(_object_store, dict) else None

    # --- Multi-session fact-list compilation branch ---
    # For queries needing evidence from multiple sessions (counting, aggregation,
    # cross-session comparison), bypass the single-schema compiler and build a
    # wide fact list from ALL candidate memories.
    if is_multi_session_query(row) and not context.get("disable_multi_session_factlist", False):
        return _run_multi_session_pipeline(
            row=row,
            context=context,
            ranked_views=ranked_views,
            selected_views=selected_views,
            selected_ids=selected_ids,
            proposal=proposal,
            prebuilt_objects=_prebuilt_objects,
            _pc_before=_pc_before,
            _row_start=_row_start,
        )

    query_family, query_family_source, compiler_output, plan_lattice_candidates = _select_compiler_plan(
        row=row,
        context=context,
        inferred_family=compiler_query_family,
        ranked_views=ranked_views,
        selected_views=selected_views,
        target_budget=int(proposal.get("target_token_budget") or 0),
        memory_labels_by_id=memory_labels_by_id,
        prebuilt_objects=_prebuilt_objects,
    )
    answer_intent = parse_answer_intent(row, query_family)
    evidence_plan = plan_evidence(row, query_family)
    # Inject question_date into plan_metadata for temporal date-diff computation.
    _q_date = str(row.get("question_date") or "").strip()
    if _q_date and isinstance(evidence_plan.plan_metadata, dict):
        evidence_plan.plan_metadata["question_date"] = _q_date
    query_targets = extract_query_targets(row, query_family)

    final_selected_views = [
        views_by_id[memory_id]
        for memory_id in compiler_output["compiled_memory_ids"]
        if memory_id in views_by_id
    ]
    final_texts = _dedupe_texts(list(compiler_output["compiled_texts"]))

    # --- Schema-conditioned budget fill ---
    all_compiled_units = list(compiler_output["compiled_units"])
    compiled_token_total = sum(
        int(u.get("token_count") or 0) for u in all_compiled_units if isinstance(u, dict)
    )
    budget_target = int(proposal.get("target_token_budget") or 600)
    budget_remaining = max(0, budget_target - compiled_token_total)
    budget_fill_units: list[dict[str, Any]] = []
    enable_budget_fill = not bool(context.get("disable_budget_fill", False))
    if enable_budget_fill and budget_remaining > 50:
        from ..evidence_units import build_evidence_units_for_views
        compiled_ids = set(compiler_output["compiled_memory_ids"])
        uncompiled_views = [v for v in ranked_views if v.memory_id not in compiled_ids]
        if uncompiled_views:
            pool_units = build_evidence_units_for_views(uncompiled_views)
            target_entities = set(
                str(e).lower() for e in (query_targets.candidate_entities if query_targets else ())
            )
            subject_entities = set(
                str(e).lower() for e in (query_targets.subject_entities if query_targets else ())
            )
            preferred_attrs = set(
                str(a).lower() for a in (query_targets.preferred_attributes if query_targets else ())
            )
            schema = str(evidence_plan.schema_name or "")
            scored = [
                (_score_fill_unit(u, target_entities, subject_entities, preferred_attrs, schema), u)
                for u in pool_units
            ]
            scored.sort(key=lambda x: -x[0])
            fill_remaining = budget_remaining
            for score, unit in scored:
                if fill_remaining <= 0:
                    break
                if score <= 0:
                    continue
                unit_dict = {
                    "unit_id": unit.unit_id,
                    "memory_id": unit.memory_id,
                    "speaker": unit.speaker,
                    "render_text": unit.render_text,
                    "token_count": unit.token_count,
                    "provenance": dict(unit.provenance) if unit.provenance else {},
                    "parent_rank": unit.parent_rank,
                    "budget_fill": True,
                    "fill_score": round(score, 2),
                    "date_key": list(unit.date_key) if unit.date_key and unit.date_key != (0, 0, 0) else None,
                }
                budget_fill_units.append(unit_dict)
                fill_remaining -= unit.token_count

    rendered_evidence_package = render_evidence_package(
        query_family=query_family,
        evidence_plan=evidence_plan,
        query_targets=query_targets,
        coverage=dict(compiler_output["coverage"]),
        compiled_units=all_compiled_units,
        slot_bindings=dict(compiler_output.get("validated_slot_bindings") or {}),
        answer_mode=str(compiler_output.get("answer_mode") or ""),
        compiler_answerable=bool(compiler_output.get("compiler_answerable")),
        invalid_slots=dict(compiler_output.get("invalid_slots") or {}),
        answer_text=compiler_output.get("answer_text"),
        answer_units=list(compiler_output.get("supporting_unit_ids") or []),
        budget_fill_units=budget_fill_units,
        mode="structured_json",
    )
    bound_evidence_package = _build_bound_evidence_package(
        intent=answer_intent,
        plan=evidence_plan,
        compiler_output=compiler_output,
    )

    selection_meta = {
        "query_family": query_family,
        "query_family_source": query_family_source,
        "legacy_selector_query_family": legacy_query_family,
        "compiler_inferred_query_family": compiler_query_family,
        "compiler_plan_lattice_candidates": list(plan_lattice_candidates),
        "query_target_entities": list(query_targets.candidate_entities),
        "query_target_subject_entities": list(query_targets.subject_entities),
        "query_target_alternatives": list(query_targets.alternative_entities),
        "query_target_attributes": list(query_targets.preferred_attributes),
        "answer_intent": answer_intent.to_dict(),
        "schema_name": evidence_plan.schema_name,
        "planner_schema_name": evidence_plan.schema_name,
        "question_subtype": compiler_output.get("question_subtype"),
        "planner_slots": [
            {
                "slot_name": slot.slot_name,
                "slot_type": slot.slot_type,
                "entity_key": slot.entity_key,
                "attribute_key": slot.attribute_key,
                "expected_unit": slot.expected_unit,
                "required": slot.required,
            }
            for slot in evidence_plan.slots
        ],
        "selection_mode": "evidence_compiler",
        "compiler_authoritative": bool(
            compiler_output["sufficient"]
            or compiler_output.get("compiled_units")
            or compiler_output.get("validated_slot_bindings")
        ),
        "proposal_mode": "proposal_retrieval",
        "proposal_score_model": (
            str(context.get("sieve_proposal_model", {}).get("model_name") or "")
            if isinstance(context.get("sieve_proposal_model"), dict)
            else None
        ),
        "compiler_score_model": (
            str(context.get("sieve_compiler_model", {}).get("model_name") or "")
            if isinstance(context.get("sieve_compiler_model"), dict)
            else None
        ),
        "use_proxy_runtime_labels": use_proxy_runtime_labels,
        "compiler_seed_memory_ids": selected_ids,
        "proposal_selected_memory_ids": selected_ids,
        "compiled_memory_ids": compiler_output["compiled_memory_ids"],
        "compiled_memory_count": len(compiler_output["compiled_memory_ids"]),
        "compiled_unit_count": len(compiler_output["compiled_units"]),
        "compiled_token_count": compiler_output["compiled_token_count"],
        "proposal_budget_window": proposal.get("proposal_budget_window"),
        "proposal_budget_overshoot_reasons": proposal.get("proposal_budget_overshoot_reasons") or [],
        "effective_target_budget": compiler_output.get("effective_target_budget"),
        "budget_hard_cap": compiler_output.get("budget_hard_cap"),
        "budget_window": compiler_output.get("budget_window"),
        "budget_overshoot_reasons": compiler_output.get("budget_overshoot_reasons") or [],
        "compiled_texts": final_texts,
        "requirements": dict(compiler_output["requirements"]),
        "coverage": dict(compiler_output["coverage"]),
        "requirement_coverage": dict(compiler_output["coverage"]),
        "slot_bindings": dict(compiler_output.get("slot_bindings") or {}),
        "raw_slot_bindings": dict(compiler_output.get("raw_slot_bindings") or {}),
        "compiler_sufficient": bool(compiler_output["sufficient"]),
        "answer_mode": str(compiler_output.get("answer_mode") or ""),
        "compiler_answerable": bool(compiler_output.get("compiler_answerable")),
        "validated_slot_bindings": dict(compiler_output.get("validated_slot_bindings") or {}),
        "invalid_slots": dict(compiler_output.get("invalid_slots") or {}),
        "answer_text": compiler_output.get("answer_text"),
        "supporting_unit_ids": list(compiler_output.get("supporting_unit_ids") or []),
        "answerability_level": compiler_output.get("answerability_level"),
        "deterministic_allowed": compiler_output.get("deterministic_allowed"),
        "reader_route_reason": compiler_output.get("reader_route_reason"),
        "reader_prompt_variant": compiler_output.get("reader_prompt_variant"),
        "slot_grounding_scores": dict(compiler_output.get("slot_grounding_scores") or {}),
        "slot_grounding_fail_reasons": dict(compiler_output.get("slot_grounding_fail_reasons") or {}),
        "semantic_support_scores": dict(compiler_output.get("semantic_support_scores") or {}),
        "semantic_support_mean": compiler_output.get("semantic_support_mean"),
        "slot_grounding_mean": compiler_output.get("slot_grounding_mean"),
        "semantic_sufficiency_score": compiler_output.get("semantic_sufficiency_score"),
        "semantic_sufficiency_passed": compiler_output.get("semantic_sufficiency_passed"),
        "semantic_sufficiency_applied": compiler_output.get("semantic_sufficiency_applied"),
        "semantic_sufficiency_reasons": list(compiler_output.get("semantic_sufficiency_reasons") or []),
        "compiler_expansion_steps": list(compiler_output["expansion_steps"]),
        "compiler_used_fallback": bool(compiler_output["used_fallback"]),
        "compiler_rescue_used": bool(compiler_output.get("rescue_used")),
        "compiler_semantic_downgraded": bool(compiler_output.get("semantic_downgraded")),
        "compiler_compiled_texts": list(compiler_output["compiled_texts"]),
        "compiler_missing_slots": dict(compiler_output.get("missing_slots") or {}),
        "compiler_projected_candidates": list(compiler_output.get("projected_candidates") or []),
        "compiler_slot_completion_ratio": compiler_output.get("slot_completion_ratio"),
        "compiler_token_budget_ratio": compiler_output.get("token_budget_ratio"),
        "compiler_minimality_ratio": compiler_output.get("minimality_ratio"),
        "budget_throttle_reason": compiler_output.get("budget_throttle_reason"),
        "budget_throttle_ratio": compiler_output.get("budget_throttle_ratio"),
        "rendered_evidence_package": rendered_evidence_package,
        "bound_evidence_package": bound_evidence_package.to_dict(),
        "proposal_debug": dict(proposal.get("selection_debug") or {}),
        "proposal_pool_size": proposal.get("selection_debug", {}).get("proposal_pool_size"),
        "proposal_rank_cutoff": proposal.get("selection_debug", {}).get("proposal_rank_cutoff"),
        "candidate_pool_augmented": proposal.get("selection_debug", {}).get("candidate_pool_augmented"),
        "answer_memory_ids": list(proposal.get("selection_debug", {}).get("answer_memory_ids") or []),
        "answer_memory_in_proposal_pool": bool(proposal.get("selection_debug", {}).get("answer_memory_in_proposal_pool")),
        "answer_memory_selected": bool(proposal.get("selection_debug", {}).get("answer_memory_selected")),
        "answer_memory_compiled": bool(
            set(proposal.get("selection_debug", {}).get("answer_memory_ids") or [])
            & set(compiler_output.get("compiled_memory_ids") or [])
        ),
        # Per-row performance delta — counters incremented during this row only.
        "_perf": {
            **_get_pc().delta_since(_pc_before),
            "row_wall_ms": round((_time.perf_counter() - _row_start) * 1000, 1),
            "candidate_pool_size": len(row.get("candidate_memories") or []),
            "selected_memory_count": len(selected_ids),
        },
    }

    return shared_result(
        final_selected_views,
        budget=proposal.get("target_token_budget"),
        budget_mode="evidence_compiler",
        selection_meta=selection_meta,
    )
