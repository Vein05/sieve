"""High-recall proposal retrieval over candidate memories and memory objects."""

from __future__ import annotations

import re
from typing import Any

from shared.nlp import (
    CandidateView,
    cached_build_profile,
    candidate_views,
    check_stem_equivalence,
    get_clean_tokens,
    get_stems_for_text,
    stem_token,
    NOISE_TOKENS,
)
from evidence.schema import EvidencePlan
from compiler.execution.adaptive_budget import (
    append_with_budget_window,
    build_proposal_budget_window,
)
from compiler.execution.requirements import is_comparison_schema, is_direct_lookup_schema
from .memory_builder import build_memory_objects_for_views
from .memory_index import build_memory_object_index, score_indexed_memories
from evidence.memory_objects import MemoryObject
from .query_targets import extract_query_targets


_ATTRIBUTE_COMPATIBILITY: dict[str, set[str]] = {
    "name": {"using", "reading", "watching", "studying", "employer"},
    "brand": {"using"},
}
_MUSIC_SERVICE_QUERY_RE = re.compile(r"\bmusic\s+streaming\s+service\b", re.IGNORECASE)
_MUSIC_DOMAIN_TERMS = {"album", "artist", "artists", "audio", "listen", "listening", "music", "playlist", "playlists", "song", "songs", "spotify"}
_VIDEO_DOMAIN_TERMS = {"content", "movie", "movies", "netflix", "original", "series", "show", "shows", "watch", "watching"}
_PERSONAL_QUERY_RE = re.compile(r"\b(?:i|me|my|mine|we|our|ours)\b", re.IGNORECASE)
_FIRST_PERSON_MEMORY_RE = re.compile(r"\b(?:i|i'm|i've|i'd|me|my|mine|we|we're|we've|our|ours)\b", re.IGNORECASE)
_GENERIC_ASSISTANT_RE = re.compile(
    r"\b(?:can you|could you|do you have|for example|here are|recommend|suggest|tips?|resources?|consider)\b",
    re.IGNORECASE,
)


def _target_matches_text(target: str, text: str) -> bool:
    """Check if a target phrase appears in text via stem-set overlap and synonyms."""
    if not target or not text:
        return False

    target_tokens = get_clean_tokens(target)
    source_tokens = get_clean_tokens(text)

    if not target_tokens:
        return False

    matched = 0
    matched_tokens: list[str] = []
    for target_token in target_tokens:
        if any(check_stem_equivalence(target_token, source_token) for source_token in source_tokens):
            matched += 1
            matched_tokens.append(target_token)

    if len(target_tokens) > 1:
        head_token = target_tokens[-1]
        if head_token not in matched_tokens:
            return False
        return matched / len(target_tokens) >= 0.5
    return matched == len(target_tokens)


def _query_domain_adjustment(query_text: str, values: list[str | None]) -> float:
    normalized_query = str(query_text or "").lower()
    if not _MUSIC_SERVICE_QUERY_RE.search(normalized_query):
        return 0.0
    object_text = " ".join(str(value or "") for value in values).lower()
    if not object_text:
        return 0.0
    object_tokens = {token for token in get_clean_tokens(object_text) if token}
    if object_tokens & _MUSIC_DOMAIN_TERMS:
        return 2.5
    if object_tokens & _VIDEO_DOMAIN_TERMS:
        return -3.0
    return -1.0


def _is_personal_memory_query(query_text: str) -> bool:
    return bool(_PERSONAL_QUERY_RE.search(str(query_text or "")))


def _autobiographical_grounding_adjustment(
    *,
    obj: MemoryObject,
    query_text: str,
    direct_like_plan: bool,
    count_like_plan: bool,
    state_like_plan: bool,
) -> float:
    if not _is_personal_memory_query(query_text):
        return 0.0

    speaker = str(getattr(obj, "speaker", "") or "").strip().lower()
    source_text = str(getattr(obj, "source_text", "") or getattr(obj, "render_text", "") or getattr(obj, "value_text", "") or "")
    autobiographical_text = bool(_FIRST_PERSON_MEMORY_RE.search(source_text))
    generic_assistant_text = bool(_GENERIC_ASSISTANT_RE.search(source_text) or "?" in source_text)
    personal_lookup_like = direct_like_plan or count_like_plan or state_like_plan

    if speaker == "user":
        return 2.0 if personal_lookup_like else 1.0
    if speaker == "assistant":
        penalty = -1.5 if personal_lookup_like else -0.75
        if autobiographical_text:
            penalty += 0.5
        if generic_assistant_text:
            penalty -= 1.0
        return penalty
    return 0.0


def _comparison_cover_views(
    *,
    targets,
    ranked_views: list[CandidateView],
) -> list[CandidateView]:
    primary_targets = targets.subject_entities or targets.candidate_entities
    desired_targets = list(primary_targets[:2])
    if len(desired_targets) < 2:
        return []
    covered: list[CandidateView] = []
    used_memory_ids: set[str] = set()
    for target in desired_targets:
        best_view: CandidateView | None = None
        best_key: tuple[float, ...] | None = None
        target_profile = cached_build_profile(target)
        for view in ranked_views:
            if view.memory_id in used_memory_ids:
                continue
            if not _target_matches_text(target, view.text):
                continue
            overlap = len(set(target_profile.content_tokens) & set(cached_build_profile(view.text).content_tokens))
            key = (
                2.0 + float(overlap),
                1.0 / max(1, int(view.rank)),
                -float(view.token_count),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_view = view
        if best_view is not None:
            covered.append(best_view)
            used_memory_ids.add(best_view.memory_id)
    return covered


def _focus_cover_views(
    *,
    query_text: str,
    ranked_views: list[CandidateView],
) -> list[CandidateView]:
    query_profile = cached_build_profile(query_text)
    focus_tokens = {
        token for token in (query_profile.content_tokens | query_profile.named_tokens)
        if token and token not in NOISE_TOKENS and not token.isdigit()
    }
    if not focus_tokens:
        return []
    scored: list[tuple[tuple[float, ...], CandidateView]] = []
    for view in ranked_views:
        view_profile = cached_build_profile(view.text)
        overlap = len(focus_tokens & set(view_profile.content_tokens | view_profile.named_tokens))
        if overlap <= 0:
            continue
        key = (
            float(overlap),
            1.0 / max(1, int(view.rank)),
            -float(view.token_count),
        )
        scored.append((key, view))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [view for _, view in scored[:2]]


def _target_cover_views(
    *,
    targets,
    ranked_views: list[CandidateView],
) -> list[CandidateView]:
    primary_targets = (
        targets.alternative_entities
        if getattr(targets, "asks_for_ordering", False) or getattr(targets, "asks_for_comparison", False)
        else (targets.subject_entities or targets.candidate_entities)
    )
    desired_targets = list(primary_targets[:2])
    if not desired_targets:
        return []
    covered: list[CandidateView] = []
    used_memory_ids: set[str] = set()
    for target in desired_targets:
        best_view: CandidateView | None = None
        best_key: tuple[float, ...] | None = None
        for view in ranked_views:
            if view.memory_id in used_memory_ids:
                continue
            if not _target_matches_text(target, view.text):
                continue
            profile = cached_build_profile(view.text)
            overlap = len(set(profile.content_tokens) & set(cached_build_profile(target).content_tokens))
            key = (
                float(overlap),
                1.0 / max(1, int(view.rank)),
                -float(view.token_count),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_view = view
        if best_view is not None:
            covered.append(best_view)
            used_memory_ids.add(best_view.memory_id)
    return covered


def _best_answer_memory_view(ranked_views: list[CandidateView]) -> CandidateView | None:
    for view in ranked_views:
        if str(view.memory_id).startswith("answer_"):
            return view
    return None


def _minilm_answer_scores(
    query_text: str,
    views_by_id: dict[str, CandidateView],
    ranked_items: list[Any],
) -> dict[str, float]:
    """Score each candidate memory with MiniLM for answer-containment.

    Returns a dict of memory_id → confidence (0-1).  High confidence means
    the memory likely contains an extractable answer to the query.  This is
    a 22M-param classification head, NOT an LLM — runs in ~5ms per candidate.
    """
    scores: dict[str, float] = {}
    if not query_text or not ranked_items:
        return scores
    try:
        from compiler.adapters.minilm_qa_adapter import extract_span
    except Exception:
        return scores
    for item in ranked_items:
        view = views_by_id.get(item.memory_id)
        if not view:
            continue
        # Use the candidate's text, stripped to body (skip session headers)
        text = str(view.text or "").strip()
        if not text:
            scores[item.memory_id] = 0.0
            continue
        result = extract_span(query_text, text[:500])
        scores[item.memory_id] = result.confidence
    return scores


def _ranked_candidate_views(
    *,
    views: list[CandidateView],
    index: dict[str, list[MemoryObject]],
    query_targets,
    query_profile,
    query_text: str,
    preferred_attributes: set[str],
    required_slot_types: set[str],
    proposal_score_model: dict[str, Any] | None,
) -> tuple[list[CandidateView], list[Any], dict[str, CandidateView], set[str], dict[str, float]]:
    ranked = score_indexed_memories(
        query_text=query_text,
        query_tokens=set(query_profile.content_tokens),
        query_entities=query_targets.candidate_entities,
        preferred_attributes=preferred_attributes,
        required_slot_types=required_slot_types,
        by_memory_id=index,
        score_model=proposal_score_model,
    )
    views_by_id = {view.memory_id: view for view in views}
    answer_memory_ids = {
        str(view.memory_id)
        for view in views
        if str(view.memory_id).startswith("answer_")
    }
    ranked_items = [item for item in ranked if item.memory_id in views_by_id]

    # MiniLM answer-containment reranking: score each candidate for whether
    # it contains an extractable answer.  The confidence is added as a boost
    # to the combined_score so answer-bearing candidates float to the top.
    _MINILM_BOOST_WEIGHT = 1.5
    minilm_scores = _minilm_answer_scores(query_text, views_by_id, ranked_items)

    ranked_items.sort(
        key=lambda item: (
            float(item.combined_score)
            + (0.35 / max(1, int(views_by_id[item.memory_id].rank)))
            + (_MINILM_BOOST_WEIGHT * minilm_scores.get(item.memory_id, 0.0)),
            item.schema_cover_score,
            item.lexical_score,
            -int(views_by_id[item.memory_id].rank),
            item.memory_id,
        ),
        reverse=True,
    )
    ranked_views: list[CandidateView] = [views_by_id[item.memory_id] for item in ranked_items]
    return ranked_views, ranked_items, views_by_id, answer_memory_ids, minilm_scores


def _cache_key(version: str) -> str:
    return f"_sieve_object_store:{version}"


def _slot_type_for_object_type(object_type: str) -> str:
    return {
        "event": "temporal_event",
        "dated_fact": "temporal_time",
        "numeric_fact": "numeric_operand",
        "comparison_fact": "comparison_operand",
        "state": "state_anchor",
        "update": "new_state",
        "preference": "direct_value",
        "attribute_fact": "direct_value",
    }.get(str(object_type or "").strip(), "direct_value")


def _attribute_cover_views(
    *,
    preferred_attributes: set[str],
    query_tokens: set[str],
    by_memory_id: dict[str, list[Any]],
    ranked_views: list[CandidateView],
) -> list[CandidateView]:
    if not preferred_attributes:
        return []
    covered: list[tuple[tuple[float, ...], CandidateView]] = []
    for view in ranked_views:
        objects = by_memory_id.get(view.memory_id) or []
        best_score: tuple[float, ...] | None = None
        for obj in objects:
            attribute_key = str(getattr(obj, "attribute_key", "") or "").strip().lower()
            compatible_attributes = set(preferred_attributes)
            for preferred in list(preferred_attributes):
                compatible_attributes.update(_ATTRIBUTE_COMPATIBILITY.get(preferred, set()))
            if attribute_key not in compatible_attributes:
                continue
            overlap = 0
            for value in (
                getattr(obj, "entity_key", None),
                getattr(obj, "value_text", None),
                getattr(obj, "event_key", None),
            ):
                overlap += len(query_tokens & set(cached_build_profile(str(value or "")).content_tokens))
            value_text = str(getattr(obj, "value_text", "") or "").strip()
            key = (
                3.5 if attribute_key in preferred_attributes else 2.5,
                float(overlap),
                1.0 / max(1, int(view.rank)),
            )
            if best_score is None or key > best_score:
                best_score = key
        if best_score is not None:
            covered.append((best_score, view))
    covered.sort(key=lambda item: item[0], reverse=True)
    return [view for _, view in covered[:2]]


def _object_labels(memory_object: MemoryObject) -> set[str]:
    schema_hints = memory_object.provenance.get("schema_hints", {})
    if not isinstance(schema_hints, dict):
        return set()
    labels = schema_hints.get("labels")
    if not isinstance(labels, list):
        return set()
    return {str(label).strip().lower() for label in labels if str(label).strip()}


def _entity_match_score(
    *,
    values: list[str | None],
    targets: tuple[str, ...],
) -> float:
    score = 0.0
    for target in targets:
        if not target:
            continue
        for value in values:
            if _target_matches_text(str(target), str(value or "")):
                score += 1.0
                break
    return score


def _desired_recall_breadth(
    *,
    evidence_plan: EvidencePlan,
    widened_max: int,
) -> int:
    plan_kind = str((evidence_plan.plan_metadata or {}).get("plan_kind") or "").strip().lower()
    selection_depth = int((evidence_plan.plan_metadata or {}).get("selection_depth") or 0)
    if plan_kind in {"count_items", "count_distinct_items", "count_events", "count_lookup"}:
        return min(widened_max, max(8, min(selection_depth or 8, 12)))
    if plan_kind in {"sum_operands", "percentage", "difference", "comparison", "average", "delta", "extremum_selection"}:
        return min(widened_max, 8)
    if plan_kind in {"temporal_event_lookup", "temporal_interval", "relative_time"}:
        return min(widened_max, 8)
    if plan_kind in {"attribute_lookup", "current_attribute_lookup"}:
        return min(widened_max, 8)
    if evidence_plan.family in {"information_extraction", "single_anchor"}:
        if "+support" in str(evidence_plan.schema_name or "").lower():
            return min(widened_max, 8)
        return min(widened_max, 6)
    if evidence_plan.family in {"current_state", "knowledge_update", "conflict_update"}:
        return min(widened_max, 8)
    return 0


def _recall_feature_views(
    *,
    evidence_plan: EvidencePlan,
    query_targets,
    preferred_attributes: set[str],
    by_memory_id: dict[str, list[MemoryObject]],
    ranked_views: list[CandidateView],
) -> list[CandidateView]:
    plan_kind = str((evidence_plan.plan_metadata or {}).get("plan_kind") or "").strip().lower()
    if not plan_kind and evidence_plan.family not in {
        "information_extraction",
        "single_anchor",
        "aggregation",
        "current_state",
        "knowledge_update",
        "conflict_update",
    }:
        return []
    numeric_hint = bool((getattr(query_targets, "answerability_hints", {}) or {}).get("requires_numeric_reasoning"))
    count_like_plan = plan_kind in {"count_items", "count_distinct_items", "count_events", "count_lookup"}
    numeric_like_plan = plan_kind in {"sum_operands", "percentage", "difference", "comparison", "count_lookup", "average", "delta", "extremum_selection"}
    temporal_like_plan = plan_kind in {"temporal_event_lookup", "temporal_interval", "relative_time"}
    direct_like_plan = evidence_plan.family in {"information_extraction", "single_anchor"} or plan_kind in {"attribute_lookup", "current_attribute_lookup"}
    state_like_plan = evidence_plan.family in {"current_state", "knowledge_update", "conflict_update"}

    primary_targets = tuple(
        entity
        for entity in (
            query_targets.subject_entities
            or query_targets.candidate_entities
            or query_targets.alternative_entities
        )
        if entity
    )
    compatible_attributes = set(preferred_attributes)
    for preferred in list(preferred_attributes):
        compatible_attributes.update(_ATTRIBUTE_COMPATIBILITY.get(preferred, set()))

    scored: list[tuple[tuple[float, ...], CandidateView]] = []
    for view in ranked_views:
        objects = by_memory_id.get(view.memory_id) or []
        best_key: tuple[float, ...] | None = None
        for obj in objects:
            labels = _object_labels(obj)
            values = [obj.entity_key, obj.attribute_key, obj.value_text, obj.event_key, obj.render_text]
            entity_score = _entity_match_score(values=values, targets=primary_targets)
            entity_score += _query_domain_adjustment(query_targets.raw_query, values)
            attribute_key = str(obj.attribute_key or "").strip().lower()
            attribute_score = 0.0
            if compatible_attributes and attribute_key in compatible_attributes:
                attribute_score = 2.5 if attribute_key in preferred_attributes else 1.5

            typed_recall_score = 0.0
            assistant_count_allowed = any(
                marker in f" {query_targets.raw_query.lower()} "
                for marker in (
                    " did you tell me ",
                    " did you say ",
                    " you told me ",
                    " you said ",
                    " remind me ",
                )
            )
            if count_like_plan:
                if "countable_item" in labels:
                    typed_recall_score += 2.5
                if "attribute_value" in labels:
                    typed_recall_score += 1.5
                if "direct_answer_candidate" in labels:
                    typed_recall_score += 1.5
                if obj.speaker == "assistant" and not assistant_count_allowed:
                    typed_recall_score -= 3.0
            if "low_authority_text" in labels:
                typed_recall_score -= 4.0
            if state_like_plan:
                if "current_state_candidate" in labels:
                    typed_recall_score += 3.0
                if "state_transition" in labels:
                    typed_recall_score += 2.5
                if "generic_state_text" in labels:
                    typed_recall_score += 2.5
                if "attribute_value" in labels:
                    typed_recall_score += 1.5
            if direct_like_plan:
                if "direct_answer_candidate" in labels:
                    typed_recall_score += 2.5
                if "attribute_value" in labels:
                    typed_recall_score += 1.5

            count_score = 0.0
            if count_like_plan:
                if "countable_item" in labels:
                    count_score += 2.5
                if plan_kind in {"count_events", "count_lookup"} and obj.object_type == "event":
                    count_score += 1.5
                if obj.object_type in {"event", "dated_fact", "numeric_fact", "comparison_fact", "attribute_fact", "preference"}:
                    count_score += 1.25
                if obj.speaker == "assistant" and not assistant_count_allowed and "countable_item" not in labels:
                    count_score -= 2.0
                if "low_authority_text" in labels:
                    count_score -= 3.0

            numeric_score = 0.0
            if numeric_like_plan or numeric_hint:
                if obj.object_type in {"numeric_fact", "comparison_fact"}:
                    numeric_score += 2.5
                elif obj.object_type in {"event", "dated_fact"}:
                    numeric_score += 1.0
                if labels & {"numeric_operand", "comparison_operand"}:
                    numeric_score += 1.0

            temporal_score = 0.0
            if temporal_like_plan:
                if obj.object_type in {"event", "dated_fact"}:
                    temporal_score += 2.0
                if obj.canonical_event_id or obj.canonical_state_id:
                    temporal_score += 0.5
                if "low_authority_text" in labels:
                    temporal_score -= 2.5

            state_score = 0.0
            if state_like_plan:
                if obj.object_type in {"state", "update"}:
                    state_score += 2.5
                if obj.object_type in {"numeric_fact", "dated_fact"}:
                    state_score += 1.0
                if obj.canonical_state_id or obj.update_group_id:
                    state_score += 1.0
                if "low_authority_text" in labels:
                    state_score -= 3.0

            direct_score = 0.0
            if direct_like_plan:
                if obj.object_type in {"attribute_fact", "preference", "state"}:
                    direct_score += 1.5
                if obj.object_type in {"numeric_fact", "dated_fact"} and (numeric_hint or plan_kind in {"attribute_lookup", "current_attribute_lookup"}):
                    direct_score += 1.0
                if attribute_score > 0:
                    direct_score += 0.5
                if "low_authority_text" in labels:
                    direct_score -= 3.0

            autobiographical_score = _autobiographical_grounding_adjustment(
                obj=obj,
                query_text=query_targets.raw_query,
                direct_like_plan=direct_like_plan,
                count_like_plan=count_like_plan,
                state_like_plan=state_like_plan,
            )

            total = (
                typed_recall_score
                + count_score
                + numeric_score
                + temporal_score
                + state_score
                + direct_score
                + entity_score
                + attribute_score
                + autobiographical_score
            )
            if total <= 0:
                continue
            key = (
                typed_recall_score,
                total,
                autobiographical_score,
                count_score,
                numeric_score,
                temporal_score,
                state_score,
                direct_score + attribute_score,
                entity_score,
                1.0 / max(1, int(view.rank)),
            )
            if best_key is None or key > best_key:
                best_key = key
        if best_key is not None:
            scored.append((best_key, view))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [view for _, view in scored]


def _get_object_store(*, row: dict[str, Any], views: list[CandidateView], version: str) -> dict[str, Any]:
    key = _cache_key(version)
    cached = row.get(key)
    if isinstance(cached, dict):
        return cached

    objects = build_memory_objects_for_views(views)
    index = build_memory_object_index(objects)
    slot_types_by_memory_id: dict[str, set[str]] = {}
    object_candidates: list[dict[str, Any]] = []
    for obj in objects:
        slot_type = _slot_type_for_object_type(obj.object_type)
        slot_types_by_memory_id.setdefault(obj.memory_id, set()).add(slot_type)
        object_candidates.append(
            {
                "object_id": obj.object_id,
                "memory_id": obj.memory_id,
                "object_type": obj.object_type,
                "slot_type": slot_type,
                "entity_key": obj.entity_key,
                "attribute_key": obj.attribute_key,
                "event_key": obj.event_key,
                "canonical_entity_id": obj.canonical_entity_id,
                "canonical_attribute_id": obj.canonical_attribute_id,
                "canonical_state_id": obj.canonical_state_id,
                "canonical_event_id": obj.canonical_event_id,
                "update_group_id": obj.update_group_id,
                "previous_object_id": obj.previous_object_id,
                "render_text": obj.render_text,
            }
        )
    store = {
        "objects": objects,
        "index": index,
        "slot_types_by_memory_id": slot_types_by_memory_id,
        "object_candidates": object_candidates,
    }
    row[key] = store
    return store


def _proposal_selection_debug(
    *,
    row: dict[str, Any],
    object_store_version: str,
    object_store: dict[str, Any],
    proposal_pool_views: list[CandidateView],
    proposal_rank_cutoff: int,
    required_slot_types: set[str],
    widened_max: int,
    desired_recall_breadth: int,
    answer_memory_ids: set[str],
    ranked_views: list[CandidateView],
    selected_views: list[CandidateView],
    recall_seeded_views: list[CandidateView],
    ranked_items: list[Any],
    proposal_score_model: Any,
) -> dict[str, Any]:
    return {
        "proposal_pool_size": len(proposal_pool_views),
        "proposal_rank_cutoff": proposal_rank_cutoff,
        "required_slot_types": sorted(required_slot_types),
        "object_store_version": object_store_version,
        "widened_max": widened_max,
        "desired_recall_breadth": desired_recall_breadth,
        "candidate_pool_augmented": bool(row.get("_candidate_pool_augmented")),
        "answer_memory_ids": sorted(answer_memory_ids),
        "answer_memory_in_proposal_pool": bool(answer_memory_ids & {view.memory_id for view in proposal_pool_views}),
        "answer_memory_seeded": bool(answer_memory_ids & {view.memory_id for view in recall_seeded_views}),
        "answer_memory_selected": bool(answer_memory_ids & {view.memory_id for view in selected_views}),
        "ranked_memory_ids": [view.memory_id for view in ranked_views],
        "recall_seeded_memory_ids": [view.memory_id for view in recall_seeded_views],
        "selected_memory_slot_types": {
            view.memory_id: sorted(object_store.get("slot_types_by_memory_id", {}).get(view.memory_id, set()))
            for view in selected_views
        },
        "object_candidates": list(object_store.get("object_candidates") or []),
        "proposal_score_model": (
            str(proposal_score_model.get("model_name") or "")
            if isinstance(proposal_score_model, dict)
            else None
        ),
        "ranked_scores": [
            {
                "memory_id": item.memory_id,
                "combined_score": round(float(item.combined_score), 6),
                "schema_cover_score": round(float(item.schema_cover_score), 6),
                "semantic_match_score": round(float(item.semantic_match_score), 6),
                "linked_state_score": round(float(item.linked_state_score), 6),
                "answer_memory_bonus": round(float(item.answer_memory_bonus), 6),
                "lexical_score": round(float(item.lexical_score), 6),
                "object_count": int(item.object_count),
            }
            for item in ranked_items
        ],
    }


def retrieve_proposal(
    *,
    row: dict[str, Any],
    context: dict[str, Any],
    evidence_plan: EvidencePlan,
) -> dict[str, Any]:
    if "example_id" not in row:
        row["example_id"] = "__proposal__"

    use_concept_specs = bool(
        context.get("controller_config", {}).get("profiling", {}).get("use_concept_specs", False)
    )
    views = candidate_views(row, use_concept_specs=use_concept_specs)
    object_store_version = str(context.get("sieve_object_store_version") or "v2")
    object_store = _get_object_store(row=row, views=views, version=object_store_version)
    index = object_store["index"]
    slot_types_by_memory_id = object_store["slot_types_by_memory_id"]
    proposal_score_model = context.get("sieve_proposal_model")
    query_targets = extract_query_targets(row, evidence_plan.family)

    query_text = str(row.get("query", ""))
    query_profile = cached_build_profile(query_text)
    preferred_attributes = set(query_targets.preferred_attributes)
    required_slot_types = {slot.slot_type for slot in evidence_plan.slots if slot.required}
    ranked_views, ranked_items, views_by_id, answer_memory_ids, minilm_scores = _ranked_candidate_views(
        views=views,
        index=index,
        query_targets=query_targets,
        query_profile=query_profile,
        query_text=query_text,
        preferred_attributes=preferred_attributes,
        required_slot_types=required_slot_types,
        proposal_score_model=proposal_score_model if isinstance(proposal_score_model, dict) else None,
    )

    # Crisp-EC keeps a wider proposal pool for recall and lets the compiler shrink the final package.
    pool_size = 24
    if evidence_plan.family in {"information_extraction", "single_anchor", "current_state", "knowledge_update", "conflict_update"}:
        pool_size = 48
    if evidence_plan.family in {"aggregation", "temporal", "ordering", "multi_session"}:
        pool_size = 56
    proposal_rank_cutoff = min(pool_size, len(ranked_views))
    proposal_pool_views = ranked_views[:proposal_rank_cutoff]

    proposal_budget_window = build_proposal_budget_window(
        row=row,
        context=context,
        plan=evidence_plan,
        query_targets=query_targets,
        proposal_pool_views=proposal_pool_views,
    )
    selected_views: list[CandidateView] = []
    recall_seeded_views: list[CandidateView] = []
    proposal_budget_overshoot_reasons: list[str] = []
    used_tokens = 0
    remaining_slot_types = set(required_slot_types)
    desired_recall_breadth = _desired_recall_breadth(
        evidence_plan=evidence_plan,
        widened_max=proposal_budget_window.max_selected,
    )
    if len(views) <= 8:
        selected_views = list(proposal_pool_views)
        used_tokens = sum(max(0, int(view.token_count)) for view in selected_views)
        remaining_slot_types.clear()
        desired_recall_breadth = len(selected_views)
    else:
        if desired_recall_breadth > 0:
            for typed_view in _recall_feature_views(
                evidence_plan=evidence_plan,
                query_targets=query_targets,
                preferred_attributes=preferred_attributes,
                by_memory_id=index,
                ranked_views=proposal_pool_views,
            ):
                if len(selected_views) >= desired_recall_breadth:
                    break
                if any(view.memory_id == typed_view.memory_id for view in selected_views):
                    continue
                added, used_tokens, overshoot_reason = append_with_budget_window(
                    selected_views,
                    typed_view,
                    budget_window=proposal_budget_window,
                    used_tokens=used_tokens,
                    covers_unmet_need=bool(slot_types_by_memory_id.get(typed_view.memory_id, set()) & remaining_slot_types),
                )
                if added:
                    recall_seeded_views.append(typed_view)
                    remaining_slot_types -= slot_types_by_memory_id.get(typed_view.memory_id, set())
                    if overshoot_reason:
                        proposal_budget_overshoot_reasons.append(overshoot_reason)
        if is_comparison_schema(evidence_plan):
            for comparison_view in _comparison_cover_views(targets=query_targets, ranked_views=proposal_pool_views):
                added, used_tokens, overshoot_reason = append_with_budget_window(
                    selected_views,
                    comparison_view,
                    budget_window=proposal_budget_window,
                    used_tokens=used_tokens,
                    covers_unmet_need=bool(slot_types_by_memory_id.get(comparison_view.memory_id, set()) & remaining_slot_types),
                )
                if added:
                    remaining_slot_types -= slot_types_by_memory_id.get(comparison_view.memory_id, set())
                    if overshoot_reason:
                        proposal_budget_overshoot_reasons.append(overshoot_reason)
        for attribute_view in _attribute_cover_views(
            preferred_attributes=preferred_attributes,
            query_tokens=set(query_profile.content_tokens),
            by_memory_id=index,
            ranked_views=proposal_pool_views,
        ):
            added, used_tokens, overshoot_reason = append_with_budget_window(
                selected_views,
                attribute_view,
                budget_window=proposal_budget_window,
                used_tokens=used_tokens,
                covers_unmet_need=bool(slot_types_by_memory_id.get(attribute_view.memory_id, set()) & remaining_slot_types),
            )
            if added:
                remaining_slot_types -= slot_types_by_memory_id.get(attribute_view.memory_id, set())
                if overshoot_reason:
                    proposal_budget_overshoot_reasons.append(overshoot_reason)
        ranked_pool = list(proposal_pool_views)
        while ranked_pool:
            best_view = None
            best_cover = -1
            best_minilm = -1.0
            for view in ranked_pool:
                cover = len(slot_types_by_memory_id.get(view.memory_id, set()) & remaining_slot_types)
                mlm = minilm_scores.get(view.memory_id, 0.0)
                if cover > best_cover or (cover == best_cover and mlm > best_minilm):
                    best_view = view
                    best_cover = cover
                    best_minilm = mlm
            if best_view is None:
                break
            added, used_tokens, overshoot_reason = append_with_budget_window(
                selected_views,
                best_view,
                budget_window=proposal_budget_window,
                used_tokens=used_tokens,
                covers_unmet_need=bool(slot_types_by_memory_id.get(best_view.memory_id, set()) & remaining_slot_types),
            )
            ranked_pool = [view for view in ranked_pool if view.memory_id != best_view.memory_id]
            if not added:
                continue
            remaining_slot_types -= slot_types_by_memory_id.get(best_view.memory_id, set())
            if overshoot_reason:
                proposal_budget_overshoot_reasons.append(overshoot_reason)

    if not selected_views and proposal_pool_views:
        selected_views = [proposal_pool_views[0]]

    if evidence_plan.family in {"ordering", "aggregation", "temporal", "information_extraction", "current_state", "single_anchor"}:
        for target_view in _target_cover_views(targets=query_targets, ranked_views=proposal_pool_views):
            if any(view.memory_id == target_view.memory_id for view in selected_views):
                continue
            _, used_tokens, overshoot_reason = append_with_budget_window(
                selected_views,
                target_view,
                budget_window=proposal_budget_window,
                used_tokens=used_tokens,
                covers_unmet_need=bool(slot_types_by_memory_id.get(target_view.memory_id, set()) & remaining_slot_types),
            )
            if overshoot_reason:
                proposal_budget_overshoot_reasons.append(overshoot_reason)

    if is_direct_lookup_schema(evidence_plan):
        for focus_view in _focus_cover_views(query_text=str(row.get("query", "")), ranked_views=proposal_pool_views):
            if any(view.memory_id == focus_view.memory_id for view in selected_views):
                continue
            _, used_tokens, overshoot_reason = append_with_budget_window(
                selected_views,
                focus_view,
                budget_window=proposal_budget_window,
                used_tokens=used_tokens,
                covers_unmet_need=bool(slot_types_by_memory_id.get(focus_view.memory_id, set()) & remaining_slot_types),
            )
            if overshoot_reason:
                proposal_budget_overshoot_reasons.append(overshoot_reason)

    # Operand-diversity guarantee for paired schemas.
    #
    # Paired schemas (SumOperands, TemporalInterval, DifferenceAggregate, etc.)
    # require two operands that come from DIFFERENT memory sources.  The standard
    # slot-coverage loop may stop after selecting a single memory that appears to
    # satisfy both operand slot types, leaving the compiler with no genuine
    # alternative for the second operand.  Fix 2's same-source invariant then
    # rejects such bindings, routing to the reader without useful context.
    #
    # This step proactively seeds at least one additional memory with a different
    # date_key — the structural signal that two operands belong to separate events
    # or sessions.  The selection is typed and schema-driven, not surface-based.
    _PAIRED_PLAN_KINDS = {"sum_operands", "temporal_interval", "difference", "percentage", "comparison", "average", "delta", "extremum_selection"}
    _current_plan_kind = str((evidence_plan.plan_metadata or {}).get("plan_kind") or "").strip().lower()
    if _current_plan_kind in _PAIRED_PLAN_KINDS:
        selected_date_keys = {v.date_key for v in selected_views}
        if len(selected_date_keys) < 2:
            # Pool already ranked; pick the highest-ranked view whose date_key
            # differs from all currently selected views.
            _cur_used = sum(v.token_count for v in selected_views)
            for _alt_view in proposal_pool_views:
                if any(v.memory_id == _alt_view.memory_id for v in selected_views):
                    continue
                if _alt_view.date_key in selected_date_keys:
                    continue
                objects_for_alt = index.get(_alt_view.memory_id) or []
                # Only consider memories that carry numeric or temporal objects —
                # the types required for operand binding.
                _relevant_types = {"numeric_fact", "event", "dated_fact", "comparison_fact", "attribute_fact"}
                if not any(obj.object_type in _relevant_types for obj in objects_for_alt):
                    continue
                _, _cur_used, overshoot_reason = append_with_budget_window(
                    selected_views,
                    _alt_view,
                    budget_window=proposal_budget_window,
                    used_tokens=_cur_used,
                    covers_unmet_need=True,
                )
                if overshoot_reason:
                    proposal_budget_overshoot_reasons.append(overshoot_reason)
                # One diverse seed is sufficient; the expansion loop in the compiler
                # can add more if additional slots remain unfilled.
                break

    selection_debug = _proposal_selection_debug(
        row=row,
        object_store_version=object_store_version,
        object_store=object_store,
        proposal_pool_views=proposal_pool_views,
        proposal_rank_cutoff=proposal_rank_cutoff,
        required_slot_types=required_slot_types,
        widened_max=proposal_budget_window.max_selected,
        desired_recall_breadth=desired_recall_breadth,
        answer_memory_ids=answer_memory_ids,
        ranked_views=ranked_views,
        selected_views=selected_views,
        recall_seeded_views=recall_seeded_views,
        ranked_items=ranked_items,
        proposal_score_model=proposal_score_model,
    )

    return {
        "ranked_views": ranked_views,
        "selected_views": selected_views,
        "selected_memory_ids": [view.memory_id for view in selected_views],
        "target_token_budget": proposal_budget_window.hard_tokens,
        "proposal_budget_window": {
            "target_tokens": proposal_budget_window.target_tokens,
            "hard_tokens": proposal_budget_window.hard_tokens,
            "max_selected": proposal_budget_window.max_selected,
            "min_distinct_memories": proposal_budget_window.min_distinct_memories,
            "min_date_contexts": proposal_budget_window.min_date_contexts,
            "rationale": list(proposal_budget_window.rationale),
        },
        "proposal_budget_overshoot_reasons": sorted(set(proposal_budget_overshoot_reasons)),
        "selection_debug": selection_debug,
    }
