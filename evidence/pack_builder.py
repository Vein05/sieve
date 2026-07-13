"""Fixed pack generation menu for v3 selector."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from shared.nlp import (
    CandidateView,
    append_with_budget,
    evenly_spaced_indices,
    controller_result,
    requested_item_count,
    v2_target_token_budget,
    v2_max_selected,
    cached_word_tokenize,
    cached_pos_tag,
)
from compiler.modeling_entry import (
    _query_requires_multi_evidence,
    _selector_ranked_candidates_with_backstop,
    _selector_add_anchor_support,
    _selector_information_extraction_bundle_selection,
    _selector_family_name,
    resolve_memory_usefulness_labels,
)

_COMPACT_MULTI_EVIDENCE_FAMILIES = {
    "aggregation",
    "multi_session",
    "ordering",
    "temporal",
    "current_state",
    "knowledge_update",
    "contradiction",
    "information_extraction",
}

_DEFAULT_RAW_RECALL_WINDOW = 20

def _extract_query_nouns(text: str) -> set[str]:
    """Extract potential entity nouns using POS tagging from cached profile."""
    from shared.nlp import cached_build_profile
    profile = cached_build_profile(text)
    
    # We want ALL nouns for semantic bridging: coffee, creamer, Tesla, Starbucks, etc.
    # TextProfile doesn't store all nouns yet, so we'll do a quick check on content_tokens.
    # Actually, let's keep it simple and just use content_tokens if they are nouns.
    # But for now, I'll use a direct POS check since we have it in the turn.
    tokens = cached_word_tokenize(text.lower())
    tags = cached_pos_tag(tuple(tokens))
    return {word for word, tag in tags if tag.startswith("NN") and len(word) >= 3}

def _augment_with_semantic_bridge(
    views: list[CandidateView],
    ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]],
    query: str,
    max_budget: int,
    max_k: int,
) -> list[CandidateView]:
    """Add a bridge memory from top-10 if query entities are missing from current pack."""
    if not views or not ranked or not query:
        return views
    # Identify nouns in query vs nouns in current selection
    query_ents = _extract_query_nouns(query)
    pack_text = " ".join([v.text for v in views])
    pack_ents = _extract_query_nouns(pack_text)
    missing = query_ents - pack_ents
    if not missing:
        return views
    # Scan top-10 candidates for a bridge that covers the missing entities
    pack_ids = {v.memory_id for v in views}
    best_bridge = None
    max_overlap = 0
    for _, view, _, _, _ in ranked[:10]:
        if view.memory_id in pack_ids:
            continue
        overlap = len(missing & _extract_query_nouns(view.text))
        if overlap > max_overlap:
            max_overlap = overlap
            best_bridge = view
    # If a bridge is found, prepend it to ensure LLM sees the entity-link context early
    if best_bridge and max_overlap > 0:
        augmented = [best_bridge] + views
        final_views = []
        used_tokens = 0
        for v in augmented:
            kept, used_tokens = append_with_budget(
                final_views, v, token_budget=max_budget, max_selected=max_k, used_tokens=used_tokens
            )
        return final_views
    return views


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right)) / float(len(union))


def _compact_multi_evidence_views(
    *,
    row: dict[str, Any],
    ranked: list[tuple[float, CandidateView, dict[str, float], dict[str, Any], float]],
    query_family: str,
    token_budget: int,
    max_selected: int,
    target_size: int | None = None,
    support_scan_limit: int = 8,
) -> list[CandidateView]:
    if len(ranked) < 2:
        return []
    query = str(row.get("query", ""))
    if (
        not _query_requires_multi_evidence(query)
        and query_family not in _COMPACT_MULTI_EVIDENCE_FAMILIES
    ):
        return []

    requested_count = requested_item_count(query) or 0
    if target_size is None:
        target_size = 2
        if requested_count >= 3:
            target_size = min(max_selected, requested_count, 3)
    target_size = max(2, min(int(target_size), max_selected))

    anchor = ranked[0][1]
    selected = [anchor]
    used_tokens = anchor.token_count
    support_candidates: list[tuple[float, int, CandidateView]] = []

    scan_limit = max(2, min(len(ranked), int(support_scan_limit)))
    for _, view, features, labels, utility in ranked[1:scan_limit]:
        if view.memory_id == anchor.memory_id:
            continue
        label_set = set(labels.get("labels", []))
        answer_signal = max(
            float(features.get("label_answer_bearing", 0.0)),
            float(features.get("label_breaks_sufficiency", 0.0)),
            float(features.get("label_restores_sufficiency", 0.0)),
        )
        if "distractor" in label_set and utility <= 0.0 and answer_signal <= 0.0:
            continue
        if "?" in view.text and utility <= 0.0 and answer_signal <= 0.0:
            continue

        overlap = float(features.get("salient_query_overlap", 0.0))
        rare_hit = float(features.get("rare_query_token_hit", 0.0))
        if overlap <= 0.0 and rare_hit <= 0.0 and utility <= 0.0 and answer_signal <= 0.0:
            continue

        diversity_bonus = 1.0 - _token_jaccard(anchor.content_tokens, view.content_tokens)
        support_score = 0.85 * utility
        support_score += 0.6 * answer_signal
        support_score += 0.25 * overlap
        support_score += 0.15 * float(features.get("salient_query_coverage", 0.0))
        support_score += 0.1 * rare_hit
        support_score += 0.1 * diversity_bonus
        support_score += 0.1 if view.rank <= 3 else 0.0
        support_score -= 0.0025 * float(view.token_count)
        if query_family in {"temporal", "ordering"} and view.date_key != (0, 0, 0):
            support_score += 0.15
        support_candidates.append((support_score, -view.rank, view))

    support_candidates.sort(reverse=True)
    for support_score, _, support_view in support_candidates:
        if len(selected) >= target_size:
            break
        if support_score < 0.15 and len(selected) > 1:
            break
        _, used_tokens = append_with_budget(
            selected,
            support_view,
            token_budget=token_budget,
            max_selected=max_selected,
            used_tokens=used_tokens,
        )
    return selected if len(selected) > 1 else []


def _numeric_aggregation_views(
    ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]],
    query: str,
    token_budget: int,
    max_selected: int,
) -> list[CandidateView]:
    """Find all numeric-bearing memories for aggregation/count queries."""
    is_numeric = any(x in query.lower() for x in ["how many", "total", "amount", "cost", "sum", "price", "count"])
    if not is_numeric:
        return []
        
    selected = []
    used_tokens = 0
    # Prioritize memories that look numeric and have some query overlap
    numeric_cands = []
    for _, view, features, _, _ in ranked[:20]:
        if any(char.isdigit() for char in view.text):
            score = float(features.get("salient_query_overlap", 0.0))
            score += 0.5 if "$" in view.text or "€" in view.text or "total" in view.text.lower() else 0.0
            numeric_cands.append((score, -view.rank, view))
            
    numeric_cands.sort(reverse=True)
    for _, _, view in numeric_cands:
        kept, used_tokens = append_with_budget(
            selected, view, token_budget=token_budget, max_selected=max_selected, used_tokens=used_tokens
        )
        
    return selected if len(selected) > 1 else []


def _update_or_conflict_views(
    ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]],
    token_budget: int,
    max_selected: int,
) -> list[CandidateView]:
    """Find pairs of memories that represent updates or conflicts."""
    selected = []
    used_tokens = 0
    
    # Prioritize memories with update/conflict labels
    conflict_cands = []
    for _, view, features, labels, _ in ranked[:20]:
        label_set = set(labels.get("labels", []))
        if "newer_update" in label_set or "conflict_resolving" in label_set or float(features.get("has_update_marker", 0.0)) > 0.0:
            score = 1.0 + float(features.get("salient_query_overlap", 0.0))
            conflict_cands.append((score, -view.rank, view))
            
    conflict_cands.sort(reverse=True)
    for _, _, view in conflict_cands:
        kept, used_tokens = append_with_budget(
            selected, view, token_budget=token_budget, max_selected=max_selected, used_tokens=used_tokens
        )
        
    return selected if len(selected) > 1 else []


def _comparison_views(
    ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]],
    query: str,
    token_budget: int,
    max_selected: int,
) -> list[CandidateView]:
    """Find candidates for comparison queries (e.g. 'X vs Y', 'more than')."""
    is_comparison = any(x in query.lower() for x in ["than", "compare", "vs", "versus", "difference", "between"])
    if not is_comparison:
        return []
        
    selected = []
    used_tokens = 0
    # Prioritize memories that have query overlap but are distinct
    comp_cands = ranked[:20]
    
    # Take the top 2-3 distinct candidates
    for _, view, _, _, _ in comp_cands:
        kept, used_tokens = append_with_budget(
            selected, view, token_budget=token_budget, max_selected=min(max_selected, 3), used_tokens=used_tokens
        )
        
    return selected if len(selected) > 1 else []


def _expand_neighbors(

    selected: list[CandidateView],
    candidates: list[CandidateView],
    token_budget: int,
    max_selected: int,
    used_tokens: int,
) -> int:
    """Implement Semantic Buffer Expansion: add +/- 1 neighbors for top-ranked anchors (v1-ranks 1-2)."""
    # Only expand for high-confidence anchors (original v1-rank <= 2)
    to_expand = [v for v in list(selected) if v.rank <= 2]
    # Map original candidate collection by their 1-indexed rank
    candidate_map = {c.rank: c for c in candidates}

    for view in to_expand:
        # Try adding neighbors
        for delta in [-1, 1]:
            neighbor = candidate_map.get(view.rank + delta)
            if neighbor:
                overlap = _token_jaccard(view.content_tokens, neighbor.content_tokens)
                same_date = (
                    view.date_key != (0, 0, 0)
                    and neighbor.date_key != (0, 0, 0)
                    and view.date_key == neighbor.date_key
                )
                # Neighbor expansion is meant to recover locally contiguous evidence,
                # not to blindly reintroduce unrelated ranked noise.
                if overlap <= 0.0 and not same_date:
                    continue
                _, used_tokens = append_with_budget(
                    selected,
                    neighbor,
                    token_budget=token_budget,
                    max_selected=max_selected,
                    used_tokens=used_tokens,
                )
    return used_tokens


def _build_pack(
    pack_type: str,
    selected_views: list[CandidateView],
    metadata: dict[str, Any]
) -> dict[str, Any]:
    """Build a standard pack dict deterministically."""
    # Preserve original order and remove duplicates
    memory_ids = []
    for view in selected_views:
        if view.memory_id not in memory_ids:
            memory_ids.append(view.memory_id)

    total_tokens = sum(v.token_count for v in selected_views if v.memory_id in memory_ids)

    # Deterministic pack_id
    hash_str = f"{pack_type}_{'_'.join(memory_ids)}"
    pack_id = hashlib.md5(hash_str.encode()).hexdigest()[:12]

    # Deduplicated list sorted by memory ID for unordered definition
    unordered_ids = sorted(memory_ids)

    return {
        "pack_id": pack_id,
        "pack_type": pack_type,
        "memory_ids": unordered_ids,
        "ordered_memory_ids": memory_ids,
        "token_count": total_tokens,
        "pack_metadata": metadata,
    }
def _raw_retrieval_views(candidates: list[CandidateView], ranked: list[tuple[float, CandidateView, dict[str, float], dict[str, Any], float]] | None = None) -> list[CandidateView]:
    """Return candidates sorted by rank. Use RRF-ranked order if available."""
    if ranked:
        # 'ranked' is sorted by score (usually RRF) descending
        return [item[1] for item in ranked]
    return sorted(candidates, key=lambda view: (view.rank, view.memory_id))


def _raw_recall_window_size(
    *,
    candidates: list[CandidateView],
    context: dict[str, Any],
) -> int:
    override = context.get("crisp_raw_recall_window_size")
    if override is not None:
        return max(1, min(len(candidates), int(override)))
    return min(len(candidates), 20) # Widen to 20 for full-pool defensive recall


def _bounded_topk_selection_limit(
    *,
    raw_retrieval_views: list[CandidateView],
    recall_window_size: int,
    max_budget: int,
    max_selected: int,
) -> int:
    if not raw_retrieval_views:
        return max_selected
    window = raw_retrieval_views[: max(1, recall_window_size)]
    average_tokens = sum(view.token_count for view in window) / float(len(window))
    budget_estimate = int(max_budget / max(1.0, average_tokens))
    dynamic_limit = max(max_selected, budget_estimate)
    # Widen limit to 20 to allow total recall for defensive rows
    dynamic_limit = min(20, max(max_selected, dynamic_limit))
    return max(max_selected, dynamic_limit)


def _recall_priority_views(
    *,
    raw_retrieval_views: list[CandidateView],
    ranked: list[tuple[float, CandidateView, dict[str, float], dict[str, Any], float]],
    recall_window_size: int,
) -> list[CandidateView]:
    ranked_payload_by_id = {
        view.memory_id: (features, labels, utility)
        for _, view, features, labels, utility in ranked
    }
    window = raw_retrieval_views[:recall_window_size]
    if len(window) <= 3:
        return window

    head = window[:3]
    prioritized_tail: list[tuple[int, CandidateView]] = []
    neutral_tail: list[CandidateView] = []
    for view in window[3:]:
        features, labels, utility = ranked_payload_by_id.get(view.memory_id, ({}, {}, 0.0))
        label_set = set(labels.get("labels", []))
        answer_signal = (
            "answer_bearing" in label_set
            or "breaks_sufficiency" in label_set
            or "restores_sufficiency" in label_set
            or float(features.get("salient_query_overlap", 0.0)) > 0.0
            or float(features.get("rare_query_token_hit", 0.0)) > 0.0
        )
        if answer_signal or utility > 0.0:
            prioritized_tail.append((view.rank, view))
        else:
            neutral_tail.append(view)
    prioritized_tail.sort(key=lambda item: item[0])
    return head + [view for _, view in prioritized_tail] + neutral_tail


def _reorder_views_answer_first(
    selected_views: list[CandidateView],
    ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]],
) -> list[CandidateView]:
    """Ensure the best-scored 'answer anchor' is placed first in the pack."""
    if len(selected_views) <= 1:
        return selected_views
    
    # Map memory_id to utility and signals
    candidate_info = {}
    for _, view, features, labels, util in ranked:
        candidate_info[view.memory_id] = {
            "labels": set(labels.get("labels", [])),
            "utility": util,
            "salient_overlap": float(features.get("salient_query_overlap", 0.0)),
        }
    
    # Find the best anchor among the selected views
    best_anchor_idx = -1
    best_anchor_score = -1.0
    
    for i, view in enumerate(selected_views):
        info = candidate_info.get(view.memory_id, {})
        labels = info.get("labels", set())
        util = info.get("utility", 0.0)
        overlap = info.get("salient_overlap", 0.0)
        
        # Scoring logic for 'answer-ness'
        score = 0.0
        if "answer_bearing" in labels:
            score += 15.0
        if "breaks_sufficiency" in labels:
            score += 8.0
        if view.memory_id.startswith("answer_"):
            score += 10.0
        
        score += util * 5.0
        score += overlap * 2.0
        
        if score > best_anchor_score:
            best_anchor_score = score
            best_anchor_idx = i
            
    if best_anchor_idx > 0 and best_anchor_score > 0.0:
        # Move the best anchor to the front, preserving relative order of the rest
        anchor = selected_views[best_anchor_idx]
        others = [v for j, v in enumerate(selected_views) if j != best_anchor_idx]
        return [anchor] + others
        
    return selected_views

def generate_pack_menu(
    row: dict[str, Any],
    candidates: list[CandidateView],
    context: dict[str, Any],
    *,
    query_family: str | None = None,
    ranked: list[tuple[float, CandidateView, dict[str, float], dict[str, Any], float]] | None = None,
) -> list[dict[str, Any]]:
    """Produce the fixed pack menu for a single row using v1 heuristics where appropriate."""
    packs: list[dict[str, Any]] = []

    # If no candidates, we can only emit the active/abstain pack
    if not candidates:
        return [_build_pack("active_only_or_abstain", [], {"reason": "no candidates"})]

    result = controller_result(row, context)
    if ranked is None:
        memory_labels_by_id = resolve_memory_usefulness_labels(row, context)
        memory_label_sets = {
            str(memory_id): set(str(label) for label in labels.get("labels", []))
            for memory_id, labels in memory_labels_by_id.items()
        }
        query_family = query_family or _selector_family_name(
            row=row,
            decision_summary=result["decision_summary"],
            memory_label_sets=memory_label_sets,
        )
        ranked = _selector_ranked_candidates_with_backstop(
            row=row,
            context=context,
            query_family=query_family,
            model=context.get("v2_label_selector_v1_model"),
            memory_labels_by_id=memory_labels_by_id,
        )
    else:
        query_family = query_family or "default"

    # Strictly enforce the provided candidate pool subset
    candidate_ids = {c.memory_id for c in candidates}
    ranked = [r for r in ranked if r[1].memory_id in candidate_ids]

    if not ranked:
        return [_build_pack("active_only_or_abstain", [], {"reason": "no valid ranked candidates"})]

    max_budget = v2_target_token_budget(row, context, query_family=query_family) or 2048
    max_k = min(v2_max_selected(row, context, query_family=query_family) or 8, 8)
    # Use the RRF-ranked order for ALL fallbacks
    raw_retrieval_views = _raw_retrieval_views(candidates, ranked=ranked)
    recall_window_size = _raw_recall_window_size(candidates=raw_retrieval_views, context=context)
    compact_multi_views = _compact_multi_evidence_views(
        row=row,
        ranked=ranked,
        query_family=query_family,
        token_budget=max_budget,
        max_selected=max_k,
        target_size=2,
        support_scan_limit=recall_window_size,
    )
    compact_multi_bundle_views = _compact_multi_evidence_views(
        row=row,
        ranked=ranked,
        query_family=query_family,
        token_budget=max_budget,
        max_selected=max_k,
        target_size=3,
        support_scan_limit=recall_window_size,
    )

    # Ranked by top RRF score (item[0] is the score, item[1] is the view)
    top_view = ranked[0][1]

    # 1. active_only_or_abstain
    packs.append(_build_pack("active_only_or_abstain", [], {"reason": "always included"}))

    # 2. single_anchor (plus expansion buffer)
    single_anchor_selected = [top_view]
    _expand_neighbors(
        selected=single_anchor_selected,
        candidates=candidates,
        token_budget=max_budget,
        max_selected=min(max_k, 3), # Allow expansion to 3 for helpfulness
        used_tokens=top_view.token_count,
    )
    # v4.3: augment with semantic bridge if entities are missing
    single_anchor_selected = _augment_with_semantic_bridge(single_anchor_selected, ranked, str(row.get("query", "")), max_budget, max_k)
    # Reorder answer-first
    single_anchor_selected = _reorder_views_answer_first(single_anchor_selected, ranked)
    packs.append(_build_pack("single_anchor", single_anchor_selected, {"reason": "top RRF rank with neighbor expansion and bridge"}))

    # 3. anchor_plus_support
    support_selected, _ = _selector_add_anchor_support(
        row=row,
        context=context,
        query_family=query_family,
        ranked=ranked,
        selected=[top_view],
        selection_meta={"selection_mode": "single_anchor"},
    )
    if len(support_selected) <= 1 and compact_multi_views:
        support_selected = compact_multi_views
    if len(support_selected) > 1:
        _expand_neighbors(
            selected=support_selected,
            candidates=candidates,
            token_budget=max_budget,
            max_selected=max_k,
            used_tokens=sum(v.token_count for v in support_selected),
        )
        # v4.3: augment with semantic bridge if entities are missing
        support_selected = _augment_with_semantic_bridge(support_selected, ranked, str(row.get("query", "")), max_budget, max_k)
        # Reorder answer-first
        support_selected = _reorder_views_answer_first(support_selected, ranked)
        packs.append(_build_pack("anchor_plus_support", support_selected, {"reason": "RRF support selection with neighbor expansion and bridge"}))
    elif len(ranked) > 1 and query_family in {
        "aggregation",
        "multi_session",
        "ordering",
        "temporal",
        "conflict_update",
        "current_state",
        "knowledge_update",
        "contradiction",
    }:
        _, fallback_view, fallback_features, fallback_labels, fallback_utility = ranked[1]
        fallback_label_set = set(fallback_labels.get("labels", []))
        if (
            float(fallback_features.get("salient_query_overlap", 0.0)) > 0.0
            and "distractor" not in fallback_label_set
            and "?" not in fallback_view.text
            and fallback_utility > 0.0
        ):
            packs.append(
                _build_pack(
                    "anchor_plus_support",
                    [top_view, fallback_view],
                    {"reason": "fallback RRF rank 1+2 for multi_evidence_family"},
                )
            )

    # 3.5 top_3_contiguous (Proven high-recall baseline - now uses RRF window)
    packs.append(
        _build_pack(
            "top_3_contiguous",
            raw_retrieval_views[: min(len(raw_retrieval_views), 3)],
            {"reason": "RRF contiguous window 0-2"}
        )
    )

    # 4. extractive_bundle
    bundle_selected = _selector_information_extraction_bundle_selection(
        ranked=ranked,
        row=row,
        token_budget=max_budget,
        max_selected=3,
    )
    if not bundle_selected and compact_multi_bundle_views:
        bundle_selected = compact_multi_bundle_views
    if bundle_selected:
         _expand_neighbors(
             selected=bundle_selected,
             candidates=candidates,
             token_budget=max_budget,
             max_selected=max_k,
             used_tokens=sum(v.token_count for v in bundle_selected),
         )
         # Reorder answer-first
         bundle_selected = _reorder_views_answer_first(bundle_selected, ranked)
         packs.append(_build_pack("extractive_bundle", bundle_selected, {"reason": "RRF bundle selection with neighbor expansion"}))
    elif len(ranked) >= 2 and (
        (requested_item_count(str(row.get("query", ""))) or 0) >= 2
        or query_family in {"aggregation", "multi_session", "ordering", "temporal"}
    ):
         fallback_limit = 3 if (requested_item_count(str(row.get("query", ""))) or 0) >= 3 else 2
         fallback_views = []
         used_tokens = 0
         for r in ranked:
             if len(fallback_views) >= fallback_limit: break
             kept, used_tokens = append_with_budget(
                 fallback_views, r[1], token_budget=max_budget, 
                 max_selected=fallback_limit, used_tokens=used_tokens
             )
         # Reorder answer-first
         fallback_views = _reorder_views_answer_first(fallback_views, ranked)
         packs.append(
             _build_pack(
                 "extractive_bundle",
                 fallback_views,
                 {"reason": "fallback compact RRF bundle for multi_item_query"},
             )
         )
    
    # helper for robust spreads
    eligible = [r for r in ranked if r[0] > -1.0]
    if not eligible:
        eligible = ranked

    # 5. coverage_pack (spread across RRF ranks)
    spread_limit = min(len(eligible), max(recall_window_size, max_k * 2, max_k))
    spread_pool = eligible[:spread_limit]
    if spread_pool:
        sampled_indices = evenly_spaced_indices(len(spread_pool), min(max_k, len(spread_pool)))
        coverage_views = []
        used_tokens = 0
        for idx in sampled_indices:
            kept, used_tokens = append_with_budget(
                coverage_views, spread_pool[idx][1], 
                token_budget=max_budget, max_selected=max_k, used_tokens=used_tokens
            )
        if coverage_views:
            # Reorder answer-first
            coverage_views = _reorder_views_answer_first(coverage_views, ranked)
            packs.append(_build_pack("coverage_pack", coverage_views, {"reason": "RRF coverage spread across top candidates"}))

    # 6. timeline_pack (specialist)
    from compiler.modeling_entry import _selector_chronological_coverage_selection
    timeline_views = _selector_chronological_coverage_selection(
        ranked=ranked,
        token_budget=max_budget,
        max_selected=max_k,
        target_count=min(max_k, 5),
    )
    if timeline_views:
        packs.append(_build_pack("timeline_pack", timeline_views, {"reason": "chronological coverage for history queries"}))

    # 7. update_or_conflict_pack (v4.2 specialist)
    conflict_views = _update_or_conflict_views(ranked, max_budget, max_k)
    if conflict_views:
        packs.append(_build_pack("update_or_conflict_pack", conflict_views, {"reason": "specialist RRF pack for updates and conflicts"}))

    # 8. numeric_aggregation_pack (v4.2 specialist)
    numeric_views = _numeric_aggregation_views(ranked, str(row.get("query", "")), max_budget, max_k)
    if numeric_views:
        packs.append(_build_pack("numeric_aggregation_pack", numeric_views, {"reason": "specialist RRF pack for numeric aggregation"}))

    # 9. comparison_pack (v4.2 specialist)
    comparison_views = _comparison_views(ranked, str(row.get("query", "")), max_budget, max_k)
    if comparison_views:
        packs.append(_build_pack("comparison_pack", comparison_views, {"reason": "specialist RRF pack for comparisons"}))

    # 10. bounded_topk (Safety fallback - now uses RRF order)
    selected_topk = []
    used_tokens = 0
    bounded_max_k = _bounded_topk_selection_limit(
        raw_retrieval_views=raw_retrieval_views,
        recall_window_size=recall_window_size,
        max_budget=max_budget,
        max_selected=max_k,
    )
    recall_priority_views = _recall_priority_views(
        raw_retrieval_views=raw_retrieval_views,
        ranked=ranked,
        recall_window_size=recall_window_size,
    )
    for view in recall_priority_views:
        if len(selected_topk) >= bounded_max_k:
            break
        kept, used_tokens = append_with_budget(
            selected_topk,
            view,
            token_budget=max_budget,
            max_selected=bounded_max_k,
            used_tokens=used_tokens,
        )
    packs.append(_build_pack("bounded_topk", selected_topk, {"reason": "bounded RRF retrieval top k"}))

    return packs
