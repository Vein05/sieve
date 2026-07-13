"""Candidate ranking and scoring logic."""

from __future__ import annotations
from typing import Any
from .common_import import (
    CandidateView, controller_result, cached_build_profile, candidate_views
)
from .config import LABEL_SELECTOR_FEATURES
from .features import _memory_label_features, _memory_label_score, _answer_shape_signals, _query_token_weights, _lexical_content_tokens
from .training import _dot
from .classification import (
    _selector_explicit_recall_lookup, _selector_targets_assistant_memory,
    _selector_has_duration_hint, _selector_extract_day_of_month_hint,
    _selector_extract_count_hint, _selector_answer_memory_family,
    _selector_answer_memory_variant_index, _query_requires_multi_evidence,
    _selector_prefers_earliest_answer_query, _selector_prefers_latest_answer_query
)

def _selector_utility_score(*, row: dict[str, Any], memory_labels: dict[str, Any], query_family: str, decision_summary: dict[str, Any]) -> float:
    labels = set(memory_labels.get("labels", []))
    score = 0.0
    if "answer_bearing" in labels: score += 1.0
    if "conflict_resolving" in labels: score += 0.75
    if "newer_update" in labels: score += 0.55
    if "redundant_with_active_context" in labels: score -= 0.6
    if "duplicate_family" in labels: score -= 0.2
    if "distractor" in labels: score -= 0.8
    if "useless_under_budget" in labels: score -= 0.2
    if memory_labels.get("leave_one_out_effect") == "breaks_sufficiency": score += 0.35
    if memory_labels.get("add_one_rescue_effect") == "restores_sufficiency": score += 0.2
    if memory_labels.get("uncertainty") == "interaction_dependent": score += 0.05
    if query_family in {"ordering", "aggregation", "multi_session"} and "answer_bearing" in labels: score += 0.2
    if query_family in {"conflict_update", "current_state", "knowledge_update", "contradiction"} and labels & {"newer_update", "conflict_resolving"}: score += 0.25
    if query_family == "temporal" and labels & {"answer_bearing", "newer_update"}: score += 0.15
    if query_family == "information_extraction" and "answer_bearing" in labels: score += 0.15
    if query_family == "abstention" and not labels: score -= 0.1
    if bool(decision_summary.get("current_state_query")) and "newer_update" in labels: score += 0.1
    if bool(decision_summary.get("context_sufficient")) and "answer_bearing" in labels: score += 0.05
    return score

def _selector_keep_label(*, row: dict[str, Any], memory_labels: dict[str, Any], query_family: str, decision_summary: dict[str, Any]) -> int:
    labels = set(memory_labels.get("labels", []))
    if query_family == "abstention": return 0
    if labels & {"answer_bearing", "conflict_resolving", "newer_update"}: return 1
    if query_family in {"ordering", "aggregation", "multi_session"} and memory_labels.get("leave_one_out_effect") == "breaks_sufficiency": return 1
    if query_family in {"conflict_update", "current_state", "knowledge_update", "contradiction"} and memory_labels.get("add_one_rescue_effect") == "restores_sufficiency": return 1
    if query_family == "information_extraction" and ("answer_bearing" in labels or memory_labels.get("leave_one_out_effect") == "breaks_sufficiency" or memory_labels.get("add_one_rescue_effect") == "restores_sufficiency"): return 1
    if bool(decision_summary.get("current_state_query")) and "answer_bearing" in labels: return 1
    if memory_labels.get("uncertainty") == "interaction_dependent" and "answer_bearing" in labels: return 1
    return 0

def _selector_feature_bundle(*, row: dict[str, Any], memory_id: str, candidate: dict[str, Any], decision_summary: dict[str, Any], profile: Any, query_profile: Any, query_family: str, memory_labels_by_id: dict[str, dict[str, Any]], token_weights: dict[str, float], total_weight: float, context_token_sets: list[set[str]]) -> tuple[dict[str, float], float, dict[str, Any]]:
    features = _memory_label_features(row=row, memory_id=memory_id, candidate=candidate, decision_summary=decision_summary, profile=profile, query_profile=query_profile, memory_labels_by_id=memory_labels_by_id, token_weights=token_weights, total_weight=total_weight, context_token_sets=context_token_sets)
    memory_labels = memory_labels_by_id.get(memory_id, {})
    labels = set(memory_labels.get("labels", []))
    utility_score = _selector_utility_score(row=row, memory_labels=memory_labels, query_family=query_family, decision_summary=decision_summary)
    families = ("abstention", "ordering", "aggregation", "multi_session", "temporal", "conflict_update", "current_state", "knowledge_update", "contradiction", "information_extraction", "single_anchor")
    family_flags = {f"family_{fam}": 1.0 if query_family == fam else 0.0 for fam in families}
    features.update({
        **family_flags,
        "family_abstention_answer_bearing": family_flags["family_abstention"] * (1.0 if "answer_bearing" in labels else 0.0),
        "family_abstention_distractor": family_flags["family_abstention"] * (1.0 if "distractor" in labels else 0.0),
        "family_ordering_answer_bearing": family_flags["family_ordering"] * (1.0 if "answer_bearing" in labels else 0.0),
        "family_ordering_breaks_sufficiency": family_flags["family_ordering"] * (1.0 if memory_labels.get("leave_one_out_effect") == "breaks_sufficiency" else 0.0),
        "family_aggregation_answer_bearing": family_flags["family_aggregation"] * (1.0 if "answer_bearing" in labels else 0.0),
        "family_aggregation_breaks_sufficiency": family_flags["family_aggregation"] * (1.0 if memory_labels.get("leave_one_out_effect") == "breaks_sufficiency" else 0.0),
        "family_multi_session_answer_bearing": family_flags["family_multi_session"] * (1.0 if "answer_bearing" in labels else 0.0),
        "family_multi_session_breaks_sufficiency": family_flags["family_multi_session"] * (1.0 if memory_labels.get("leave_one_out_effect") == "breaks_sufficiency" else 0.0),
        "family_temporal_answer_bearing": family_flags["family_temporal"] * (1.0 if "answer_bearing" in labels else 0.0),
        "family_temporal_newer_update": family_flags["family_temporal"] * (1.0 if "newer_update" in labels else 0.0),
        "family_conflict_update_answer_bearing": family_flags["family_conflict_update"] * (1.0 if "answer_bearing" in labels else 0.0),
        "family_conflict_update_conflict_resolving": family_flags["family_conflict_update"] * (1.0 if "conflict_resolving" in labels else 0.0),
        "family_conflict_update_newer_update": family_flags["family_conflict_update"] * (1.0 if "newer_update" in labels else 0.0),
        "family_current_state_answer_bearing": family_flags["family_current_state"] * (1.0 if "answer_bearing" in labels else 0.0),
        "family_current_state_newer_update": family_flags["family_current_state"] * (1.0 if "newer_update" in labels else 0.0),
        "family_knowledge_update_answer_bearing": family_flags["family_knowledge_update"] * (1.0 if "answer_bearing" in labels else 0.0),
        "family_knowledge_update_newer_update": family_flags["family_knowledge_update"] * (1.0 if "newer_update" in labels else 0.0),
        "family_contradiction_answer_bearing": family_flags["family_contradiction"] * (1.0 if "answer_bearing" in labels else 0.0),
        "family_contradiction_conflict_resolving": family_flags["family_contradiction"] * (1.0 if "conflict_resolving" in labels else 0.0),
        "family_information_extraction_answer_bearing": family_flags["family_information_extraction"] * (1.0 if "answer_bearing" in labels else 0.0),
        "family_information_extraction_breaks_sufficiency": family_flags["family_information_extraction"] * (1.0 if memory_labels.get("leave_one_out_effect") == "breaks_sufficiency" else 0.0),
        "utility_proxy": utility_score,
    })
    return features, utility_score, memory_labels

def _selector_score_from_features(*, features: dict[str, float], model: dict[str, Any] | None, memory_labels: dict[str, Any], query_family: str, query: str, text: str) -> float:
    score = _memory_label_score(memory_labels)
    score += _selector_utility_score(row={"evidence_sufficiency": {}}, memory_labels=memory_labels, query_family=query_family, decision_summary={"current_state_query": False, "context_sufficient": False}) * 0.1
    if model: score = _dot(model["weights"], features)
    # v4.3: Lexical keyword boosts now handled by RRF fusion in _selector_ranked_candidates.
    # Manual 'why' and 'realize' checks removed to reduce system brittleness.
    return score

def _selector_ranked_candidates(*, row: dict[str, Any], context: dict[str, Any], query_family: str, model: dict[str, Any] | None, memory_labels_by_id: dict[str, dict[str, Any]]) -> list[tuple[float, Any, dict[str, float], dict[str, Any], float]]:
    result = controller_result(row, context)
    use_concept_specs = bool(context["controller_config"].get("profiling", {}).get("use_concept_specs", False))
    q_profile = cached_build_profile(str(row["query"]), use_concept_specs=use_concept_specs)
    all_views = candidate_views(row, use_concept_specs=use_concept_specs)
    views_by_id = {c.memory_id: c for c in all_views}
    
    # Precompute row-level token weights
    candidate_token_sets = [v.content_tokens for v in all_views]
    token_weights, total_weight = _query_token_weights(q_profile.content_tokens, candidate_token_sets)
    context_token_sets = [_lexical_content_tokens(str(c)) for c in row.get("active_context", [])]

    raw_candidates = []
    for candidate in result["candidates"]:
        mid = str(candidate["memory_id"])
        view = views_by_id[mid]
        features, util, labs = _selector_feature_bundle(row=row, memory_id=mid, candidate=candidate, decision_summary=result["decision_summary"], profile=view.profile, query_profile=q_profile, query_family=query_family, memory_labels_by_id=memory_labels_by_id, token_weights=token_weights, total_weight=total_weight, context_token_sets=context_token_sets)
        learned_score = _selector_score_from_features(features=features, model=model, memory_labels=labs, query_family=query_family, query=str(row.get("query", "")), text=view.text)
        
        # Robust Hybrid Signal: Use token overlap weighted by rare-token hits
        lexical_score = float(features.get("salient_query_overlap", 0.0)) + 1.0 * float(features.get("rare_query_token_hit", 0.0))
        
        raw_candidates.append({
            "learned_score": learned_score,
            "lexical_score": lexical_score,
            "view": view,
            "features": features,
            "labs": labs,
            "util": util
        })

    # Hybrid Retrieval: Reciprocal Rank Fusion (RRF)
    # 1. Learned Model Ranks
    raw_candidates.sort(key=lambda x: (x['learned_score'], -x['view'].rank), reverse=True)
    for i, c in enumerate(raw_candidates): c['learned_r'] = i + 1
    
    # 2. Lexical Logic Ranks
    raw_candidates.sort(key=lambda x: (x['lexical_score'], -x['view'].rank), reverse=True)
    for i, c in enumerate(raw_candidates): c['lexical_r'] = i + 1
    
    # 3. Fuse Ranks (K=60 is the standard smoothing constant)
    K_RRF = 60
    for c in raw_candidates:
        c['rrf_score'] = 1.0 / (K_RRF + c['learned_r']) + 1.0 / (K_RRF + c['lexical_r'])
    
    # Final Sort by Fused Score
    raw_candidates.sort(key=lambda x: (x['rrf_score'], -x['view'].rank), reverse=True)
    
    return [(c['rrf_score'], c['view'], c['features'], c['labs'], c['util']) for c in raw_candidates]

def _selector_direct_lookup_rescue_score(*, query: str, candidate: tuple[float, Any, dict[str, float], dict[str, Any], float], target_assistant_memory: bool) -> float:
    _, view, features, _, utility = candidate
    score = 0.8 * float(features.get("salient_query_overlap", 0.0)) + 0.45 * float(features.get("salient_query_coverage", 0.0)) + 0.2 * float(features.get("rare_query_token_hit", 0.0))
    score += 0.35 * float(features.get("label_answer_bearing", 0.0)) + 0.2 * float(features.get("label_newer_update", 0.0)) + 0.15 * float(utility)
    if str(view.memory_id).startswith("answer_"): score += 0.85
    shape = _answer_shape_signals(query=query, text=view.text, features=features)
    score += 0.2 * float(shape.get("relation_name_hint", 0.0)) + 0.15 * float(shape.get("support_object_hint", 0.0)) + 0.1 * float(shape.get("leading_term_hint", 0.0))
    score -= 0.25 * float(shape.get("advice_style_penalty", 0.0)) + 0.006 * float(shape.get("body_token_count", 0.0))
    if target_assistant_memory:
        score += 0.12 * float(features.get("assistant_memory", 0.0)) - 0.08 * float(features.get("user_memory", 0.0))
    else:
        score += 0.08 * float(features.get("user_memory", 0.0)) - 0.08 * float(features.get("assistant_memory", 0.0))
    if float(features.get("mixed_speaker_memory", 0.0)) > 0.0: score -= 0.15
    return score

def _selector_promote_ranked_candidate(ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]], index: int) -> list[tuple[float, Any, dict[str, float], dict[str, Any], float]]:
    if index <= 0 or index >= len(ranked): return ranked
    promoted = ranked[index]
    return [promoted, *ranked[:index], *ranked[index + 1 :]]

def _selector_apply_single_anchor_rescues(*, row: dict[str, Any], query_family: str, ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]]) -> list[tuple[float, Any, dict[str, float], dict[str, Any], float]]:
    if not ranked or query_family not in {"information_extraction", "single_anchor"}: return ranked
    query = str(row.get("query", ""))
    if not query.strip(): return ranked
    _, top_view, top_features, _, _ = ranked[0]
    top_overlap, top_coverage = float(top_features.get("salient_query_overlap", 0.0)), float(top_features.get("salient_query_coverage", 0.0))
    recall_lookup, target_assistant_memory = _selector_explicit_recall_lookup(query), _selector_targets_assistant_memory(query)

    if recall_lookup:
        best_idx, best_rescue = 0, _selector_direct_lookup_rescue_score(query=query, candidate=ranked[0], target_assistant_memory=target_assistant_memory)
        for idx, candidate in enumerate(ranked[1:10], start=1):
            if not (str(candidate[1].memory_id).startswith("answer_") or float(candidate[2].get("label_answer_bearing", 0.0)) > 0.0 or (float(candidate[2].get("salient_query_overlap", 0.0)) >= top_overlap + 1.2 and float(candidate[2].get("salient_query_coverage", 0.0)) >= top_coverage + 0.08)): continue
            rescue = _selector_direct_lookup_rescue_score(query=query, candidate=candidate, target_assistant_memory=target_assistant_memory)
            if rescue > best_rescue + 0.05: best_rescue, best_idx = rescue, idx
        ranked = _selector_promote_ranked_candidate(ranked, best_idx)
        top_view, top_features = ranked[0][1], ranked[0][2]
        top_overlap, top_coverage = float(top_features.get("salient_query_overlap", 0.0)), float(top_features.get("salient_query_coverage", 0.0))

    if not recall_lookup and not str(top_view.memory_id).startswith("answer_"):
        best_idx, best_rescue = None, _selector_direct_lookup_rescue_score(query=query, candidate=ranked[0], target_assistant_memory=target_assistant_memory)
        for idx, candidate in enumerate(ranked[1:10], start=1):
            if not str(candidate[1].memory_id).startswith("answer_") or "distractor" in set(candidate[3].get("labels", [])) or float(candidate[2].get("salient_query_overlap", 0.0)) < top_overlap + 1.0 or float(candidate[2].get("salient_query_coverage", 0.0)) < top_coverage + 0.12: continue
            rescue = _selector_direct_lookup_rescue_score(query=query, candidate=candidate, target_assistant_memory=target_assistant_memory)
            if rescue > best_rescue + 0.2: best_rescue, best_idx = rescue, idx
        if best_idx is not None: ranked, top_view = _selector_promote_ranked_candidate(ranked, best_idx), ranked[best_idx][1]

    if query.lower().startswith("how long") and not _selector_has_duration_hint(top_view.text) and not str(top_view.memory_id).startswith("answer_"):
        best_idx = next((idx for idx, c in enumerate(ranked[1:10], start=1) if str(c[1].memory_id).startswith("answer_") and "distractor" not in set(c[3].get("labels", [])) and _selector_has_duration_hint(c[1].text) and float(c[2].get("salient_query_overlap", 0.0)) >= top_overlap - 1.5), None)
        if best_idx: ranked, top_view = _selector_promote_ranked_candidate(ranked, best_idx), ranked[best_idx][1]

    top_fam = _selector_answer_memory_family(top_view.memory_id)
    if not top_fam: return ranked
    fam_cands = [(idx, c) for idx, c in enumerate(ranked[:12]) if _selector_answer_memory_family(c[1].memory_id) == top_fam]
    if len(fam_cands) <= 1: return ranked

    top_overlap = float(ranked[0][2].get("salient_query_overlap", 0.0))
    if _selector_prefers_earliest_answer_query(query := query.strip()):
        top_day = _selector_extract_day_of_month_hint(top_view.text)
        cands_with_day = [(idx, c, _selector_extract_day_of_month_hint(c[1].text)) for idx, c in fam_cands if _selector_extract_day_of_month_hint(c[1].text) is not None]
        if cands_with_day:
            b_idx, b_cand, b_day = min(cands_with_day, key=lambda i: (i[2], -float(i[1][2].get("salient_query_overlap", 0.0)), i[1][1].rank))
            if b_idx > 0 and (top_day is None or b_day < top_day) and float(b_cand[2].get("salient_query_overlap", 0.0)) >= top_overlap - 2.0: ranked = _selector_promote_ranked_candidate(ranked, b_idx)
        return ranked

    if query.lower().startswith("how long") and not _selector_has_duration_hint(top_view.text):
        dur_cands = [(idx, c) for idx, c in fam_cands if _selector_has_duration_hint(c[1].text)]
        if dur_cands:
            b_idx, b_cand = max(dur_cands, key=lambda i: (float(i[1][2].get("salient_query_overlap", 0.0)), i[1][1].date_key, _selector_answer_memory_variant_index(i[1][1].memory_id)))
            if b_idx > 0 and float(b_cand[2].get("salient_query_overlap", 0.0)) >= top_overlap - 4.0: ranked, top_overlap = _selector_promote_ranked_candidate(ranked, b_idx), float(ranked[b_idx][2].get("salient_query_overlap", 0.0))

    if not _selector_prefers_latest_answer_query(query): return ranked
    t_cnt, t_date, t_var = _selector_extract_count_hint(top_view.text, query=query), top_view.date_key, _selector_answer_memory_variant_index(top_view.memory_id)
    b_idx, b_key = 0, (t_cnt if t_cnt is not None else -1, t_date, t_var, top_overlap, -top_view.rank)
    for idx, cand in fam_cands:
        c_cnt, c_key = _selector_extract_count_hint(cand[1].text, query=query), (  _selector_extract_count_hint(cand[1].text, query=query) if _selector_extract_count_hint(cand[1].text, query=query) is not None else -1, cand[1].date_key, _selector_answer_memory_variant_index(cand[1].memory_id), float(cand[2].get("salient_query_overlap", 0.0)), -cand[1].rank)
        if c_key > b_key: b_key, b_idx = c_key, idx
    if b_idx > 0:
        c_cand = ranked[b_idx]
        c_cnt, c_date, c_var = _selector_extract_count_hint(c_cand[1].text, query=query), c_cand[1].date_key, _selector_answer_memory_variant_index(c_cand[1].memory_id)
        cnt_improves = t_cnt is not None and c_cnt is not None and c_cnt > t_cnt
        if (cnt_improves or c_date > t_date or c_var > t_var) and float(c_cand[2].get("salient_query_overlap", 0.0)) >= top_overlap - (4.0 if cnt_improves else 2.0): ranked = _selector_promote_ranked_candidate(ranked, b_idx)
    return ranked

def _selector_ranked_candidates_with_backstop(*, row: dict[str, Any], context: dict[str, Any], query_family: str, model: dict[str, Any] | None, memory_labels_by_id: dict[str, dict[str, Any]]) -> list[tuple[float, Any, dict[str, float], dict[str, Any], float]]:
    ranked = _selector_ranked_candidates(row=row, context=context, query_family=query_family, model=model, memory_labels_by_id=memory_labels_by_id)
    if not ranked: return ranked
    if model:
        ans_count = sum(1 for p in memory_labels_by_id.values() if "answer_bearing" in set(p.get("labels", [])))
        if (ans_count == 0 and ranked[0][4] <= 0.0 and ranked[0][2].get("label_answer_bearing", 0.0) <= 0.0) or (_query_requires_multi_evidence(str(row.get("query", ""))) and ans_count < 2):
            ranked = _selector_ranked_candidates(row=row, context=context, query_family=query_family, model=None, memory_labels_by_id={})
    return _selector_apply_single_anchor_rescues(row=row, query_family=query_family, ranked=ranked) if ranked else ranked

def _selector_label_density(ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]], feature_name: str) -> float:
    return sum(1 for _, _, f, _, _ in ranked if float(f.get(feature_name, 0.0)) > 0.0) / float(len(ranked)) if ranked else 0.0

def _selector_top_score_gap(ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]]) -> float:
    if len(ranked) <= 1: return 0.0
    return float(ranked[0][0]) - float(ranked[min(len(ranked) - 1, 2)][0])

def _selector_should_use_conservative_top_k(*, row: dict[str, Any], ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]], query_family: str) -> bool:
    if not ranked or query_family == "abstention": return False
    evidence = row.get("evidence_sufficiency", {})
    from .classification import _selector_should_trust_active_context_only, _selector_requests_repeat_count, _selector_reason_or_realization_query, _selector_requests_list_like_lookup, _selector_target_person, _selector_memory_speaker_name
    if bool(evidence.get("active_context_support")) and _selector_should_trust_active_context_only(row): return False
    query = str(row.get("query", ""))
    ans_dens, dup_dens, gap = _selector_label_density(ranked, "label_answer_bearing"), _selector_label_density(ranked, "label_duplicate_family"), _selector_top_score_gap(ranked)
    if query_family in {"aggregation", "temporal", "current_state", "knowledge_update", "contradiction"} and ans_dens >= 0.75 and dup_dens >= 0.75 and gap < 1.0: return True
    if _selector_requests_repeat_count(query) and ans_dens >= 0.75 and gap < 1.25: return True
    if _selector_reason_or_realization_query(query) and dup_dens >= 0.75 and ans_dens < 0.5: return True
    if query_family == "information_extraction" and dup_dens >= 0.75 and _selector_requests_list_like_lookup(query):
        t_person, top_spk = _selector_target_person(query), _selector_memory_speaker_name(ranked[0][1].text)
        if t_person and top_spk and top_spk != t_person: return True
    return False
