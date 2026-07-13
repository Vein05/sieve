"""Final candidate selection and budgeting policies."""

from __future__ import annotations
from typing import Any
from .common_import import (
    CandidateView, append_with_budget, shared_result, v2_max_selected, v2_target_token_budget, evenly_spaced_indices
)
from .classification import (
    requested_item_count, _selector_needs_abstention_backup, _selector_target_person, _selector_memory_speaker_name,
    _selector_reason_or_realization_query, _selector_requests_action_set
)
from .ranking import _selector_utility_score

def _selector_target_count(*, row: dict[str, Any], query_family: str, positive_count: int, max_selected: int) -> int:
    req_cnt, q_fam = requested_item_count(str(row.get("query", ""))) or 0, query_family
    if q_fam == "abstention": return 0
    if q_fam in {"ordering", "aggregation", "multi_session"}:
        return min(max_selected, max(1, min(max(2, req_cnt or positive_count or 2), max(positive_count, 1))))
    if q_fam == "knowledge_update": return min(max_selected, 1)
    if q_fam in {"conflict_update", "current_state", "contradiction", "temporal"}:
        return min(max_selected, max(1, min(2 if positive_count > 1 else 1, max(positive_count, 1))))
    return min(max_selected, max(1, min(req_cnt or 1, max(positive_count, 1))))

def _selector_token_budget(*, row: dict[str, Any], context: dict[str, Any], query_family: str) -> int | None:
    return v2_target_token_budget(row, context, query_family=query_family)

def _selector_should_allow_top_fallback(*, row: dict[str, Any], decision_summary: dict[str, Any], query_family: str) -> bool:
    if query_family == "abstention":
        if not bool(row.get("evidence_sufficiency", {}).get("active_context_support")): return False
        return len([s for s in (row.get("active_context") or []) if str(s).strip()]) <= 2 and _selector_needs_abstention_backup(str(row.get("query", "")))
    return not (row.get("evidence_sufficiency", {}).get("active_context_support") or (bool(decision_summary.get("context_sufficient")) and row.get("active_context")))

def _selector_selection_sort_key(candidate: CandidateView, query_family: str) -> tuple[int, int, int, int, int]:
    year, month, day = candidate.date_key
    if candidate.date_key != (0, 0, 0):
        if query_family in {"ordering", "aggregation", "multi_session"}: return (0, year, month, day, candidate.rank)
        if query_family in {"temporal", "current_state", "knowledge_update", "conflict_update", "contradiction"}: return (0, -year, -month, -day, candidate.rank)
    return (1, 0, 0, 0, candidate.rank)

def _selector_chronological_coverage_selection(ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]], *, token_budget: int | None, max_selected: int, target_count: int) -> list[CandidateView]:
    positive = [item for item in ranked if item[4] > 0.0 or item[2]["label_answer_bearing"] > 0.0 or item[2]["label_breaks_sufficiency"] > 0.0 or item[2].get("label_restores_sufficiency", 0.0) > 0.0]
    if not positive or len(positive) < target_count: positive = ranked
    if not positive: return []
    ordered = sorted(positive, key=lambda i: _selector_selection_sort_key(i[1], "ordering"))
    sampled = evenly_spaced_indices(len(ordered), min(len(ordered), target_count))
    selected, used_tokens, seen_ids = [], 0, set()
    for idx in sampled:
        view = ordered[idx][1]
        if view.memory_id not in seen_ids:
            kept, used_tokens = append_with_budget(selected, view, token_budget=token_budget, max_selected=max_selected, used_tokens=used_tokens)
            if kept: seen_ids.add(view.memory_id)
    return selected

def _selector_spread_selection(ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]], *, token_budget: int | None, max_selected: int, target_count: int) -> list[CandidateView]:
    positive = [item for item in ranked if item[4] > 0.0 or item[2]["label_answer_bearing"] > 0.0 or item[2]["label_breaks_sufficiency"] > 0.0 or item[2].get("label_restores_sufficiency", 0.0) > 0.0]
    if not positive or len(positive) < target_count: positive = ranked
    if not positive: return []
    sampled, selected, used_tokens = evenly_spaced_indices(len(positive), min(len(positive), target_count)), [], 0
    for idx in sampled: kept, used_tokens = append_with_budget(selected, positive[idx][1], token_budget=token_budget, max_selected=max_selected, used_tokens=used_tokens)
    return selected

def _selector_top_selection(ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]], *, token_budget: int | None, max_selected: int, target_count: int, query_family: str | None = None) -> list[CandidateView]:
    selected, used_tokens = [], 0
    for _, view, features, _, utility in ranked:
        if len(selected) >= target_count: break
        if query_family != "information_extraction" and (utility <= 0.0 and features["label_answer_bearing"] <= 0.0 and features["label_breaks_sufficiency"] <= 0.0 and features.get("label_restores_sufficiency", 0.0) <= 0.0): continue
        kept, used_tokens = append_with_budget(selected, view, token_budget=token_budget, max_selected=max_selected, used_tokens=used_tokens)
    return selected

def _selector_information_extraction_bundle_selection(ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]], *, row: dict[str, Any], token_budget: int | None, max_selected: int) -> list[CandidateView]:
    if not ranked: return []
    target_person, query = _selector_target_person(str(row.get("query", ""))), str(row.get("query", ""))
    def align_bonus(v: CandidateView) -> float: return 0.75 if target_person and _selector_memory_speaker_name(v.text) == target_person else -0.75 if target_person and _selector_memory_speaker_name(v.text) else 0.0
    from .features import _answer_shape_signals
    def echo_pen(v: CandidateView, f: dict[str, float]) -> float: return 0.5 if "?" in v.text and float(_answer_shape_signals(query=query, text=v.text, features=f)["body_token_count"]) <= 12.0 else 0.0
    def shape_score(v: CandidateView, f: dict[str, float]) -> float:
        s = _answer_shape_signals(query=query, text=v.text, features=f)
        return 1.0*float(s.get("relation_name_hint", 0.0)) + 0.85*float(s.get("support_object_hint", 0.0)) + 0.7*float(s.get("acronym_hint", 0.0)) + 0.45*float(s.get("focus_density", 0.0)) - 0.35*float(s.get("generic_role_only", 0.0)) - 0.01*float(s.get("body_token_count", 0.0))

    anchors = [i for i in ranked[:6] if i[2].get("label_duplicate_family", 0.0) > 0.0]
    is_spk_dup = False
    if target_person:
        spk_anchors = [i for i in anchors if _selector_memory_speaker_name(i[1].text) == target_person]
        if spk_anchors: anchors, is_spk_dup = spk_anchors, len(spk_anchors) > 1
    if not anchors: anchors = ranked[:1]
    top_score, top_view, top_features, _, _ = max(anchors, key=lambda i: (i[0] + 0.35*i[4] + align_bonus(i[1]) - echo_pen(i[1], i[2]), shape_score(i[1], i[2]), -i[1].rank))
    if top_features.get("label_duplicate_family", 0.0) <= 0.0 or (top_features.get("label_answer_bearing", 0.0) <= 0.0 and top_features.get("label_breaks_sufficiency", 0.0) <= 0.0 and not is_spk_dup): return []

    t_overlap, t_rare, t_shape = float(top_features.get("salient_query_overlap", 0.0)), float(top_features.get("rare_query_token_hit", 0.0)), shape_score(top_view, top_features)
    bundle_cands = []
    for score, view, features, _, _ in ranked:
        if features.get("label_duplicate_family", 0.0) <= 0.0: continue
        spk_bonus = align_bonus(view)
        if target_person and spk_bonus < 0.0 and features.get("label_answer_bearing", 0.0) <= 0.0: continue
        ovl = float(features.get("salient_query_overlap", 0.0))
        sh_val = shape_score(view, features)
        if ovl < t_overlap - 0.25 and (ovl < t_overlap - 0.5 or sh_val < t_shape + 0.35): continue
        if float(features.get("rare_query_token_hit", 0.0)) < t_rare or score > top_score + 1e-6: continue
        bundle_cands.append((sh_val + spk_bonus - echo_pen(view, features), score, -view.rank, view))
    bundle_cands.sort(reverse=True)
    if is_spk_dup and bundle_cands:
        top_idx = next((idx for idx, c in enumerate(bundle_cands) if c[3].memory_id == top_view.memory_id), None)
        if top_idx: bundle_cands.insert(0, bundle_cands.pop(top_idx))
    if target_person and bundle_cands:
        spk_cands = [i for i in bundle_cands if _selector_memory_speaker_name(i[3].text) == target_person]
        if spk_cands: bundle_cands = spk_cands
        elif align_bonus(bundle_cands[0][3]) > 0.0: bundle_cands = [i for i in bundle_cands if align_bonus(i[3]) >= 0.0 or i[3] == bundle_cands[0][3]]

    selected, used_tokens = [], 0
    for _, _, _, view in bundle_cands:
        if len(selected) >= min(max_selected, 3): break
        kept, used_tokens = append_with_budget(selected, view, token_budget=token_budget, max_selected=max_selected, used_tokens=used_tokens)
    return selected if len(selected) > 1 else []

def _selector_policy_select(*, row: dict[str, Any], context: dict[str, Any], ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]], query_family: str) -> tuple[list[CandidateView], dict[str, Any]]:
    m_sel = v2_max_selected(row, context, query_family=query_family)
    t_budget = _selector_token_budget(row=row, context=context, query_family=query_family)
    thresh = float(context["v2_label_selector_v1_model"]["threshold"]) if ranked and "v2_label_selector_v1_model" in context else 0.0
    margin = round((ranked[0][0] if ranked else 0.0) - thresh, 3)
    pos_cnt = sum(1 for i in ranked if i[4] > 0.0 or i[2]["label_answer_bearing"] > 0.0 or i[2]["label_breaks_sufficiency"] > 0.0 or i[2].get("label_restores_sufficiency", 0.0) > 0.0)
    t_cnt, reason_codes, mode = _selector_target_count(row=row, query_family=query_family, positive_count=pos_cnt, max_selected=m_sel), [f"QUERY_FAMILY_{query_family.upper()}"], "single_anchor"

    if query_family == "abstention":
        if str(row.get("evidence_sufficiency", {}).get("label", "")).strip().lower() == "abstention" or not ranked or (ranked[0][0] if ranked else 0.0) < max(thresh, 0.15):
            return [], {"query_family": query_family, "selection_mode": "abstain", "selection_margin": margin, "anchor_count": 0, "reason_codes": reason_codes + ["ABSTENTION_GATE"]}
        reason_codes, mode = reason_codes + ["ABSTENTION_OVERRIDE"], "abstain_fallback"

    if query_family in {"ordering", "aggregation", "multi_session"}:
        sel = _selector_spread_selection(ranked, token_budget=t_budget, max_selected=m_sel, target_count=t_cnt)
        if sel: return sel, {"query_family": query_family, "selection_mode": "coverage_spread", "selection_margin": margin, "anchor_count": len(sel), "reason_codes": reason_codes + ["COVERAGE_ROUTER"]}

    if query_family in {"contradiction", "temporal"} and t_cnt > 1:
        sel = _selector_chronological_coverage_selection(ranked, token_budget=t_budget, max_selected=m_sel, target_count=t_cnt)
        if sel: return sel, {"query_family": query_family, "selection_mode": "contradiction_pair" if query_family == "contradiction" else "temporal_pair", "selection_margin": margin, "anchor_count": len(sel), "reason_codes": reason_codes + ["CONTRADICTION_PAIR" if query_family == "contradiction" else "TEMPORAL_PAIR"]}

    if query_family in {"conflict_update", "current_state", "knowledge_update", "contradiction", "temporal"}:
        sel = _selector_top_selection(ranked, token_budget=t_budget, max_selected=m_sel, target_count=t_cnt, query_family=query_family)
        if sel:
            codes = list(reason_codes)
            if query_family in {"conflict_update", "current_state"} and len(sel) > 1: codes.append("CONFLICT_PAIR")
            if query_family == "temporal" and len(sel) > 1: codes.append("TEMPORAL_HISTORY")
            return sel, {"query_family": query_family, "selection_mode": "update_resolution", "selection_margin": margin, "anchor_count": len(sel), "reason_codes": codes}

    if query_family == "information_extraction":
        sel = _selector_information_extraction_bundle_selection(ranked, row=row, token_budget=t_budget, max_selected=m_sel)
        if sel: return sel, {"query_family": query_family, "selection_mode": "extractive_bundle", "selection_margin": margin, "anchor_count": len(sel), "reason_codes": reason_codes + ["DUPLICATE_FAMILY_BUNDLE"]}

    sel = _selector_top_selection(ranked, token_budget=t_budget, max_selected=m_sel, target_count=max(1, t_cnt), query_family=query_family)
    if not sel: reason_codes, mode = reason_codes + ["NO_SELECTION"], "abstain"
    return sel, {"query_family": query_family, "selection_mode": mode, "selection_margin": margin, "anchor_count": len(sel), "reason_codes": reason_codes}

def _selector_apply_top_fallback(*, row: dict[str, Any], decision_summary: dict[str, Any], query_family: str, ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]], selected: list[CandidateView], selection_meta: dict[str, Any]) -> tuple[list[CandidateView], dict[str, Any]]:
    if selected or not ranked or not _selector_should_allow_top_fallback(row=row, decision_summary=decision_summary, query_family=query_family): return selected, selection_meta
    query = str(row.get("query", ""))
    raw_ranked = sorted(ranked, key=lambda item: (item[1].rank, item[1].memory_id))
    if query_family == "abstention":
        pos = [i for i in raw_ranked if i[4] > 0.0 or i[2]["label_answer_bearing"] > 0.0 or i[2]["label_breaks_sufficiency"] > 0.0 or i[2].get("label_restores_sufficiency", 0.0) > 0.0] or raw_ranked
        if _selector_requests_action_set(query):
            b_sel = []
            for _, v, *_ in pos:
                if v not in b_sel: b_sel.append(v)
                if len(b_sel) >= 2: break
            if b_sel: return b_sel, {**selection_meta, "selection_mode": "fallback_backup_set", "reason_codes": list(selection_meta.get("reason_codes", [])) + ["TOP_FALLBACK"]}
        if pos and (str(query).strip().lower().startswith("who ") or _selector_reason_or_realization_query(query)):
            from .features import _answer_shape_signals
            def p_score(i):
                s = _answer_shape_signals(query=query, text=i[1].text, features=i[2])
                sc = i[4] + 1.5*float(i[2].get("label_answer_bearing", 0.0)) + 1.2*float(s.get("relation_name_hint", 0.0)) + 1.0*float(s.get("support_object_hint", 0.0)) + 0.4*float(s.get("focus_density", 0.0)) - 0.3*float(s.get("generic_role_only", 0.0))
                if "realize" in query.lower(): sc += 1.0 if "realize" in i[1].text.lower() or "realized" in i[1].text.lower() else -0.4
                return (sc, -i[1].rank)
            return [max(pos[:4], key=p_score)[1]], {**selection_meta, "selection_mode": "fallback_top_anchor", "reason_codes": list(selection_meta.get("reason_codes", [])) + ["TOP_FALLBACK"]}
    top_score, top_view, _, top_labels, utility = raw_ranked[0]
    if top_score < -0.05 and utility <= 0.0 and not top_labels.get("labels"): return selected, selection_meta
    return [top_view], {**selection_meta, "selection_mode": "fallback_top_anchor", "reason_codes": list(selection_meta.get("reason_codes", [])) + ["TOP_FALLBACK"]}

def _selector_anchor_needs_support(*, row: dict[str, Any], query_family: str, selection_mode: str | None, anchor: CandidateView, anchor_features: dict[str, float]) -> bool:
    if query_family != "information_extraction" or selection_mode not in {"single_anchor", "fallback_top_anchor"}: return False
    query = str(row.get("query", "")).strip().lower()
    if not query or query.startswith("why ") or "realize" in query or "realized" in query: return False
    from .classification import _selector_explicit_recall_lookup, _selector_recall_lookup_is_fragile
    from .features import _answer_shape_signals
    recall_lookup, sigs = _selector_explicit_recall_lookup(query), _answer_shape_signals(query=str(row.get("query", "")), text=anchor.text, features=anchor_features)
    if recall_lookup and _selector_recall_lookup_is_fragile(query): return True
    if anchor.rank >= 4: return recall_lookup
    if anchor.rank >= 2 and query.startswith(("what color ", "which color ", "what was the color", "what name ", "which name ")): return recall_lookup
    if anchor.rank >= 3 and float(sigs.get("body_token_count", 0.0)) <= 12.0: return recall_lookup
    return False

def _selector_support_candidate_score(*, row: dict[str, Any], anchor: CandidateView, candidate: CandidateView, features: dict[str, float]) -> float:
    from .features import _answer_shape_signals
    sigs = _answer_shape_signals(query=str(row.get("query", "")), text=candidate.text, features=features)
    q_like = 1.0 if "?" in candidate.text else 0.0
    sc = 0.9*float(features.get("salient_query_overlap", 0.0)) + 0.25*float(features.get("salient_query_coverage", 0.0)) + 0.2*float(features.get("rare_query_token_hit", 0.0))
    sc += 0.25 if candidate.rank <= 2 else 0.0
    sc += 0.15 if candidate.rank <= 4 else 0.0
    sc += 0.15*float(sigs.get("definition_list_hint", 0.0)) - 0.35*q_like - 0.15 if candidate.memory_id == anchor.memory_id else 0.0
    sc -= 0.0025*float(candidate.token_count)
    return sc

def _selector_add_anchor_support(*, row: dict[str, Any], context: dict[str, Any], query_family: str, ranked: list[tuple[float, Any, dict[str, float], dict[str, Any], float]], selected: list[CandidateView], selection_meta: dict[str, Any]) -> tuple[list[CandidateView], dict[str, Any]]:
    if len(selected) != 1 or not ranked: return selected, selection_meta
    anchor = selected[0]
    anchor_payload = next((i for i in ranked if i[1].memory_id == anchor.memory_id), None)
    if not anchor_payload or not _selector_anchor_needs_support(row=row, query_family=query_family, selection_mode=str(selection_meta.get("selection_mode") or ""), anchor=anchor, anchor_features=anchor_payload[2]): return selected, selection_meta
    t_bud, m_sel = _selector_token_budget(row=row, context=context, query_family=query_family), v2_max_selected(row, context, query_family=query_family)
    if m_sel <= 1: return selected, selection_meta
    used_tokens, supp_cands = sum(c.token_count for c in selected), []
    for _, view, features, labels, _ in ranked:
        if view.memory_id == anchor.memory_id or float(features.get("rare_query_token_hit", 0.0)) <= 0.0 or float(features.get("salient_query_overlap", 0.0)) <= 0.0 or "distractor" in set(labels.get("labels", [])): continue
        supp_cands.append((_selector_support_candidate_score(row=row, anchor=anchor, candidate=view, features=features), view.rank, view))
    supp_cands.sort(key=lambda i: (i[0], -i[1]), reverse=True)
    for score, _, view in supp_cands:
        if score < 0.45: break
        aug = list(selected)
        kept, _ = append_with_budget(aug, view, token_budget=t_bud, max_selected=m_sel, used_tokens=used_tokens)
        if kept: return aug, {**selection_meta, "selection_mode": "anchor_plus_support", "anchor_count": len(aug), "reason_codes": list(selection_meta.get("reason_codes", [])) + ["SUPPORT_MEMORY"]}
    return selected, selection_meta

def _selector_finalize_selection_order(*, selected: list[CandidateView], query_family: str, selection_mode: str | None) -> list[CandidateView]:
    if not selected: return selected
    if query_family in {"ordering", "aggregation", "multi_session"} or selection_mode in {"temporal_pair", "contradiction_pair"}: return sorted(selected, key=lambda c: _selector_selection_sort_key(c, "ordering"))
    if query_family in {"temporal", "current_state", "knowledge_update", "conflict_update", "contradiction"}: return sorted(selected, key=lambda c: _selector_selection_sort_key(c, query_family))
    return selected
