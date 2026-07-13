"""Heuristics, post-processing, and manual rescue logic."""

from typing import Any
from .features import (
    _memory_recall_query, _memory_recall_target_role, _extractive_recall_query,
    _update_intent_query, _repeat_count_query, _candidate_memory_text,
    _answer_shape_signals, _usefulness_query_is_binary_verification,
    _usefulness_query_requests_temporal_pair
)
from .common_import import (
    requested_item_count, is_event_ordering_query,
    cached_word_tokenize, cached_pos_tag, stem_token
)

def _stem_token(token: str) -> str:
    return stem_token(token)

def _support_rescue_score(*, candidate: dict[str, Any], features: dict[str, float]) -> float:
    text = _candidate_memory_text(candidate)
    question_like = 1.0 if "?" in text else 0.0
    return (
        1.25 * float(features.get("salient_query_overlap", 0.0)) + 1.0 * float(features.get("necessity", 0.0))
        + 0.5 * float(features.get("retrieval_prior", 0.0)) + 0.35 * float(features.get("top_ranked_candidate", 0.0))
        + 0.3 * float(features.get("salient_query_coverage", 0.0)) + 0.5 * float(features.get("is_personal", 0.0))
        + 0.35 * float(features.get("user_memory", 0.0)) + 0.2 * float(features.get("name_overlap", 0.0))
        + 0.15 * float(features.get("rare_query_token_hit", 0.0)) - 0.35 * float(features.get("assistant_memory", 0.0))
        - 0.3 * question_like + 0.05 * float(candidate.get("score", 0.0))
    )

def _support_rescue_answer_shape_score(payload: dict[str, Any]) -> float:
    features = payload.get("features", {})
    ans = _answer_shape_signals(query=str(payload.get("query", "")), text=str(payload.get("text", "")), features=features)
    return (
        0.7 * float(features.get("is_personal", 0.0)) + 0.55 * float(features.get("user_memory", 0.0))
        - 0.35 * float(features.get("assistant_memory", 0.0)) - 0.8 * float(payload.get("question_like", 0.0))
        + 1.0 * float(ans.get("relation_name_hint", 0.0)) + 0.85 * float(ans.get("support_object_hint", 0.0))
        + 0.7 * float(ans.get("acronym_hint", 0.0)) + 1.2 * float(ans.get("definition_list_hint", 0.0))
        + 0.5 * float(ans.get("focus_density", 0.0)) - 0.45 * float(ans.get("generic_role_only", 0.0))
        - 0.8 * float(ans.get("advice_style_penalty", 0.0)) - 0.012 * float(ans.get("body_token_count", 0.0))
    )

def _extractive_recall_role_bonus(*, query: str, payload: dict[str, Any]) -> float:
    role = _memory_recall_target_role(query)
    f = payload.get("features", {})
    a, u, m, p = float(f.get("assistant_memory", 0.0)), float(f.get("user_memory", 0.0)), float(f.get("mixed_speaker_memory", 0.0)), float(f.get("is_personal", 0.0))
    if role == "assistant": return 1.2 * a - 0.9 * u - 0.35 * m
    if role == "self": return 1.25 * u + 0.6 * p - 0.9 * a - 0.25 * m
    return 0.0

def _extractive_recall_rescue_memory_ids(*, row: dict[str, Any], decision_summary: dict[str, Any], candidate_payloads: list[dict[str, Any]]) -> list[str]:
    query = str(row.get("query", ""))
    lowered = query.strip().lower()
    role = _memory_recall_target_role(query)
    if not _memory_recall_query(query) or not _extractive_recall_query(query) or role == "neutral": return []
    if decision_summary.get("current_state_query") or _update_intent_query(query) or decision_summary.get("query_type") == "temporal" or _repeat_count_query(query): return []
    eligible = [p for p in candidate_payloads if float(p.get("question_like", 0.0)) <= 0.0]
    if not eligible: return []
    best_support = max(float(p.get("support_score", float("-inf"))) for p in eligible)
    margin = 4.0 if role == "assistant" or "remind me" in lowered else 2.5
    shortlisted = [p for p in eligible if float(p.get("support_score", float("-inf"))) >= best_support - margin]
    if not shortlisted: return []
    def d_score(p: dict[str, Any]) -> float: return _support_rescue_answer_shape_score(p) + _extractive_recall_role_bonus(query=query, payload=p)
    best = max(shortlisted, key=lambda p: (d_score(p), float(p.get("support_score", 0.0)), -int(p.get("rank", 1))))
    return [str(best["memory_id"])] if d_score(best) > 0.0 else []

def _support_rescue_memory_ids(*, row: dict[str, Any], decision_summary: dict[str, Any], candidate_payloads: list[dict[str, Any]]) -> list[str]:
    query = str(row.get("query", ""))
    repeat_q = _repeat_count_query(query)
    if not _memory_recall_query(query) or decision_summary.get("current_state_query") or _update_intent_query(query) or (decision_summary.get("query_type") == "temporal" and not repeat_q) or not candidate_payloads: return []
    ranked = sorted(candidate_payloads, key=lambda item: (item["support_score"], -item["rank"]), reverse=True)
    best, top_ret = ranked[0], min(candidate_payloads, key=lambda item: item["rank"])
    pref = best
    if not repeat_q and top_ret["support_score"] >= best["support_score"] - 0.5 and (top_ret["features"].get("rare_query_token_hit", 0.0) > 0.0 or top_ret["features"].get("salient_query_overlap", 0.0) >= best["features"].get("salient_query_overlap", 0.0) - 0.5):
        pref = top_ret
    if pref["support_score"] < 1.6: return []
    if not repeat_q:
        margin = 0.85 if query.strip().lower().startswith(("who ", "which ", "where ", "what type ", "what kind ", "what state ", "what city ", "what country ")) else 0.4
        ans_candidates = [p for p in ranked if p["support_score"] >= pref["support_score"] - margin]
        if ans_candidates:
            best_ans = max(ans_candidates, key=lambda p: (_support_rescue_answer_shape_score(p), p["support_score"], -p["rank"]))
            if _support_rescue_answer_shape_score(best_ans) >= (_support_rescue_answer_shape_score(pref) + 0.3): pref = best_ans
    ids = [pref["memory_id"]]
    if repeat_q or (requested_item_count(query) or 0) > 1:
        floor = pref["support_score"] - 0.45
        for p in ranked[1:]:
            if p["memory_id"] not in ids and p["support_score"] >= floor and p["features"].get("rare_query_token_hit", 0.0) > 0.0:
                ids.append(p["memory_id"])
                if len(ids) >= 3: break
    return ids

def _direct_answer_rescue_memory_ids(*, row: dict[str, Any], candidate_payloads: list[dict[str, Any]]) -> list[str]:
    query = str(row.get("query", ""))
    lowered = query.strip().lower()
    if not lowered: return []
    tokens = cached_word_tokenize(lowered)
    tags = cached_pos_tag(tuple(tokens))
    words = set(tokens)
    
    why = any(w == "why" and t == "WRB" for w, t in tags)
    realize = "realize" in words or "realized" in words
    action = ("what" in words or "how" in words) and ("do" in words or "done" in words)
    
    if not (why or realize or action): return []
    if not eligible: return []
    def d_score(p: dict[str, Any]) -> float:
        s = _support_rescue_answer_shape_score(p)
        lt = str(p.get("text", "")).lower()
        if why: s += 1.8 if any(m in lt for m in ("because ", "'cause", "cause ", "reason")) else -1.2
        if realize: s += 2.0 if ("realize" in lt or "realized" in lt) else -1.5
        if action:
            s += 0.35 * float(p.get("features", {}).get("salient_query_overlap", 0.0))
            # Use cached tokens for body count
            body_tokens = cached_word_tokenize(lt)
            s += 0.2 if len(body_tokens) <= 32 else -0.2
        return s
    best_s = max(float(p.get("support_score", float("-inf"))) for p in eligible)
    if action:
        sh = sorted([p for p in eligible if float(p.get("support_score", float("-inf"))) >= best_s - 1.0], key=lambda p: (float(p.get("support_score", 0.0)), d_score(p), -int(p.get("rank", 1))), reverse=True)
        return [str(p["memory_id"]) for p in sh[:2]]
    margin = 4.0 if why else 2.5
    sh = [p for p in eligible if float(p.get("support_score", float("-inf"))) >= best_s - margin]
    if not sh: return []
    best = max(sh, key=lambda p: (d_score(p), float(p.get("support_score", 0.0)), -int(p.get("rank", 1))))
    return [str(best["memory_id"])] if d_score(best) > 0.0 else []

def _postprocess_predicted_memory_usefulness_labels(*, row: dict[str, Any], decision_summary: dict[str, Any], labels_by_memory: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    query = str(row.get("query", ""))
    req_count, active_ctx_sup = requested_item_count(query) or 0, bool(row.get("evidence_sufficiency", {}).get("active_context_support"))
    bin_q, temp_q = _usefulness_query_is_binary_verification(query), _usefulness_query_requests_temporal_pair(query)
    special_ans = bool(decision_summary.get("current_state_query") or _update_intent_query(query) or bin_q or temp_q or decision_summary.get("query_type") == "temporal")
    multi_anchor = bool(is_event_ordering_query(row) or temp_q or req_count > 1 or decision_summary.get("query_type") == "temporal" or bin_q)
    norm, ans_ids = {}, []
    for mid, payload in labels_by_memory.items():
        labels = set(str(label) for label in payload.get("labels", []))
        if special_ans and labels & {"conflict_resolving", "newer_update"}: labels.add("answer_bearing")
        if "distractor" in labels and not labels & {"answer_bearing", "conflict_resolving", "newer_update"}: labels.add("useless_under_budget")
        if not labels: labels.add("useless_under_budget")
        next_p = {"labels": sorted(labels), "leave_one_out_effect": "breaks_sufficiency" if payload.get("leave_one_out_effect") == "breaks_sufficiency" else "no_change", "add_one_rescue_effect": "restores_sufficiency" if payload.get("add_one_rescue_effect") == "restores_sufficiency" else "no_change"}
        if payload.get("uncertainty") == "interaction_dependent": next_p["uncertainty"] = "interaction_dependent"
        norm[mid] = next_p
        if "answer_bearing" in labels: ans_ids.append(mid)
    if len(ans_ids) == 1 and not active_ctx_sup:
        norm[ans_ids[0]]["leave_one_out_effect"], norm[ans_ids[0]]["add_one_rescue_effect"] = "breaks_sufficiency", "restores_sufficiency"
    if len(ans_ids) > 1 and multi_anchor:
        for mid in ans_ids: norm[mid]["add_one_rescue_effect"], norm[mid]["uncertainty"] = "restores_sufficiency", "interaction_dependent"
    return norm

def _apply_support_rescue(*, row: dict[str, Any], decision_summary: dict[str, Any], labels_by_memory: dict[str, dict[str, Any]], candidate_payloads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    query = str(row.get("query", ""))
    if decision_summary.get("current_state_query") or _update_intent_query(query): return labels_by_memory
    
    e_rescue = _extractive_recall_rescue_memory_ids(row=row, decision_summary=decision_summary, candidate_payloads=candidate_payloads)
    s_rescue = _support_rescue_memory_ids(row=row, decision_summary=decision_summary, candidate_payloads=candidate_payloads)
    d_rescue = _direct_answer_rescue_memory_ids(row=row, candidate_payloads=candidate_payloads)
    
    final_rescue = list(dict.fromkeys([*e_rescue, *d_rescue])) if e_rescue else list(dict.fromkeys(d_rescue))
    if not final_rescue: final_rescue = s_rescue
    if not final_rescue: return labels_by_memory
    
    sup_by_id = {p["memory_id"]: p["support_score"] for p in candidate_payloads}
    feat_by_id = {p["memory_id"]: p["features"] for p in candidate_payloads}
    existing_ans = {mid for mid, p in labels_by_memory.items() if "answer_bearing" in set(p.get("labels", []))}
    best_ex = max((sup_by_id.get(mid, float("-inf")) for mid in existing_ans), default=float("-inf"))
    res_best = max(sup_by_id.get(mid, float("-inf")) for mid in final_rescue)
    top_res = min(final_rescue, key=lambda mid: next(p["rank"] for p in candidate_payloads if p["memory_id"] == mid))
    
    if set(final_rescue) == existing_ans: return labels_by_memory
    if best_ex != float("-inf"):
        if d_rescue and set(final_rescue) != existing_ans: pass
        elif res_best > best_ex + 0.35: pass
        elif len(final_rescue) < len(existing_ans) and not _repeat_count_query(query): pass
        elif set(final_rescue) != existing_ans and all(feat_by_id.get(mid, {}).get("rare_query_token_hit", 0.0) <= 0.0 for mid in existing_ans - set(final_rescue)) and all(feat_by_id.get(mid, {}).get("rare_query_token_hit", 0.0) > 0.0 for mid in final_rescue): pass
        elif top_res not in existing_ans and res_best >= best_ex - 0.35: pass
        else: return labels_by_memory

    rescued = {mid: {"labels": list(p.get("labels", [])), "leave_one_out_effect": p.get("leave_one_out_effect", "no_change"), "add_one_rescue_effect": p.get("add_one_rescue_effect", "no_change"), **({"uncertainty": p["uncertainty"]} if p.get("uncertainty") else {})} for mid, p in labels_by_memory.items()}
    res_set, multi = set(final_rescue), len(final_rescue) > 1
    for mid, payload in rescued.items():
        labels = set(str(label) for label in payload.get("labels", []))
        if mid in res_set:
            labels -= {"distractor", "useless_under_budget"}
            labels.add("answer_bearing")
            payload.update({"leave_one_out_effect": "breaks_sufficiency" if not multi else "no_change", "add_one_rescue_effect": "restores_sufficiency"})
            if multi: payload["uncertainty"] = "interaction_dependent"
        else:
            labels -= {"answer_bearing", "conflict_resolving", "newer_update"}
            payload.update({"leave_one_out_effect": "no_change", "add_one_rescue_effect": "no_change"})
            payload.pop("uncertainty", None)
        payload["labels"] = sorted(labels)
    return rescued
