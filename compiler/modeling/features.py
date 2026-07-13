"""Feature extraction and query intent detection logic."""

import re
from typing import Any

from .common_import import cached_word_tokenize, cached_pos_tag, stem_token

def _stem_token(token: str) -> str:
    return stem_token(token)

from controller.models import TextProfile
from .common_import import is_event_ordering_query, requested_item_count

_LEXICAL_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LEXICAL_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "what", "when", "where",
    "which", "have", "has", "had", "your", "about", "only", "item", "items", "there",
    "were", "them", "they", "into", "last", "over", "again", "after", "before",
    "many", "much", "more", "than", "will",
}

_ANSWER_SHAPE_RELATION_NAME_RE = re.compile(
    r"\b(?:at|from|to|with|named|called|colleague|friend|coworker|co-worker|partner|neighbor|roommate|mentor|teacher|brother|sister|mom|mother|dad|father|wife|husband|aunt|uncle|grandma|grandpa|relative|nephew|niece)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)
_ANSWER_SHAPE_SUPPORT_OBJECT_RE = re.compile(
    r"\bhelp(?:s|ed|ing)?\s+([a-z0-9+ -]{0,40})(?:people|person|individuals?|folks|kids|children|famil(?:y|ies)|community)\b"
)
_ANSWER_SHAPE_GENERIC_ROLE_RE = re.compile(
    r"\b(?:colleague|friend|coworker|co-worker|partner|neighbor|roommate|mentor|teacher|brother|sister|mom|mother|dad|father|wife|husband|aunt|uncle|grandma|grandpa|relative|nephew|niece)\b"
)
_ANSWER_SHAPE_LEADING_TERM_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9' /&()+-]{1,48}\s*(?:-|:)\s+")
_ANSWER_SHAPE_DEFINITION_LIST_RE = re.compile(
    r"(?:^|[.]\s+)[A-Za-z][A-Za-z0-9' /&()+-]{1,48}\s+-\s+[A-Za-z]"
)
_ANSWER_SHAPE_ADVICE_OPENING_RE = re.compile(
    r"^\s*(?:if you|you may|you might|consider|try to|remember to|don't|do not|be sure to|it's important to)\b"
)

def _lexical_content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _LEXICAL_TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in _LEXICAL_STOPWORDS
    }

def _usefulness_query_is_binary_verification(query: str) -> bool:
    tokens = cached_word_tokenize(str(query or "").lower())
    tags = cached_pos_tag(tuple(tokens))
    if not tags: return False
    # Check if starts with MD (modal), VB* (verb), or specific auxiliaries
    first_word, first_tag = tags[0]
    return first_tag.startswith(("MD", "VB")) or first_word in ("is", "are", "am", "was", "were", "has", "have", "had", "do", "does", "did", "can", "will")

def _usefulness_query_requests_temporal_pair(query: str) -> bool:
    tokens = cached_word_tokenize(str(query or "").lower())
    words = set(tokens)
    # Check for interval indicators
    if "between" in words or ("how" in words and "long" in words):
        if words & {"days", "weeks", "months", "years", "time", "duration"}:
            return True
    return False

def _memory_recall_query(query: str) -> bool:
    tokens = cached_word_tokenize(str(query or "").lower())
    words = set(tokens)
    if "remind" in words or "previous" in words or "last" in words:
        return True
    # Starts with wh-question
    tags = cached_pos_tag(tuple(tokens))
    if tags and tags[0][1] in ("WDT", "WP", "WRB"):
        return True
    return False

def _repeat_count_query(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in ("how many times", "multiple times", "more than once", "again and again", "repeatedly"))

def _update_intent_query(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in ("current ", "currently ", "right now", "latest ", "newest ", "updated ", "update ", "still ", "now "))

def _update_sensitive_query(*, query: str, decision_summary: dict[str, Any]) -> float:
    if bool(decision_summary.get("current_state_query")):
        return 1.0
    lowered = query.strip().lower()
    if not lowered:
        return 0.0
    return 1.0 if _update_intent_query(lowered) else 0.0

def _memory_recall_target_role(query: str) -> str:
    tokens = cached_word_tokenize(str(query or "").lower())
    words = set(tokens)
    tags = cached_pos_tag(tuple(tokens))
    
    # Assistant role: "you mentioned", "you said"
    if "you" in words:
        if words & {"mentioned", "said", "suggested", "told", "referred"}:
            return "assistant"
            
    # Self role: "did I say", "my preference"
    if "my" in words or "i" in words:
        # Check for first-person subjects or possessives
        if any(w == "i" and t == "PRP" for w, t in tags):
            return "self"
        if any(w == "my" and t == "PRP$" for w, t in tags):
            return "self"
            
    return "neutral"

def _extractive_recall_query(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered:
        return False
    if "remind me" in lowered:
        return True
    return lowered.startswith(("what ", "which ", "who ", "where ", "when "))

def _candidate_memory_text(candidate: dict[str, Any]) -> str:
    return str(candidate.get("text") or candidate.get("content") or candidate.get("memory_text") or candidate.get("rendered") or candidate.get("snippet") or "")

def _structured_update_markers(text: str) -> dict[str, float]:
    lowered = text.lower()
    policy_update = 1.0 if (
        "memory_policy_update" in lowered
        or "status=forget_prior_memory" in lowered
        or "forget_prior_memory" in lowered
    ) else 0.0
    stale_memory = 1.0 if (
        "earlier_user_profile" in lowered
        or "prior_memory=" in lowered and "status=forget_prior_memory" not in lowered
    ) else 0.0
    forget_marker = 1.0 if any(marker in lowered for marker in ("forget", "ignored in future", "no longer")) else 0.0
    return {
        "policy_update_marker": policy_update,
        "stale_memory_marker": stale_memory,
        "forget_marker": forget_marker,
    }

def _memory_body_text(text: str) -> str:
    body = str(text).split("|")[-1].strip()
    if ":" not in body:
        return body
    speaker, remainder = body.split(":", 1)
    if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{0,31}", speaker.strip()):
        return remainder.strip()
    return body

def _learning_features(*, row: dict[str, Any], candidate: dict[str, Any], decision_summary: dict[str, Any], profile: TextProfile, query_profile: TextProfile) -> dict[str, float]:
    retrieval_prior = float(candidate["signals"]["retrieval_prior"])
    necessity = float(candidate["signals"]["necessity"])
    freshness = float(candidate["signals"]["freshness"])
    contradiction_risk = float(candidate["signals"]["contradiction_risk"])
    redundancy_risk = float(candidate["signals"]["redundancy_or_unnecessary_personalization"])
    query = str(row.get("query", ""))
    current_state_query = 1.0 if decision_summary.get("current_state_query") else 0.0
    temporal_query = 1.0 if decision_summary.get("query_type") == "temporal" else 0.0
    context_sufficient = 1.0 if decision_summary.get("context_sufficient") else 0.0
    update_sensitive_query = _update_sensitive_query(query=query, decision_summary=decision_summary)
    name_overlap = 1.0 if query_profile.named_tokens & profile.named_tokens else 0.0
    structured = _structured_update_markers(_candidate_memory_text(candidate))
    gated_policy_update = structured["policy_update_marker"] * update_sensitive_query
    gated_stale_memory = structured["stale_memory_marker"] * update_sensitive_query
    gated_forget_marker = structured["forget_marker"] * update_sensitive_query
    return {
        "bias": 1.0, "retrieval_prior": retrieval_prior, "necessity": necessity, "freshness": freshness,
        "contradiction_safe": 1.0 - contradiction_risk, "redundancy_safe": 1.0 - redundancy_risk,
        "has_update_marker": 1.0 if profile.has_update_marker else 0.0, "has_time_marker": 1.0 if profile.has_time_marker else 0.0,
        "is_preference": 1.0 if profile.is_preference else 0.0, "is_sensitive": 1.0 if profile.is_sensitive else 0.0, "is_personal": 1.0 if profile.is_personal else 0.0,
        "current_state_query": current_state_query, "temporal_query": temporal_query, "context_sufficient": context_sufficient, "name_overlap": name_overlap,
        "update_for_current_state": (1.0 if profile.has_update_marker else 0.0) * current_state_query,
        "time_for_temporal": (1.0 if profile.has_time_marker else 0.0) * temporal_query,
        "sensitive_without_necessity": (1.0 if profile.is_sensitive else 0.0) * max(0.0, 0.5 - necessity),
        "policy_update_marker": gated_policy_update,
        "stale_memory_marker": gated_stale_memory,
        "forget_marker": gated_forget_marker,
        "policy_update_for_current_state": gated_policy_update * current_state_query,
        "stale_memory_for_current_state": gated_stale_memory * current_state_query,
    }

def _answer_shape_signals(*, query: str, text: str, features: dict[str, float] | None = None) -> dict[str, float]:
    body = _memory_body_text(text)
    lowered_query = query.strip().lower()
    
    # Heuristic: limit expensive regexes on extremely large bodies if they aren't likely to hit
    # or just perform them once. 
    # For extraction, we only really care if they exist (except for definition list).
    
    query_tokens = _lexical_content_tokens(query)
    body_tokens = _lexical_content_tokens(body)
    
    # Optimization: Use search instead of findall for single-hint checks
    relation_name_hits = len(_ANSWER_SHAPE_RELATION_NAME_RE.findall(body)) if len(body) < 10000 else (1 if _ANSWER_SHAPE_RELATION_NAME_RE.search(body) else 0)
    
    acronym_hint = 0.0
    if len(body) < 20000:
        acronym_tokens = [token for token in re.findall(r"\b[A-Z]{2,}(?:\+)?\b", body) if token.lower().rstrip("+") not in query_tokens]
        acronym_hint = float(bool(acronym_tokens))
    else:
        # Faster check for large bodies: just see if there's ANY likely acronym
        acronym_hint = 1.0 if re.search(r"\b[A-Z]{2,}(?:\+)?\b", body) else 0.0

    support_object_hint = 1.0 if "support" in lowered_query and _ANSWER_SHAPE_SUPPORT_OBJECT_RE.search(body.lower()) else 0.0
    generic_role_only = 1.0 if lowered_query.startswith("who ") and relation_name_hits == 0 and _ANSWER_SHAPE_GENERIC_ROLE_RE.search(body.lower()) else 0.0
    leading_term_hint = 1.0 if _ANSWER_SHAPE_LEADING_TERM_RE.search(body) else 0.0
    definition_list_hint = 1.0 if len(_ANSWER_SHAPE_DEFINITION_LIST_RE.findall(body)) >= 2 else 0.0
    advice_style_penalty = 1.0 if _ANSWER_SHAPE_ADVICE_OPENING_RE.search(body.lower()) else 0.0
    
    focus_density = 0.0
    if features is not None:
        focus_density = float(features.get("salient_query_overlap", 0.0)) / float(max(len(body_tokens), 1))
        
    return {
        "relation_name_hint": float(relation_name_hits > 0), "acronym_hint": acronym_hint,
        "support_object_hint": support_object_hint, "generic_role_only": generic_role_only,
        "leading_term_hint": leading_term_hint, "definition_list_hint": definition_list_hint,
        "advice_style_penalty": advice_style_penalty, "body_token_count": float(len(body_tokens)), "focus_density": focus_density,
    }

def _query_token_weights(query_tokens: set[str], candidate_token_sets: list[set[str]]) -> tuple[dict[str, float], float]:
    token_weights: dict[str, float] = {}
    total_weight = 0.0
    for token in sorted(query_tokens):
        df = sum(1 for ts in candidate_token_sets if token in ts)
        weight = 1.0 + max(0.0, len(candidate_token_sets) - df) / float(max(len(candidate_token_sets), 1))
        token_weights[token] = weight
        total_weight += weight
    return token_weights, total_weight

def _salient_query_overlap_features(*, query_tokens: set[str], candidate_tokens: set[str], token_weights: dict[str, float], total_weight: float) -> tuple[float, float, float]:
    if not query_tokens: return 0.0, 0.0, 0.0
    overlap_weight = sum(token_weights[token] for token in sorted(query_tokens & candidate_tokens))
    rare_token_hit = 1.0 if any(token_weights[token] >= 1.5 for token in (query_tokens & candidate_tokens)) else 0.0
    return overlap_weight, overlap_weight / float(total_weight or 1.0), rare_token_hit

def _context_lexical_overlap(*, candidate_tokens: set[str], context_token_sets: list[set[str]]) -> float:
    if not context_token_sets or not candidate_tokens:
        return 0.0
    best_overlap = 0.0
    for chunk_tokens in context_token_sets:
        if not chunk_tokens:
            continue
        overlap = len(candidate_tokens & chunk_tokens) / float(len(candidate_tokens))
        best_overlap = max(best_overlap, overlap)
    return best_overlap

def _memory_usefulness_features(*, row: dict[str, Any], candidate: dict[str, Any], decision_summary: dict[str, Any], profile: TextProfile, query_profile: TextProfile, token_weights: dict[str, float], total_weight: float, context_token_sets: list[set[str]] | None = None) -> dict[str, float]:
    # Use a row-level cache to avoid redundant featurization of the same candidate in different phases
    cache = row.setdefault("_usefulness_features_cache", {})
    memory_id = str(candidate["memory_id"])
    if memory_id in cache:
        return dict(cache[memory_id])

    features = _learning_features(row=row, candidate=candidate, decision_summary=decision_summary, profile=profile, query_profile=query_profile)
    query = str(row.get("query", ""))
    c_text = _candidate_memory_text(candidate)
    requested_count = requested_item_count(query) or 0
    c_rank = max(1, int(candidate.get("rank", 1)))
    q_tokens = query_profile.content_tokens
    c_tokens = profile.content_tokens
    salient_overlap, salient_coverage, rare_hit = _salient_query_overlap_features(query_tokens=q_tokens, candidate_tokens=c_tokens, token_weights=token_weights, total_weight=total_weight)
    lower_text = c_text.lower()
    assistant_memory = 1.0 if "assistant:" in lower_text else 0.0
    user_memory = 1.0 if "user:" in lower_text else 0.0
    features.update({
        "active_context_present": 1.0 if row.get("active_context") else 0.0,
        "top_ranked_candidate": 1.0 if c_rank == 1 else 0.0,
        "candidate_rank_inverse": 1.0 / float(c_rank),
        "multi_item_request": 1.0 if requested_count > 1 else 0.0,
        "binary_verification_query": 1.0 if _usefulness_query_is_binary_verification(query) else 0.0,
        "temporal_pair_query": 1.0 if _usefulness_query_requests_temporal_pair(query) else 0.0,
        "event_ordering_query": 1.0 if is_event_ordering_query(row) else 0.0,
        "memory_recall_query": 1.0 if _memory_recall_query(query) else 0.0,
        "repeat_count_query": 1.0 if _repeat_count_query(query) else 0.0,
        "query_candidate_overlap": len(q_tokens & c_tokens) / float(max(len(q_tokens), 1)),
        "salient_query_overlap": salient_overlap,
        "salient_query_coverage": salient_coverage,
        "rare_query_token_hit": rare_hit,
        "assistant_memory": assistant_memory,
        "user_memory": user_memory,
        "mixed_speaker_memory": 1.0 if assistant_memory and user_memory else 0.0,
        "context_redundancy_hit": _context_lexical_overlap(candidate_tokens=c_tokens, context_token_sets=context_token_sets or []),
    })
    
    cache[memory_id] = dict(features)
    return features

def _memory_label_score(memory_labels: dict[str, Any]) -> float:
    labels = set(memory_labels.get("labels", []))
    score = 0.0
    if "answer_bearing" in labels: score += 1.0
    if "conflict_resolving" in labels: score += 0.75
    if "newer_update" in labels: score += 0.55
    if "distractor" in labels: score -= 0.8
    if "redundant_with_active_context" in labels: score -= 0.6
    if "duplicate_family" in labels: score -= 0.2
    if "useless_under_budget" in labels: score -= 0.2
    if memory_labels.get("leave_one_out_effect") == "breaks_sufficiency": score += 0.35
    if memory_labels.get("add_one_rescue_effect") == "restores_sufficiency": score += 0.2
    if memory_labels.get("uncertainty") == "interaction_dependent": score += 0.05
    return score

def _memory_label_features(*, row: dict[str, Any], memory_id: str, candidate: dict[str, Any], decision_summary: dict[str, Any], profile: TextProfile, query_profile: TextProfile, token_weights: dict[str, float], total_weight: float, context_token_sets: list[set[str]] | None = None, memory_labels_by_id: dict[str, dict[str, Any]] | None = None) -> dict[str, float]:
    features = _memory_usefulness_features(row=row, candidate=candidate, decision_summary=decision_summary, profile=profile, query_profile=query_profile, token_weights=token_weights, total_weight=total_weight, context_token_sets=context_token_sets)
    labels = set((memory_labels_by_id or row.get("memory_usefulness_labels", {})).get(memory_id, {}).get("labels", []))
    label_ans, label_conf, label_new = (1.0 if "answer_bearing" in labels else 0.0), (1.0 if "conflict_resolving" in labels else 0.0), (1.0 if "newer_update" in labels else 0.0)
    features.update({
        "label_score": _memory_label_score((memory_labels_by_id or row.get("memory_usefulness_labels", {})).get(memory_id, {})),
        "label_answer_bearing": label_ans, "label_conflict_resolving": label_conf, "label_newer_update": label_new,
        "label_distractor": 1.0 if "distractor" in labels else 0.0, "label_redundant_with_active_context": 1.0 if "redundant_with_active_context" in labels else 0.0,
        "label_duplicate_family": 1.0 if "duplicate_family" in labels else 0.0, "label_useless_under_budget": 1.0 if "useless_under_budget" in labels else 0.0,
        "label_breaks_sufficiency": 1.0 if (memory_labels_by_id or row.get("memory_usefulness_labels", {})).get(memory_id, {}).get("leave_one_out_effect") == "breaks_sufficiency" else 0.0,
        "label_restores_sufficiency": 1.0 if (memory_labels_by_id or row.get("memory_usefulness_labels", {})).get(memory_id, {}).get("add_one_rescue_effect") == "restores_sufficiency" else 0.0,
        "label_interaction_dependent": 1.0 if (memory_labels_by_id or row.get("memory_usefulness_labels", {})).get(memory_id, {}).get("uncertainty") == "interaction_dependent" else 0.0,
        "answer_bearing_without_active_context": label_ans * (1.0 - (1.0 if row.get("active_context") else 0.0)),
        "answer_bearing_when_context_insufficient": label_ans * (1.0 - features.get("context_sufficient", 0.0)),
        "newer_update_for_current_state_label": label_new * features.get("current_state_query", 0.0),
        "conflict_resolving_under_risk": label_conf * float(candidate["signals"]["contradiction_risk"]),
    })
    return features
