"""Scoring logic for the rule_v0 controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from controller.constants import (
    ADVICE_QUERY_PATTERNS,
    BROAD_QUERY_FOCUS_TOKENS,
    CONCEPT_SPECS,
    CURRENT_STATE_QUERY_PATTERNS,
    GENERIC_VALUE_TOKENS,
    NEGATION_PATTERNS,
    QUERY_INTENT_TOKENS,
    QUERY_PERSONALIZATION_PATTERNS,
    TEMPORAL_HINTS,
    WRITING_HELP_PATTERNS,
)
from controller.models import TextProfile
from controller.text_utils import build_profile, tokenize, contains_pattern
import tiktoken
from functools import lru_cache

@lru_cache(maxsize=1)
def _get_tiktoken_encoding():
    return tiktoken.encoding_for_model("gpt-4")



def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class OverlapScore:
    matched_units: int
    total_units: int

    @property
    def value(self) -> float:
        if self.total_units <= 0:
            return 0.0
        return self.matched_units / self.total_units

    def at_least(self, threshold: float) -> bool:
        if self.total_units <= 0:
            return threshold <= 0.0
        threshold_units = round(threshold * 20)
        if abs(threshold * 20 - threshold_units) > 1e-9:
            raise ValueError(f"Threshold {threshold!r} is not representable in twentieths.")
        return self.matched_units * 20 >= self.total_units * int(threshold_units)


def weighted_overlap_score(query_tokens: set[str], candidate_tokens: set[str]) -> OverlapScore:
    if not query_tokens:
        return OverlapScore(matched_units=0, total_units=0)
    overlap = query_tokens & candidate_tokens
    numerator = 0
    denominator = 0
    for token in query_tokens:
        weight = 5
        if len(token) >= 7 or token.isdigit():
            weight = 7
        elif token in TEMPORAL_HINTS:
            weight = 6
        denominator += weight
        if token in overlap:
            numerator += weight
    return OverlapScore(matched_units=numerator, total_units=denominator)


def weighted_overlap(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    return weighted_overlap_score(query_tokens, candidate_tokens).value


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def query_type(query_profile: TextProfile) -> str:
    normalized = query_profile.normalized_text
    tokens = query_profile.token_set
    if normalized.startswith(("when ", "how long ago ", "how many days ago ", "how many weeks ago ", "how many months ago ", "how many years ago ")):
        return "temporal"
    if "remind" in tokens or "remember" in tokens:
        return "recall"
    if "recommend" in tokens or "suggest" in tokens:
        return "recommendation"
    if "advice" in tokens or "tips" in tokens or any(pattern in normalized for pattern in ADVICE_QUERY_PATTERNS):
        return "recommendation"
    if normalized.startswith("when "):
        return "temporal"
    if any(pattern in normalized for pattern in WRITING_HELP_PATTERNS):
        return "writing_help"
    if normalized.startswith(("did ", "do ", "is ", "am ", "are ")):
        return "status"
    if any(pattern in normalized for pattern in QUERY_PERSONALIZATION_PATTERNS):
        return "recommendation"
    if normalized.startswith(("what should i", "which seat should i")):
        return "recommendation"
    if normalized.startswith(("what ", "which ", "would ")):
        return "entity"
    return "generic"


def is_current_state_query(query_profile: TextProfile, qtype: str) -> bool:
    if qtype == "temporal":
        return query_profile.normalized_text.startswith(("when should ", "when is ", "when are "))
    normalized = query_profile.normalized_text
    tokens = query_profile.token_set
    if any(pattern in normalized for pattern in CURRENT_STATE_QUERY_PATTERNS):
        return True
    if qtype == "status":
        return True
    if qtype in {"entity", "recommendation"} and tokens & {"i", "my", "me", "we", "our"}:
        if "should" in tokens or "finally" in tokens or ("am" in tokens and normalized.startswith("what ")):
            return True
    return normalized.startswith(("is ", "am ", "are ", "do ", "does "))


def query_anchor_tokens(query_profile: TextProfile) -> set[str]:
    anchors = {token for token in query_profile.content_tokens if token not in QUERY_INTENT_TOKENS}
    return anchors or query_profile.content_tokens


def query_focus_tokens(query_profile: TextProfile, qtype: str) -> set[str]:
    anchors = query_anchor_tokens(query_profile)
    if qtype in {"recall", "recommendation", "writing_help"}:
        focused = {token for token in anchors if token not in BROAD_QUERY_FOCUS_TOKENS}
        if focused:
            return focused
    return anchors


def query_overlap(
    query_profile: TextProfile,
    candidate_profile: TextProfile,
    qtype: str,
) -> tuple[OverlapScore, OverlapScore]:
    broad_overlap = weighted_overlap_score(query_profile.content_tokens, candidate_profile.content_tokens)
    focused_tokens = query_focus_tokens(query_profile, qtype)
    focused_overlap = weighted_overlap_score(focused_tokens, candidate_profile.content_tokens)
    return focused_overlap, broad_overlap


def primary_concepts(profile: TextProfile) -> set[str]:
    if not profile.concept_scores:
        return set()
    max_score = max(profile.concept_scores.values())
    threshold = max(0.25, max_score * 0.8)
    return {concept for concept, score in profile.concept_scores.items() if score >= threshold}


def concept_overlap(query_concepts: set[str], candidate_profile: TextProfile) -> float:
    if not query_concepts:
        return 0.0
    score = sum(candidate_profile.concept_scores.get(concept, 0.0) for concept in query_concepts)
    return clamp(score / len(query_concepts))


def question_fit(
    query: str,
    query_profile: TextProfile,
    query_primary_concepts: set[str],
    candidate_profile: TextProfile,
    lexical_overlap: OverlapScore,
    broad_overlap: OverlapScore,
) -> float:
    qtype = query_type(query_profile)
    shared_primary = {
        concept for concept in query_primary_concepts if concept in candidate_profile.concept_scores
    }

    if qtype == "temporal":
        if shared_primary:
            for concept in shared_primary:
                markers = CONCEPT_SPECS[concept]["answer_markers"]
                if any(
                    marker in candidate_profile.normalized_text
                    if " " in marker
                    else marker in candidate_profile.token_set
                    for marker in markers
                ):
                    return 1.0
            if candidate_profile.has_time_marker and lexical_overlap.at_least(0.45):
                return 0.4
        elif candidate_profile.has_time_marker and lexical_overlap.at_least(0.25):
            return 0.7

    if qtype == "recall":
        if shared_primary:
            for concept in shared_primary:
                markers = CONCEPT_SPECS[concept]["answer_markers"]
                if any(
                    marker in candidate_profile.normalized_text
                    if " " in marker
                    else marker in candidate_profile.token_set
                    for marker in markers
                ):
                    return 1.0
            return 0.55
        if lexical_overlap.at_least(0.6):
            return 0.9
        if lexical_overlap.at_least(0.35):
            return 0.6
        if broad_overlap.at_least(0.45):
            return 0.15
        if broad_overlap.at_least(0.25):
            return 0.05

    if qtype in {"recommendation", "writing_help"}:
        if lexical_overlap.at_least(0.55):
            if candidate_profile.is_preference:
                return 1.0
            if candidate_profile.is_personal and not candidate_profile.is_sensitive:
                return 0.85
            return 0.7
        if lexical_overlap.at_least(0.35):
            if candidate_profile.is_preference:
                return 0.85
            if candidate_profile.is_personal and not candidate_profile.is_sensitive:
                return 0.65
            return 0.45
        if candidate_profile.is_preference and broad_overlap.at_least(0.25):
            return 0.25
        if candidate_profile.is_personal and not candidate_profile.is_sensitive and broad_overlap.at_least(0.25):
            return 0.15
        return 0.0

    if qtype in {"status", "entity"} and candidate_profile.has_update_marker:
        if lexical_overlap.at_least(0.1) or bool(candidate_profile.named_tokens & query_profile.named_tokens):
            return 0.75

    if shared_primary:
        for concept in shared_primary:
            markers = CONCEPT_SPECS[concept]["answer_markers"]
            if any(
                marker in candidate_profile.normalized_text
                if " " in marker
                else marker in candidate_profile.token_set
                for marker in markers
            ):
                return 1.0
        return 0.45

    if lexical_overlap.at_least(0.55):
        return 0.6
    if lexical_overlap.at_least(0.25):
        return 0.3
    return 0.0


def base_necessity(
    query: str,
    query_profile: TextProfile,
    query_primary_concepts: set[str],
    candidate_profile: TextProfile,
    use_question_fit: bool = True,
) -> float:
    qtype = query_type(query_profile)
    lexical_score, broad_lexical_score = query_overlap(query_profile, candidate_profile, qtype)
    lexical = lexical_score.value
    broad_lexical = broad_lexical_score.value
    concept = concept_overlap(query_primary_concepts, candidate_profile)
    qfit = (
        question_fit(
            query,
            query_profile,
            query_primary_concepts,
            candidate_profile,
            lexical_score,
            broad_lexical_score,
        )
        if use_question_fit
        else 0.0
    )
    base = 0.45 * max(lexical, concept) + 0.25 * min(lexical, concept) + 0.30 * qfit
    if qtype in {"recall", "recommendation", "writing_help"} and lexical < 0.2 and broad_lexical_score.at_least(0.25):
        base *= 0.55
    name_recall_query = qtype == "recall" and "name" in query_profile.content_tokens
    if name_recall_query and lexical_score.at_least(0.2):
        unseen_named = candidate_profile.named_tokens - query_profile.named_tokens
        if 0 < len(unseen_named) <= 3:
            base += 0.08
    if query_profile.named_tokens:
        if query_profile.named_tokens & candidate_profile.named_tokens:
            base += 0.05
        elif candidate_profile.named_tokens and not name_recall_query:
            base *= 0.75
    return clamp(base)


def shared_focus_concepts(
    left: TextProfile,
    right: TextProfile,
    query_primary_concepts: set[str],
) -> set[str]:
    shared = set(left.concept_scores) & set(right.concept_scores)
    if query_primary_concepts:
        shared &= query_primary_concepts
    return shared


def same_family(
    left: TextProfile,
    right: TextProfile,
    query_primary_concepts: set[str],
) -> bool:
    if shared_focus_concepts(left, right, query_primary_concepts):
        return True
    return jaccard(left.content_tokens, right.content_tokens) >= 0.45


def salient_tokens(profile: TextProfile) -> set[str]:
    return {
        token
        for token in profile.content_tokens
        if token not in GENERIC_VALUE_TOKENS and len(token) >= 3
    }


def value_conflict(
    left: TextProfile,
    right: TextProfile,
    query_primary_concepts: set[str],
) -> bool:
    if not same_family(left, right, query_primary_concepts):
        return False
    lexical_similarity = jaccard(left.content_tokens, right.content_tokens)
    if lexical_similarity >= 0.7:
        return False

    left_values = salient_tokens(left)
    right_values = salient_tokens(right)
    if not left_values or not right_values:
        return False

    shared_values = left_values & right_values
    if shared_focus_concepts(left, right, query_primary_concepts) and (
        left.has_update_marker or right.has_update_marker
    ):
        return lexical_similarity < 0.8
    if not shared_values:
        return True
    if len(shared_values) <= 1 and (left.has_update_marker or right.has_update_marker):
        return True
    return False


def is_newer(left: TextProfile, right: TextProfile) -> bool:
    if left.date_key and right.date_key:
        return left.date_key > right.date_key
    if left.date_key and not right.date_key:
        return True
    if not left.date_key and right.date_key:
        return False
    if left.has_update_marker and not right.has_update_marker:
        return True
    return False


def query_warrants_personalization(query_profile: TextProfile, qtype: str) -> bool:
    normalized = query_profile.normalized_text
    if qtype in {"recommendation", "writing_help"}:
        return True
    if any(pattern in normalized for pattern in QUERY_PERSONALIZATION_PATTERNS):
        return True
    return any(token in normalized.split() for token in {"i", "my", "me"})


def token_count(text: str) -> int:
    try:
        enc = _get_tiktoken_encoding()
        return len(enc.encode(text))
    except Exception:
        # Fallback to lexical count if tiktoken fails
        return len(tokenize(text))



def run_example(
    row: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    profiling_cfg = config.get("profiling", {})
    use_concept_specs = bool(profiling_cfg.get("use_concept_specs", False))
    use_question_fit = bool(profiling_cfg.get("use_question_fit", True))
    use_context_current_override = bool(profiling_cfg.get("use_context_current_override", True))
    query = str(row["query"])
    query_profile = build_profile(query, use_concept_specs=use_concept_specs)
    query_primary = primary_concepts(query_profile)
    qtype = query_type(query_profile)
    current_state_query = is_current_state_query(query_profile, qtype)

    from controller.text_utils import merge_profiles
    context_turns = row.get("active_context", [])
    if not context_turns:
        context_profile = None
    else:
        # Build incrementally using cached turn profiles
        turn_profiles = [build_profile(t, use_concept_specs=use_concept_specs) for t in context_turns]
        context_profile = merge_profiles(turn_profiles)
        
    context_coverage = (
        base_necessity(
            query,
            query_profile,
            query_primary,
            context_profile,
            use_question_fit=use_question_fit,
        )
        if context_profile
        else 0.0
    )
    context_current_override = bool(
        use_context_current_override
        and
        context_profile
        and current_state_query
        and (
            context_profile.has_update_marker
            or weighted_overlap_score(query_profile.content_tokens, context_profile.content_tokens).at_least(0.15)
            or any(
                contains_pattern(pattern, context_profile.normalized_text, context_profile.token_set)
                for pattern in NEGATION_PATTERNS
            )
        )
    )
    if context_current_override:
        context_coverage = max(context_coverage, 0.75)
    context_near_sufficient = context_coverage >= 0.45
    context_sufficient = context_coverage >= 0.5

    candidates = []
    for rank, candidate in enumerate(row["candidate_memories"], start=1):
        profile = build_profile(str(candidate["content"]), use_concept_specs=use_concept_specs)
        raw_necessity = base_necessity(
            query,
            query_profile,
            query_primary,
            profile,
            use_question_fit=use_question_fit,
        )
        candidates.append(
            {
                "memory_id": str(candidate["memory_id"]),
                "rank": rank,
                "text": str(candidate["content"]),
                "profile": profile,
                "raw_necessity": raw_necessity,
            }
        )

    newest_update_key = max(
        (candidate["profile"].date_key or (0, 0, 0) for candidate in candidates if candidate["profile"].has_update_marker),
        default=None,
    )
    any_update_candidate = any(candidate["profile"].has_update_marker for candidate in candidates)

    for candidate in candidates:
        profile = candidate["profile"]
        raw_necessity = candidate["raw_necessity"]
        if qtype in {"recommendation", "writing_help"} and profile.is_preference and not profile.is_sensitive:
            raw_necessity = max(raw_necessity, 0.58)
        if current_state_query and not context_sufficient:
            if profile.has_update_marker:
                raw_necessity = max(raw_necessity, 0.45)
                if newest_update_key is not None and (profile.date_key or (0, 0, 0)) == newest_update_key:
                    raw_necessity = max(raw_necessity, 0.68)
            elif any_update_candidate:
                raw_necessity *= 0.55
        candidate["raw_necessity"] = clamp(raw_necessity)

    for candidate in candidates:
        profile = candidate["profile"]
        candidate["retrieval_prior"] = clamp(
            1.0 - ((candidate["rank"] - 1) / max(len(candidates) - 1, 1))
        )

        duplicate_with_context = False
        conflicts_with_context = False
        if context_profile:
            duplicate_with_context = (
                jaccard(profile.content_tokens, context_profile.content_tokens) >= 0.65
                or (
                    same_family(profile, context_profile, query_primary)
                    and not value_conflict(profile, context_profile, query_primary)
                    and jaccard(salient_tokens(profile), salient_tokens(context_profile)) >= 0.4
                )
            )
            conflicts_with_context = same_family(profile, context_profile, query_primary) and value_conflict(
                profile, context_profile, query_primary
            )
            if context_current_override and not duplicate_with_context:
                conflicts_with_context = True

        candidate["duplicate_with_context"] = duplicate_with_context
        candidate["conflicts_with_context"] = conflicts_with_context

        if context_profile and (context_sufficient or context_near_sufficient or duplicate_with_context):
            if duplicate_with_context:
                necessity = min(candidate["raw_necessity"], 0.15)
            elif conflicts_with_context:
                necessity = min(candidate["raw_necessity"], 0.10)
            elif context_near_sufficient:
                necessity = min(candidate["raw_necessity"], 0.35)
            else:
                necessity = min(candidate["raw_necessity"], 0.25)
        else:
            necessity = candidate["raw_necessity"]
            if current_state_query and any_update_candidate and not profile.has_update_marker:
                necessity = min(necessity, 0.10)
        candidate["necessity"] = necessity

    for candidate in candidates:
        profile = candidate["profile"]
        newer_family_alternative = False
        stronger_duplicate = False
        for other in candidates:
            if other["memory_id"] == candidate["memory_id"]:
                continue
            other_profile = other["profile"]
            if not same_family(profile, other_profile, query_primary):
                continue
            if is_newer(other_profile, profile) and value_conflict(profile, other_profile, query_primary):
                newer_family_alternative = True
            if (
                other["rank"] < candidate["rank"]
                and other["raw_necessity"] >= candidate["raw_necessity"] + 0.05
                and not is_newer(profile, other_profile)
                and (
                    jaccard(profile.content_tokens, other_profile.content_tokens) >= 0.25
                    or shared_focus_concepts(profile, other_profile, query_primary)
                )
            ):
                stronger_duplicate = True

        if current_state_query and qtype in {"temporal", "status"} and not profile.has_update_marker:
            newer_same_family_update = any(
                other["memory_id"] != candidate["memory_id"]
                and other["profile"].has_update_marker
                and same_family(profile, other["profile"], query_primary)
                for other in candidates
            )
            if newer_same_family_update:
                newer_family_alternative = True

        candidate["has_newer_family_alternative"] = newer_family_alternative

        if newer_family_alternative:
            freshness = 0.0
        elif profile.has_update_marker and (current_state_query or qtype == "temporal"):
            freshness = 1.0
        elif qtype == "temporal" and (profile.date_key is not None or profile.has_time_marker):
            freshness = 1.0
        else:
            freshness = 0.5
        candidate["freshness"] = freshness

        contradiction_risk = 0.0
        if candidate["conflicts_with_context"]:
            contradiction_risk = 1.0
        elif newer_family_alternative:
            contradiction_risk = 0.85
        elif any(contains_pattern(pattern, profile.normalized_text, profile.token_set) for pattern in NEGATION_PATTERNS):
            contradiction_risk = 0.5
        candidate["contradiction_risk"] = contradiction_risk

        redundancy_risk = 0.0
        personalization_risk = 0.0
        if context_sufficient or context_near_sufficient or candidate["duplicate_with_context"]:
            if candidate["duplicate_with_context"]:
                redundancy_risk = 1.0
            elif candidate["necessity"] < 0.2:
                redundancy_risk = 0.8
            elif context_near_sufficient and not context_sufficient:
                redundancy_risk = 0.75 if qtype in {"recall", "recommendation", "writing_help"} else 0.6
            else:
                redundancy_risk = 0.65 if qtype in {"recall", "recommendation", "writing_help"} else 0.5
        elif stronger_duplicate:
            redundancy_risk = 0.8

        if profile.is_sensitive and candidate["raw_necessity"] < 0.5:
            personalization_risk = 1.0
        elif profile.is_sensitive:
            personalization_risk = 0.8
        elif profile.is_personal:
            if candidate["raw_necessity"] >= 0.55:
                personalization_risk = 0.0
            elif (
                profile.is_preference
                and not profile.is_sensitive
                and query_warrants_personalization(query_profile, qtype)
            ):
                personalization_risk = 0.0
            elif query_warrants_personalization(query_profile, qtype) and candidate["raw_necessity"] >= 0.45:
                personalization_risk = 0.0
            elif candidate["raw_necessity"] < 0.4:
                personalization_risk = 0.7
            else:
                personalization_risk = 0.4

        candidate["redundancy_or_unnecessary_personalization"] = max(
            redundancy_risk,
            personalization_risk,
        )
        candidate["has_redundancy_flag"] = redundancy_risk >= 0.7
        candidate["has_personalization_flag"] = personalization_risk >= 0.7

    weights = config["weights"]
    threshold = float(config["thresholds"]["commit_score_gte"])
    hard_cfg = config["thresholds"]["hard_suppress"]

    for candidate in candidates:
        hard_suppress = False
        if candidate["contradiction_risk"] >= float(hard_cfg["contradiction_risk_gte"]):
            hard_suppress = True
        elif (
            candidate["freshness"] == float(hard_cfg["freshness_equals"])
            and hard_cfg["require_newer_family_alternative"]
            and candidate["has_newer_family_alternative"]
        ):
            hard_suppress = True
        elif (
            candidate["redundancy_or_unnecessary_personalization"]
            >= float(hard_cfg["redundancy_or_unnecessary_personalization_gte"])
            and candidate["necessity"] < float(hard_cfg["necessity_lt"])
        ):
            hard_suppress = True

        if hard_suppress:
            score = 0.0
            decision = "suppress"
        else:
            score = clamp(
                weights["retrieval_prior"] * candidate["retrieval_prior"]
                + weights["necessity"] * candidate["necessity"]
                + weights["freshness"] * candidate["freshness"]
                + weights["contradiction_risk"] * candidate["contradiction_risk"]
                + weights["redundancy_or_unnecessary_personalization"]
                * candidate["redundancy_or_unnecessary_personalization"]
            )
            decision = "commit" if score >= threshold else "suppress"

        candidate["score"] = round(score, 3)
        candidate["decision"] = decision
        candidate["hard_suppress"] = hard_suppress

    fallback_cfg = config["thresholds"]["positive_control_fallback"]
    fallback_applied = False
    if fallback_cfg["enabled"] and not context_sufficient and not any(
        candidate["decision"] == "commit" for candidate in candidates
    ):
        top = candidates[0]
        if (
            top["necessity"] >= float(fallback_cfg["necessity_gte"])
            and top["contradiction_risk"] < float(fallback_cfg["contradiction_risk_lt"])
            and top["redundancy_or_unnecessary_personalization"]
            < float(fallback_cfg["redundancy_or_unnecessary_personalization_lt"])
        ):
            top["decision"] = "commit"
            fallback_applied = True

    max_selected = int(config["selection"]["max_selected"])
    committed = sorted(
        (candidate for candidate in candidates if candidate["decision"] == "commit"),
        key=lambda item: item["score"],
        reverse=True,
    )
    allowed_ids = {candidate["memory_id"] for candidate in committed[:max_selected]}
    for candidate in candidates:
        candidate["dropped_by_budget"] = False
        if candidate["decision"] == "commit" and candidate["memory_id"] not in allowed_ids:
            candidate["decision"] = "suppress"
            candidate["dropped_by_budget"] = True

    selected = [candidate for candidate in candidates if candidate["decision"] == "commit"]
    selected_memory_ids = [candidate["memory_id"] for candidate in selected]
    selected_token_count = sum(token_count(candidate["text"]) for candidate in selected)
    predicted_action = "commit" if selected else "suppress"

    output_candidates = []
    for candidate in candidates:
        reason_codes: list[str] = []
        if candidate["retrieval_prior"] >= 0.67:
            reason_codes.append("REL_HIGH")
        else:
            reason_codes.append("REL_LOW")

        if candidate["necessity"] >= 0.55:
            reason_codes.append("NECESSARY_FOR_QUERY")
        else:
            reason_codes.append("NOT_NECESSARY_FOR_QUERY")

        if candidate["freshness"] >= 0.75:
            reason_codes.append("FRESH_OR_UNCONTESTED")
        elif candidate["has_newer_family_alternative"]:
            reason_codes.append("STALE_SUPERSEDED")

        if candidate["contradiction_risk"] >= 0.5:
            reason_codes.append("CONTRADICTION_RISK")
        if candidate["has_redundancy_flag"]:
            reason_codes.append("REDUNDANT_WITH_STRONGER_MEMORY")
        if candidate["has_personalization_flag"]:
            reason_codes.append("UNNECESSARY_PERSONALIZATION")

        if candidate["hard_suppress"]:
            reason_codes.append("SUPPRESSED_BY_HARD_RULE")
        elif candidate["decision"] == "commit":
            reason_codes.append("COMMITTED_BY_THRESHOLD")
        else:
            reason_codes.append("SUPPRESSED_BY_THRESHOLD")

        if fallback_applied and candidate["memory_id"] == candidates[0]["memory_id"] and candidate["decision"] == "commit":
            reason_codes.append("POSITIVE_CONTROL_FALLBACK")
        if candidate["dropped_by_budget"]:
            reason_codes.append("DROPPED_BY_BUDGET")

        output_candidates.append(
            {
                "memory_id": candidate["memory_id"],
                "rank": candidate["rank"],
                "text": candidate["text"],
                "signals": {
                    "retrieval_prior": round(candidate["retrieval_prior"], 3),
                    "necessity": round(candidate["necessity"], 3),
                    "freshness": round(candidate["freshness"], 3),
                    "contradiction_risk": round(candidate["contradiction_risk"], 3),
                    "redundancy_or_unnecessary_personalization": round(
                        candidate["redundancy_or_unnecessary_personalization"], 3
                    ),
                },
                "score": candidate["score"],
                "decision": candidate["decision"],
                "reason_codes": reason_codes,
            }
        )

    return {
        "example_id": row.get("example_id"),
        "source": row.get("source"),
        "harm_type": row.get("harm_type"),
        "query": query,
        "active_context": row.get("active_context", []),
        "controller_version": config["controller_version"],
        "gold_decision": row.get("gold_decision", {}),
        "decision_summary": {
            "predicted_action": predicted_action,
            "predicted_selected_memory_ids": selected_memory_ids,
            "selected_count": len(selected),
            "selected_token_count": selected_token_count,
            "max_selected": max_selected,
            "threshold": threshold,
            "context_sufficient": context_sufficient,
            "context_coverage": round(context_coverage, 3),
            "context_near_sufficient": context_near_sufficient,
            "use_concept_specs": use_concept_specs,
            "use_question_fit": use_question_fit,
            "query_type": qtype,
            "current_state_query": current_state_query,
            "use_context_current_override": use_context_current_override,
            "context_current_override": context_current_override,
        },
        "candidates": output_candidates,
    }
