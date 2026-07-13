"""Query and content classification heuristics."""

from __future__ import annotations
import re
from typing import Any
from .common_import import is_event_ordering_query, requested_item_count

_COUNT_WORD_HINTS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

_MONTH_NAME_RE = (
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
)

def _query_is_binary_verification(query: str) -> bool:
    lowered = query.strip().lower()
    return lowered.startswith(
        ("have i ", "had i ", "did i ", "do i ", "am i ", "was i ", "were i ", "has the ", "have the ", "did the ", "is the ", "are the ")
    )

def _query_requests_temporal_pair(row: dict[str, Any]) -> bool:
    query = str(row.get("query", "")).strip().lower()
    if not query: return False
    padded = f" {query} "
    pair_markers = (" between ", "how long between", "time between")
    if any(marker in padded for marker in pair_markers):
        return True
    if re.search(r"\bfrom\s+.+?\s+to\s+.+$", query):
        return True
    if re.search(r"\bhow\s+many\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\b.*\b(?:between|from)\b", query):
        return True
    if re.search(r"\bhow\s+(?:long|many)\b.*\b(?:did it take|took|passed)\b", query):
        return True
    if re.search(r"\bhow\s+many\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\b.*\b(?:after|before|until|since)\b", query):
        return True
    if re.search(r"\bhow\s+long\b.*\b(?:wait|waited|after|before|until|since)\b", query):
        return True
    return False

def _query_is_ordering_choice(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered: return False
    if " happened first" in lowered or " came first" in lowered: return True
    if " which " in f" {lowered} " and " first" in lowered and (" or " in lowered or " before " in lowered): return True
    if lowered.startswith(("who did i ", "what did i ", "where did i ", "when did i ")) and " first" in lowered and " or " in lowered: return True
    if "what did i" in lowered and "first" in lowered and (" or " in lowered or " before " in lowered): return True
    return False

def _query_requires_multi_evidence(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered: return False
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered)
    if _query_is_ordering_choice(query): return True
    if _query_requests_temporal_pair({"query": query}): return True
    if lowered.startswith(("what percentage", "what percent", "what are the two", "which two", "what were the two", "what are the total", "what were the total")): return True
    composite_markers = (
        " total number ", " total cost ", " total amount ", " total distance ", " total price ", " in total ", " altogether ", " combined ",
        " sum of ", " minimum amount ", " maximum amount ", " how much more ", " how many more ", " how much less ", " how many less ",
        " compared to ", " compared with ", " difference between ", " increase in ", " decrease in ", " percentage discount ", " both ", " as well as ",
    )
    padded = f" {normalized} "
    if any(marker in padded for marker in composite_markers): return True
    if lowered.startswith("how many ") and (" and " in lowered or " since " in lowered or " than " in lowered): return True
    if lowered.startswith("how many ") and any(
        marker in padded
        for marker in (
            " own ",
            " do i have ",
            " have i ",
            " currently ",
            " right now ",
            " now ",
            " including ",
            " left ",
            " remaining ",
            " are there ",
            " took ",
            " take ",
            " waited ",
            " wait ",
            " passed ",
            " after ",
            " before ",
            " until ",
        )
    ): return True
    if lowered.startswith("how much ") and any(
        marker in padded for marker in (" more ", " less ", " compared ", " than ", " cost ", " costs ", " price ", " spent ")
    ): return True
    if lowered.startswith("how long ") and " ago " not in padded: return True
    if lowered.startswith("what is the total ") or lowered.startswith("what was the total "): return True
    return False


def _is_strict_count_query(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered:
        return False
    padded = f" {lowered} "
    if not (
        lowered.startswith("how many ")
        or lowered.startswith("number of ")
        or lowered.startswith("count of ")
    ):
        return False
    if re.search(r"\bhow\s+many\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\b", lowered):
        return False
    return True

def _selector_prefers_extractive_lookup(query: str) -> bool:
    lowered = query.strip().lower()
    return lowered.startswith(("what ", "who ", "which ", "when ", "where ", "why ", "how many ", "how long ")) or any(m in lowered for m in ("what was the name", "extract", "remind me", "last time"))

def _selector_requests_action_set(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered: return False
    return lowered.startswith(("what has ", "what did ")) and (" done" in lowered or lowered.endswith(" do") or " do " in lowered)

def _selector_requests_list_like_lookup(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered: return False
    return lowered.startswith(("what types of ", "which types of ", "what kind of ", "what kinds of ", "which kind of ", "which kinds of ", "what forms of ", "which forms of "))

def _selector_prefers_reason_lookup(query: str) -> bool:
    lowered = query.strip().lower()
    return lowered.startswith("why ")

def _selector_requests_repeat_count(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered: return False
    return any(marker in lowered for marker in ("how many times", "multiple times", "more than once", "again and again", "repeatedly"))

def _selector_reason_or_realization_query(query: str) -> bool:
    return _selector_prefers_reason_lookup(query) or " realize" in query.lower() or " realized" in query.lower()

def _selector_needs_abstention_backup(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered: return False
    return lowered.startswith("who ") or _selector_requests_action_set(query) or _selector_reason_or_realization_query(query)

def _selector_target_person(query: str) -> str:
    stripped = query.strip()
    if not stripped: return ""
    possessive_match = re.search(r"\b([A-Z][a-z]+)'s\b", stripped)
    if possessive_match: return possessive_match.group(1)
    patterns = (
        r"\bwhat has ([A-Z][a-z]+)\b", r"\bwhat did ([A-Z][a-z]+)\b", r"\bwhat does ([A-Z][a-z]+)\b", r"\bwhy did ([A-Z][a-z]+)\b",
        r"\bwho did ([A-Z][a-z]+)\b", r"\bwhen did ([A-Z][a-z]+)\b", r"\bhow long has ([A-Z][a-z]+)\b", r"\bwould ([A-Z][a-z]+)\b",
        r"\bhas ([A-Z][a-z]+)\b", r"\bhad ([A-Z][a-z]+)\b", r"\bdid ([A-Z][a-z]+)\b", r"\bdoes ([A-Z][a-z]+)\b", r"\bis ([A-Z][a-z]+)\b", r"\bwas ([A-Z][a-z]+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, stripped)
        if match: return match.group(1)
    return ""

def _selector_memory_speaker_name(text: str) -> str:
    match = re.match(r"\s*([A-Z][a-z]+):", text)
    return match.group(1) if match else ""

def _selector_update_intent_query(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered: return False
    return any(marker in lowered for marker in ("current ", "currently ", "right now", "latest ", "newest ", "updated ", "update ", "still ", "now "))

def _selector_explicit_current_marker_query(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered: return False
    padded = f" {lowered} "
    markers = (
        " current ",
        " currently ",
        " right now ",
        " latest ",
        " newest ",
        " updated ",
        " update ",
        " still ",
        " now ",
        " final status ",
        " current status ",
        " current state ",
        " most recent ",
        " no longer ",
        " anymore ",
    )
    return any(marker in padded for marker in markers)

def _selector_should_trust_active_context_only(row: dict[str, Any]) -> bool:
    active_context = [str(item).strip() for item in row.get("active_context") or [] if str(item).strip()]
    if not active_context: return False
    if all("| assistant:" in item.lower() for item in active_context): return False
    return True

def _selector_family_name(*, row: dict[str, Any], decision_summary: dict[str, Any], memory_label_sets: dict[str, set[str]]) -> str:
    evidence = row.get("evidence_sufficiency", {})
    if str(evidence.get("label", "")).strip().lower() == "abstention": return "abstention"
    if bool(evidence.get("active_context_support")) and _selector_should_trust_active_context_only(row): return "abstention"
    query = str(row.get("query", "")).strip().lower()
    query_type_name = str(decision_summary.get("query_type", "")).strip().lower()
    current_state_query = bool(decision_summary.get("current_state_query"))
    explicit_current_marker_query = _selector_explicit_current_marker_query(query)

    if is_event_ordering_query(row) or _query_is_ordering_choice(query): return "ordering"
    if _query_requests_temporal_pair(row): return "temporal"

    requested_count = requested_item_count(query) or 0
    update_like_query = explicit_current_marker_query or _selector_update_intent_query(query)
    binary_query = _query_is_binary_verification(query)
    multi_evidence_query = _query_requires_multi_evidence(query)
    if _selector_requests_action_set(query): return "aggregation"
    if _is_strict_count_query(query) and not update_like_query:
        return "aggregation"
    if multi_evidence_query: return "current_state" if update_like_query else "aggregation"
    if _selector_prefers_reason_lookup(query) and not any((current_state_query, binary_query, update_like_query)): return "information_extraction"

    update_resolution_query = (
        " changed " in f" {query} "
        or " change " in f" {query} "
        or " update " in f" {query} "
        or " updated " in f" {query} "
        or " instead of " in f" {query} "
        or " from " in f" {query} " and " to " in f" {query} "
    )
    contradiction_like_query = binary_query and any(
        marker in f" {query} "
        for marker in (" still ", " anymore ", " no longer ", " true ", " false ", " or did ")
    )
    if query_type_name in {"knowledge_update", "stale_update"}: return "knowledge_update"
    if update_resolution_query and update_like_query: return "knowledge_update"
    if contradiction_like_query and not update_like_query: return "contradiction"
    if query_type_name == "temporal": return "temporal"
    if requested_count > 1: return "aggregation"
    if update_like_query: return "current_state"
    if current_state_query and not _selector_prefers_extractive_lookup(query):
        return "current_state"
    if _selector_prefers_extractive_lookup(query): return "information_extraction"
    return "single_anchor"

def _selector_explicit_recall_lookup(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered: return False
    if "remind me" in lowered: return True
    markers = (
        "previous conversation", "previous chat", "previous discussion", "our earlier conversation", "our earlier chat",
        "you mentioned", "you said", "you told me", "you recommended", "you referred to", "i mentioned", "i said", "i told you",
        "i'm trying to recall", "i am trying to recall", "trying to recall", "i remember you", "i wanted to follow up",
        "follow up on our previous", "follow up on our", "going back to our previous", "going through our previous",
    )
    return any(marker in lowered for marker in markers)

def _selector_recall_lookup_is_fragile(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered: return False
    if (requested_item_count(query) or 0) > 1 or lowered.startswith("how many ") or re.search(r"\b\d+(?:st|nd|rd|th)\b", lowered): return True
    markers = (
        " what were the ", " which were the ", " what are the ", " other ", " list of ", " outlined ", " options ", " objectives ",
        " steps ", " items ", " languages ", " parameters ", " songs ", " chapters ", " what color ", " which color ",
        " what was the color", " what name ", " which name ", " name of that ", " what kind of ", " which kind of ", " what type of ", " which type of ",
    )
    return any(marker in lowered for marker in markers)

def _selector_prefers_latest_answer_query(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered: return False
    if _selector_update_intent_query(query) or any(m in lowered for m in ("most recent", "latest", "newest", "so far")): return True
    if lowered.startswith("how many ") and any(m in lowered for m in (" have i ", " have we ", " are on ", " are in ", " have i taken ", " have i tried ", " have i attended ", " have i gone ", " now", " currently")): return True
    return False

def _selector_prefers_earliest_answer_query(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered: return False
    padded = f" {lowered} "
    if any(m in padded for m in (" first ", " earliest ", " initial ")): return True
    return lowered.startswith(("when did i first", "what was the first", "which was the first"))

def _selector_targets_assistant_memory(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered: return False
    return any(m in lowered for m in ("you mentioned", "you said", "you told me", "you suggested", "you recommended", "you referred", "you wrote", "you gave me"))

def _selector_answer_memory_family(memory_id: str) -> str:
    memory_key = str(memory_id).strip()
    if not memory_key.startswith("answer_"): return ""
    return re.sub(r"_[0-9]+$", "", memory_key)

def _selector_answer_memory_variant_index(memory_id: str) -> int:
    match = re.search(r"_([0-9]+)$", str(memory_id).strip())
    return int(match.group(1)) if match else -1

def _selector_memory_body_text(text: str) -> str:
    body = str(text).split("|")[-1].strip()
    if ":" not in body: return body
    speaker, remainder = body.split(":", 1)
    if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{0,31}", speaker.strip()): return remainder.strip()
    return body

def _selector_parse_count_token(token: str) -> int | None:
    normalized = token.strip().lower()
    if not normalized: return None
    if normalized.isdigit():
        value = int(normalized)
        return value if 0 <= value <= 200 else None
    return _COUNT_WORD_HINTS.get(normalized)

def _selector_query_count_units(query: str) -> tuple[str, ...]:
    lowered = query.strip().lower()
    if not lowered: return ()
    unit_aliases = {
        "trip": ("trip", "trips"), "time": ("time", "times"), "recipe": ("recipe", "recipes"), "woman": ("woman", "women"),
        "man": ("man", "men"), "member": ("member", "members"), "event": ("event", "events"), "session": ("session", "sessions"), "person": ("person", "people"),
    }
    detected = []
    for aliases in unit_aliases.values():
        if any(alias in lowered for alias in aliases): detected.extend(aliases)
    return tuple(sorted(set(detected)))

def _selector_extract_count_hint(text: str, query: str | None = None) -> int | None:
    body = _selector_memory_body_text(text).lower()
    if query:
        query_units = _selector_query_count_units(query)
        if query_units:
            unit_pattern = "|".join(re.escape(unit) for unit in query_units)
            patterns = (
                rf"\b(\d{{1,3}}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:{unit_pattern})\b",
                rf"\btried(?:\s+out)?\s+(\d{{1,3}}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
            )
            for pattern in patterns:
                match = re.search(pattern, body)
                if match:
                    value = _selector_parse_count_token(match.group(1))
                    if value is not None: return value
            if "women" in query_units:
                team_match = re.search(r"\b(\d{1,3})\s+(?:people|members?)\b", body)
                half_match = re.search(r"\bhalf\b[^.]{0,40}\bwomen\b", body)
                if team_match and half_match:
                    team_size = int(team_match.group(1))
                    if team_size >= 2: return team_size // 2
            return None
    patterns = (
        r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:trips?|times?|recipes?|women|woman|man|people|members?|events?|sessions?)\b",
        r"\btried(?:\s+out)?\s+(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
        r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+of\b[^.]{0,60}\b(?:trips?|times?|recipes?|women|people|members?|events?|sessions?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, body)
        if match:
            value = _selector_parse_count_token(match.group(1))
            if value is not None: return value
    return None

def _selector_has_duration_hint(text: str) -> bool:
    body = _selector_memory_body_text(text).lower()
    return re.search(r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:days?|weeks?|months?|years?)\b", body) is not None

def _selector_extract_day_of_month_hint(text: str) -> int | None:
    body = _selector_memory_body_text(text).lower()
    for pattern in (rf"\b(?:on\s+the\s+|on\s+)?(\d{{1,2}})(?:st|nd|rd|th)\s+of\s+{_MONTH_NAME_RE}\b", rf"\b{_MONTH_NAME_RE}\s+(\d{{1,2}})(?:st|nd|rd|th)\b"):
        match = re.search(pattern, body)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 31: return value
    return None
