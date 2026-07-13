"""Low-level query feature extraction used by AnswerIntent."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from ...common import cached_word_tokenize, cached_pos_tag

try:
    from ...query_targets import QueryTargets, extract_query_targets
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight environments
    @dataclass(frozen=True)
    class QueryTargets:
        raw_query: str
        normalized_query: str
        query_family: str
        schema_name: str
        candidate_entities: tuple[str, ...]
        alternative_entities: tuple[str, ...]
        candidate_numbers: tuple[str, ...]
        preferred_attributes: tuple[str, ...]
        choice_markers: tuple[str, ...]
        asks_for_comparison: bool
        asks_for_ordering: bool
        asks_for_current_state: bool
        asks_for_temporal_difference: bool
        asks_for_relative_time: bool
        asks_for_recall_support: bool
        required_slots: tuple[str, ...]
        answerability_hints: dict[str, Any]
        planning_payload: dict[str, Any]


    def _fallback_entities(query: str) -> tuple[str, ...]:
        between_match = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)", query)
        if between_match:
            return tuple(part.strip(" ,?") for part in between_match.groups() if part.strip(" ,?"))
        if " or " in query:
            pieces = [part.strip(" ,?") for part in query.split(" or ")]
            return tuple(piece for piece in pieces[-2:] if piece)
        return ()


    def extract_query_targets(row: dict[str, Any], query_family: str) -> QueryTargets:
        query = normalized_query(row)
        comparison = any(marker in query for marker in (" more", " less", " versus ", " vs ", " compared ", " than ", "difference between"))
        ordering = any(marker in query for marker in (" first ", " second ", " before ", " earliest ", " latest "))
        current_state = any(marker in query for marker in ("current", "currently", "right now", "latest"))
        temporal_difference = " between " in query or " after " in query
        relative_time = " ago " in f" {query} "
        recall_support = any(marker in query for marker in ("last time", "mentioned", "follow up", "which one", "what did i"))
        return QueryTargets(
            raw_query=str(row.get("query", "") or ""),
            normalized_query=query,
            query_family=query_family,
            schema_name="",
            candidate_entities=_fallback_entities(query),
            alternative_entities=(),
            candidate_numbers=tuple(match.group(0) for match in re.finditer(r"\b\d+(?:\.\d+)?\b", query)),
            preferred_attributes=(),
            choice_markers=(),
            asks_for_comparison=comparison,
            asks_for_ordering=ordering,
            asks_for_current_state=current_state,
            asks_for_temporal_difference=temporal_difference,
            asks_for_relative_time=relative_time,
            asks_for_recall_support=recall_support,
            required_slots=(),
            answerability_hints={},
            planning_payload={},
        )

_TARGET_UNIT_RE = re.compile(
    r"\b(days?|weeks?|months?|years?|hours?|minutes?|dollars?|bucks|percent(?:age)?|mph|km|kilometers?|miles?)\b",
    re.IGNORECASE,
)
_MEASURED_VALUE_RE = re.compile(
    r"(?:[$€£]\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:%|percent(?:age)?|mph|km|kilometers?|miles?|hours?|minutes?|days?|weeks?|months?|years?|bucks?|dollars?))",
    re.IGNORECASE,
)
_NUMERIC_CUE_WORDS = frozenset(
    {
        "total",
        "cost",
        "costs",
        "price",
        "prices",
        "amount",
        "amounts",
        "age",
        "ages",
        "duration",
        "durations",
        "wait",
        "difference",
        "differences",
        "speed",
        "speeds",
        "spent",
        "spend",
        "pay",
        "paid",
        "expense",
        "expenses",
        "salary",
        "salaries",
        "income",
        "incomes",
        "rate",
        "rates",
    }
)
_AVERAGE_MARKERS = (" average ", " mean ", " on average ")
_EXTREMUM_MARKERS = (" most ", " least ", " highest ", " lowest ", " biggest ", " smallest ", " largest ", " shortest ", " longest ")
_DELTA_MARKERS = (" increase ", " decrease ", " gain ", " gained ", " growth ", " lost ", " loss ", " drop ", " dropped ", " change in ")
_EXPLICIT_TEMPORAL_REFERENCE_RE = re.compile(
    r"\b(?:today|yesterday|tomorrow|tonight|this\s+(?:morning|afternoon|evening|week|month|year)|"
    r"last\s+(?:night|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"next\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"on\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE,
)
_TEMPORAL_AGGREGATE_SCOPE_RE = re.compile(
    r"\b(?:in|during|through|throughout|across|over|from|between|since)\b",
    re.IGNORECASE,
)
_TEMPORAL_AGGREGATE_ACTIVITY_RE = re.compile(
    r"\b(?:activities|expenses|events|trips|visits|sessions|rides|classes|meetings|flights|festivals|gifts|purchases|sales|expenses)\b",
    re.IGNORECASE,
)
_EVENT_DURATION_ANCHOR_RE = re.compile(
    r"\b(?:trip|visit|stay|vacation|holiday|journey|tour|class|classes|course|courses|internship|program|"
    r"subscription|membership|project|meeting|festival|workout|ride|flight|camp|camping|"
    r"fellowship|residency|deployment|mission|apprenticeship|bootcamp|retreat|conference|seminar|"
    r"relationship|marriage|engagement|lease|contract|treatment|therapy|diet|training|hike|race|"
    r"marathon|job|shift|commute|session|semester|renovation|repair)\b",
    re.IGNORECASE,
)


def normalized_query(row: dict[str, Any]) -> str:
    return str(row.get("query", "") or "").strip().lower()


def interrogative_for_query(query: str) -> str:
    normalized = normalized_text(query)
    if "how many" in normalized:
        return "how_many"
    if "how much" in normalized:
        return "how_much"
    if "how long" in normalized:
        return "how_long"
    if "how often" in normalized:
        return "how_often"
    if re.search(r"\b(?:who|whom)\b", normalized):
        return "who"
    if normalized.startswith("who "):
        return "who"
    if normalized.startswith("where ") or " where " in normalized:
        return "where"
    if normalized.startswith("when ") or " when " in normalized:
        return "when"
    if normalized.startswith("which "):
        return "which"
    if normalized.startswith("why "):
        return "why"
    if re.search(r"^how\s+(?:do|did|does|can|could|would|should)\b", normalized):
        return "how"
    return "what"


def normalized_text(value: str) -> str:
    return str(value or "").strip().lower()


def _word_tokenize(text: str) -> list[str]:
    return cached_word_tokenize(text)


def _pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
    return cached_pos_tag(tuple(tokens))


def is_numeric_query(query: str) -> bool:
    normalized = normalized_text(query)
    padded = f" {normalized} "
    tokens = _word_tokenize(normalized)
    tags = _pos_tag(tokens)
    if " how many " in padded or " how much " in padded:
        return True
    words = {t for t, _ in tags}
    if words & _NUMERIC_CUE_WORDS:
        return True
    if " how long " in padded:
        return True
    if " number of " in padded or " count of " in padded:
        return True
    if _MEASURED_VALUE_RE.search(normalized):
        interrogative = interrogative_for_query(normalized)
        if interrogative in {"how_many", "how_much", "how_long"}:
            return True
        if words & _NUMERIC_CUE_WORDS:
            return True
    return False


def needs_support_slot(query: str) -> bool:
    padded = f" {normalized_text(query)} "
    return any(
        marker in padded
        for marker in (
            " support ",
            " detail ",
            " explain ",
            " advice ",
            " advise ",
            " recommend ",
            " last time ",
            " mentioned ",
            " follow up ",
            " the one ",
            " what did i ",
            " which one ",
        )
    )


def has_current_state_marker(query: str) -> bool:
    padded = f" {normalized_text(query)} "
    if re.search(r"\b(earliest to latest|latest to earliest|from earliest to latest|from latest to earliest|first to last|last to first|order of|sequence)\b", padded):
        return False
    return any(
        marker in padded
        for marker in (
            " current ",
            " currently ",
            " right now ",
            " latest ",
            " newest ",
            " updated ",
            " still ",
            " now ",
            " most recent ",
            " no longer ",
            " anymore ",
            " so far ",
        )
    )


def has_explicit_temporal_reference(query: str) -> bool:
    padded = f" {normalized_text(query)} "
    if " ago " in padded:
        return False
    return bool(_EXPLICIT_TEMPORAL_REFERENCE_RE.search(padded))


def has_latest_resolution_marker(query: str) -> bool:
    padded = f" {normalized_text(query)} "
    if re.search(r"\b(earliest to latest|latest to earliest|from earliest to latest|from latest to earliest|first to last|last to first|order of|sequence)\b", padded):
        return False
    return bool(
        re.search(
            r"\b(?:most\s+recent|most\s+recently|latest|newest)\b",
            padded,
        )
    )


def has_update_resolution_marker(query: str) -> bool:
    padded = f" {normalized_text(query)} "
    return any(
        marker in padded
        for marker in (
            " changed ",
            " change ",
            " update ",
            " updated ",
            " instead of ",
            " from ",
        )
    ) and (" to " in padded or " instead of " in padded or " changed " in padded or " update " in padded)


def is_duration_query(query: str) -> bool:
    normalized = normalized_text(query)
    tokens = _word_tokenize(normalized)
    words = set(tokens)
    if "how" in words and "long" in words:
        return True
    if re.search(r"\bhow\s+many\s+(days?|weeks?|months?|years?|hours?|minutes?)\b", normalized):
        return True
    if re.search(r"\bhow\s+many\s+(days?|weeks?|months?|years?|hours?|minutes?)\b.*\bago\b", normalized):
        return True
    if re.search(r"\bhow\s+many\s+(days?|weeks?|months?|years?|hours?|minutes?)\b.*\b(?:older|younger)\b", normalized):
        return True
    return False


def is_multi_count_query(query: str) -> bool:
    tokens = _word_tokenize(normalized_text(query))
    words = set(tokens)
    if is_duration_query(query):
        return False
    is_count_request = ("how" in words and "many" in words) or ("number" in words and "of" in words) or ("count" in words and "of" in words)
    if not is_count_request:
        return False
    if words & {"currently", "current", "latest", "newest", "still", "now"}:
        return False
    if words & {"total", "altogether", "combined", "across", "both", "all", "times", "sum"}:
        return True
    return False


def is_strict_count_query(query: str) -> bool:
    tokens = _word_tokenize(normalized_text(query))
    words = set(tokens)
    if is_duration_query(query):
        return False
    is_count_request = ("how" in words and "many" in words) or ("number" in words and "of" in words) or ("count" in words and "of" in words)
    if not is_count_request:
        return False
    if words & {"current", "currently", "right", "now", "latest", "newest", "still"}:
        return False
    return True


def is_average_query(query: str) -> bool:
    padded = f" {normalized_text(query)} "
    return any(marker in padded for marker in _AVERAGE_MARKERS)


def is_extremum_selection_query(query: str) -> bool:
    normalized = normalized_text(query)
    padded = f" {normalized} "
    if any(marker in padded for marker in (" most recent ", " latest ", " newest ", " first ", " last ")):
        return False
    if not any(marker in padded for marker in _EXTREMUM_MARKERS):
        return False
    if not any(padded.startswith(prefix) for prefix in (" which ", " where ", " who ")):
        return False
    return is_numeric_query(query) or any(
        marker in padded
        for marker in (" spend ", " spent ", " cost ", " costs ", " price ", " prices ", " followers ", " likes ", " views ", " age ", " gpa ")
    )


def is_delta_query(query: str) -> bool:
    padded = f" {normalized_text(query)} "
    if not (padded.startswith(" how much ") or padded.startswith(" what ")):
        return False
    if any(marker in padded for marker in (" difference between ", " compared ", " versus ", " vs ", " than ")):
        return False
    if not any(marker in padded for marker in _DELTA_MARKERS):
        return False
    return is_numeric_query(query) or any(
        marker in padded
        for marker in (" followers ", " weight ", " money ", " price ", " spend ", " cost ", " salary ", " income ")
    )


def is_duration_sum_query(query: str) -> bool:
    normalized = normalized_text(query)
    padded = f" {normalized} "
    if not is_duration_query(query):
        return False
    if any(marker in padded for marker in (" total ", " in total ", " combined ", " altogether ")):
        return True
    return bool(
        re.search(r"\b(?:spent|spend|took|take)\b", normalized)
        and re.search(r"\b(?:trips|visits|sessions|workouts|rides|flights|meetings|events)\b", normalized)
    )


def has_event_duration_anchor(query: str) -> bool:
    normalized = normalized_text(query)
    if not is_duration_query(normalized):
        return False
    if not re.search(r"\b(?:i|my|we|our)\b", normalized):
        return False
    return bool(_EVENT_DURATION_ANCHOR_RE.search(normalized))


def has_reference_event_clause(query: str) -> bool:
    normalized = normalized_text(query)
    if not is_duration_query(normalized):
        return False
    return bool(
        re.search(
            r"\b(?:when|before|after|since|until)\s+i\b",
            normalized,
        )
    )


def is_temporally_scoped_numeric_aggregate_query(query: str) -> bool:
    normalized = normalized_text(query)
    padded = f" {normalized} "
    interrogative = interrogative_for_query(normalized)
    if interrogative not in {"how_many", "how_much", "how_long", "what"}:
        return False
    if not (is_numeric_query(query) or is_duration_query(query)):
        return False
    if has_current_state_marker(query):
        return False
    if re.search(r"\bwhen\s+i\b", normalized):
        return False
    if any(marker in padded for marker in (" most recent ", " most recently ", " latest ", " newest ")):
        return False
    has_scope = bool(_TEMPORAL_AGGREGATE_SCOPE_RE.search(normalized))
    has_activity_set = bool(_TEMPORAL_AGGREGATE_ACTIVITY_RE.search(normalized))
    has_totalizer = any(marker in padded for marker in (" total ", " in total ", " altogether ", " combined ", " across ", " all ", " both "))
    return bool((has_scope and has_activity_set) or (has_scope and has_totalizer))


def target_unit(query: str) -> str | None:
    normalized = normalized_text(query)
    if " how much time " in f" {normalized} " or " how long " in f" {normalized} ":
        return None
    for match in _TARGET_UNIT_RE.finditer(normalized):
        start = match.start()
        prefix = normalized[max(0, start - 8) : start]
        if prefix.endswith("every ") or prefix.endswith("each ") or prefix.endswith("per "):
            continue
        return normalized_text(match.group(1))
    return None


def answer_type_for_interrogative(interrogative: str, *, numeric_query: bool, duration_query: bool, current_state: bool) -> str:
    if interrogative == "who":
        return "person"
    if interrogative == "where":
        return "location"
    if interrogative == "when":
        return "time"
    if interrogative == "how_long":
        return "duration"
    if interrogative in {"how_many", "how_much"}:
        return "duration" if duration_query else "number"
    if duration_query:
        return "duration"
    if current_state:
        return "state"
    if interrogative == "what" and numeric_query:
        return "number"
    return "attribute"


def cardinality_for_targets(targets: QueryTargets, *, family: str, query: str) -> str:
    if targets.asks_for_comparison or targets.asks_for_ordering:
        return "pair"
    if family == "aggregation" and (
        is_multi_count_query(query)
        or is_strict_count_query(query)
        or is_average_query(query)
        or is_extremum_selection_query(query)
        or is_delta_query(query)
        or is_duration_sum_query(query)
        or " total " in f" {normalized_text(query)} "
    ):
        return "set"
    return "single"


def operator_for_targets(targets: QueryTargets, *, family: str, query: str, numeric_query: bool, duration_query: bool) -> str:
    normalized = f" {normalized_text(query)} "
    if family in {"knowledge_update", "conflict_update"}:
        return "update"
    if family == "temporal" and targets.asks_for_relative_time:
        return "relative_time"
    if family == "temporal" and targets.asks_for_temporal_difference:
        return "diff"
    if family == "current_state" or targets.asks_for_current_state:
        return "current_state"
    if targets.asks_for_comparison:
        return "compare"
    if targets.asks_for_ordering:
        return "order"
    if family == "aggregation" and is_average_query(query):
        return "average"
    if family == "aggregation" and is_extremum_selection_query(query):
        return "select_extreme"
    if family == "aggregation" and is_delta_query(query):
        return "delta"
    if family == "aggregation" and (" total " in normalized or " in total " in normalized or " combined " in normalized or " altogether " in normalized):
        return "sum"
    if family == "aggregation" and is_duration_sum_query(query):
        return "sum"
    if family == "aggregation" and normalized.startswith(" how many ") and not duration_query:
        return "count"
    if family == "aggregation" and (is_multi_count_query(query) or is_strict_count_query(query)):
        return "count"
    if duration_query:
        return "lookup"
    if numeric_query:
        return "lookup"
    return "lookup"


def abstention_sensitivity(targets: QueryTargets) -> str:
    if targets.asks_for_comparison or targets.asks_for_ordering:
        return "contrastive"
    if targets.asks_for_current_state:
        return "closed_world"
    return "open_world"


def is_multi_session_query(row: dict[str, Any]) -> bool:
    """Detect queries that need evidence from multiple conversation sessions.

    Only uses the dataset's question_type label.  Heuristic regex detection
    caused 109 false positives (single-session "how many" questions,
    temporal-reasoning date-difference queries) which hurt accuracy when
    routed through the multi-session fact-list path.
    """
    question_type = str(row.get("question_type") or "").strip().lower()
    return question_type == "multi-session"


def extract_features(row: dict[str, Any], query_family: str) -> tuple[str, QueryTargets]:
    query = normalized_query(row)
    return query, extract_query_targets(row, query_family)
