"""Centralized constants, feature lists, and thresholds for CRISP modeling."""

from .common_import import repo_root

DEFAULT_TRAINING_SLICE_PATHS = [
    "dataset-slices/v3_global_pack_train_dev_test_labeled.jsonl",
    "dataset-slices/v4_external_seed_train_dev_test_labeled.jsonl",
]

METHOD_MODEL_CACHE_PATH = (
    repo_root() / "results" / "model_cache" / "active_method_models.json"
)
METHOD_MODEL_CACHE_PROTOCOL_VERSION = 6
METHOD_CACHE_MODEL_KEYS = (
    "v2_usefulness_label_model",
    "v2_label_selector_v1_model",
)
# Backward-compatible aliases.
ACTIVE_METHOD_MODEL_CACHE_PATH = METHOD_MODEL_CACHE_PATH
CACHED_METHOD_MODEL_KEYS = METHOD_CACHE_MODEL_KEYS

LEARNED_PRUNER_FEATURES = [
    "bias",
    "retrieval_prior",
    "necessity",
    "freshness",
    "contradiction_safe",
    "redundancy_safe",
    "has_update_marker",
    "has_time_marker",
    "is_preference",
    "is_sensitive",
    "is_personal",
    "current_state_query",
    "temporal_query",
    "context_sufficient",
    "name_overlap",
    "update_for_current_state",
    "time_for_temporal",
    "sensitive_without_necessity",
    "policy_update_marker",
    "stale_memory_marker",
    "forget_marker",
    "policy_update_for_current_state",
    "stale_memory_for_current_state",
]

USEFULNESS_PREDICTOR_FEATURES = [
    *LEARNED_PRUNER_FEATURES,
    "active_context_present",
    "top_ranked_candidate",
    "candidate_rank_inverse",
    "multi_item_request",
    "binary_verification_query",
    "temporal_pair_query",
    "event_ordering_query",
    "memory_recall_query",
    "repeat_count_query",
    "query_candidate_overlap",
    "salient_query_overlap",
    "salient_query_coverage",
    "rare_query_token_hit",
    "assistant_memory",
    "user_memory",
    "mixed_speaker_memory",
]

MEMORY_LABEL_TARGETS = (
    "answer_bearing",
    "conflict_resolving",
    "newer_update",
    "distractor",
    "redundant_with_active_context",
    "duplicate_family",
    "useless_under_budget",
)

MEMORY_EFFECT_TARGETS = (
    "leave_one_out_breaks_sufficiency",
    "add_one_rescue_restores_sufficiency",
    "interaction_dependent",
)

LABEL_PRUNER_FEATURES = [
    *USEFULNESS_PREDICTOR_FEATURES,
    "label_score",
    "label_answer_bearing",
    "label_conflict_resolving",
    "label_newer_update",
    "label_distractor",
    "label_redundant_with_active_context",
    "label_duplicate_family",
    "label_useless_under_budget",
    "label_breaks_sufficiency",
    "label_restores_sufficiency",
    "label_interaction_dependent",
    "answer_bearing_without_active_context",
    "answer_bearing_when_context_insufficient",
    "newer_update_for_current_state_label",
    "conflict_resolving_under_risk",
]

LABEL_SELECTOR_FEATURES = [
    *LABEL_PRUNER_FEATURES,
    "family_abstention",
    "family_ordering",
    "family_aggregation",
    "family_multi_session",
    "family_temporal",
    "family_conflict_update",
    "family_current_state",
    "family_knowledge_update",
    "family_contradiction",
    "family_information_extraction",
    "family_single_anchor",
    "family_abstention_answer_bearing",
    "family_abstention_distractor",
    "family_ordering_answer_bearing",
    "family_ordering_breaks_sufficiency",
    "family_aggregation_answer_bearing",
    "family_aggregation_breaks_sufficiency",
    "family_multi_session_answer_bearing",
    "family_multi_session_breaks_sufficiency",
    "family_temporal_answer_bearing",
    "family_temporal_newer_update",
    "family_conflict_update_answer_bearing",
    "family_conflict_update_conflict_resolving",
    "family_conflict_update_newer_update",
    "family_current_state_answer_bearing",
    "family_current_state_newer_update",
    "family_knowledge_update_answer_bearing",
    "family_knowledge_update_newer_update",
    "family_contradiction_answer_bearing",
    "family_contradiction_conflict_resolving",
    "family_information_extraction_answer_bearing",
    "family_information_extraction_breaks_sufficiency",
    "utility_proxy",
]
