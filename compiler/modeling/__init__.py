"""CRISP Modeling Logic package."""

# Classification & Intents
from .classification import (
    _query_is_binary_verification,
    _query_requests_temporal_pair,
    _query_requires_multi_evidence,
    _selector_family_name,
    _selector_explicit_recall_lookup,
    _selector_targets_assistant_memory,
)

# Feature Extraction
from .features import (
    _query_token_weights,
    _lexical_content_tokens,
)

# Ranking / Selection / Training may depend on optional local deps.
try:
    from .ranking import (
        _selector_ranked_candidates,
        _selector_ranked_candidates_with_backstop,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    pass

try:
    from .selection import (
        _selector_add_anchor_support,
        _selector_information_extraction_bundle_selection,
        _selector_chronological_coverage_selection,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    pass

try:
    from .training import (
        train_selector_model,
    )
    train_v2_label_selector_v1 = train_selector_model
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    pass
