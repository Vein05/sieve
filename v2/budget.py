"""Token budget enforcement for the v2 packagers.

Two responsibilities:

1. :func:`count_tokens` — a single tiktoken-backed token counter. It reuses
   v1's ``controller.logic.token_count`` (which already loads the encoding
   from tiktoken and falls back to a *lexical* tokenizer, never ``.split()``,
   when tiktoken is unavailable). This is the P0 #2 discipline: efficiency
   metrics must never be computed with ``str.split()``.

2. :func:`truncate_units_to_budget` — greedy whole-unit packing. Units are
   added in priority order while the running total stays within budget. If the
   highest-priority unit alone is too large, its prefix is hard-truncated so a
   labelled budget cell never exceeds its advertised rate.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default encoding name; overridable via the response-surface config. tiktoken
# is imported lazily inside the counter so this module imports cleanly (and the
# test-suite runs) even in environments without a tiktoken wheel.
DEFAULT_ENCODING = "cl100k_base"
UNBOUNDED_BUDGET = 0

_ENCODING_CACHE: dict[str, object] = {}


def _tiktoken_encoding(encoding_name: str):
    """Return a cached tiktoken encoding, or ``None`` if tiktoken is absent."""
    if encoding_name in _ENCODING_CACHE:
        return _ENCODING_CACHE[encoding_name]
    try:  # lazy import — see module docstring
        import tiktoken
    except Exception:  # pragma: no cover - environment-dependent
        _ENCODING_CACHE[encoding_name] = None
        return None
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception:  # pragma: no cover - unknown encoding name
        enc = None
    _ENCODING_CACHE[encoding_name] = enc
    return enc


def count_tokens(text: str, *, encoding: str = DEFAULT_ENCODING) -> int:
    """Count tokens in *text* with tiktoken, falling back to v1's lexical count.

    Uses the tiktoken encoding named by *encoding* (default ``cl100k_base``,
    the same family v1's ``controller.logic.token_count`` uses). When tiktoken
    is unavailable it falls back to v1's lexical tokenizer
    (``controller.text_utils.tokenize``) — never ``str.split()`` (P0 #2).

    tiktoken is imported lazily so this counter (and the whole test-suite) runs
    in environments without a tiktoken wheel. In the production env tiktoken is
    a declared v1 dependency (see requirements.txt), so real efficiency metrics
    match v1 exactly.
    """
    text = str(text or "")
    if not text:
        return 0
    enc = _tiktoken_encoding(encoding)
    if enc is not None:
        return len(enc.encode(text))
    # Fallback path: v1's lexical tokenizer (NOT str.split()).
    from controller.text_utils import tokenize as _lexical_tokenize

    return len(_lexical_tokenize(text))


@dataclass(frozen=True)
class BudgetPacking:
    """Result of :func:`truncate_units_to_budget`.

    ``units`` is the kept prefix (in the input priority order). ``token_count``
    never exceeds a positive budget. ``overflow`` records that the first unit
    required hard truncation. Budget ``0`` means unbounded.
    """

    units: list[str]
    token_count: int
    overflow: bool


def hard_truncate_text(
    text: str,
    budget: int,
    *,
    encoding: str = DEFAULT_ENCODING,
) -> str:
    """Return the longest practical text prefix within a positive budget."""
    text = str(text or "")
    if budget == UNBOUNDED_BUDGET or count_tokens(text, encoding=encoding) <= budget:
        return text
    if budget < 0:
        raise ValueError("budget must be non-negative; use 0 for unbounded")
    enc = _tiktoken_encoding(encoding)
    if enc is not None:
        clipped = enc.decode(enc.encode(text)[:budget]).rstrip()
        while clipped and count_tokens(clipped, encoding=encoding) > budget:
            clipped = clipped[:-1].rstrip()
        return clipped
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(text[:mid], encoding=encoding) <= budget:
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip()


def _pack_bounded(
    units: list[str], budget: int, encoding: str, joiner: str
) -> BudgetPacking:
    kept: list[str] = []
    for unit in units:
        candidate = kept + [unit]
        total = count_tokens(joiner.join(candidate), encoding=encoding)
        if total <= budget:
            kept = candidate
            continue
        if not kept:
            clipped = hard_truncate_text(unit, budget, encoding=encoding)
            return BudgetPacking(
                units=[clipped] if clipped else [],
                token_count=count_tokens(clipped, encoding=encoding),
                overflow=True,
            )
        break
    return BudgetPacking(
        units=kept,
        token_count=count_tokens(joiner.join(kept), encoding=encoding),
        overflow=False,
    )


def truncate_units_to_budget(
    units: list[str],
    budget: int,
    *,
    encoding: str = DEFAULT_ENCODING,
    joiner: str = "\n",
) -> BudgetPacking:
    """Greedily pack whole units into *budget* tokens, preserving order.

    Adds whole units while the joined total stays ``<= budget``. If the first
    unit cannot fit, a prefix is retained and ``overflow=True`` records the
    truncation. Budget ``0`` is the explicit unbounded control.

    Args:
        units: candidate units in priority order (highest priority first).
        budget: token budget (tiktoken count of the joined result).
        encoding: tiktoken encoding name.
        joiner: string used to join units when measuring the running total.
    """
    clean = [str(u) for u in units if str(u).strip()]
    if not clean:
        return BudgetPacking(units=[], token_count=0, overflow=False)
    if budget < 0:
        raise ValueError("budget must be non-negative; use 0 for unbounded")
    if budget == UNBOUNDED_BUDGET:
        return BudgetPacking(
            units=clean,
            token_count=count_tokens(joiner.join(clean), encoding=encoding),
            overflow=False,
        )

    return _pack_bounded(clean, budget, encoding, joiner)
