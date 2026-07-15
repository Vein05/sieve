"""Evidence-style packagers and the style registry.

Evidence policies from ``research/idea-v2.md``:

* ``raw``               - top-20 candidates in retrieval order (control)
* ``filtered_raw``      - distractors dropped, survivors kept verbatim
* ``extractive``        - top sentences, verbatim, chronological order
* ``selected_raw``      - v1-selected passages rendered verbatim
* ``selected_extractive`` - v1-selected passages rendered sentence-wise
* ``structured``        - re-rendered v1 SIEVE package
* ``structured_quotes`` - structured block + exact source quotes
* ``summary``           - generic abstractive summary (injected compile-LLM)
* ``sieve_v1``          - exact full-prompt published-system control

Look up a packager with :func:`get_packager`; iterate all styles via
:data:`STYLE_REGISTRY` (dict is insertion-ordered to match the config).
"""

from __future__ import annotations

from v2.packagers.base import Packager, PackagerContext
from v2.packagers.extractive import ExtractivePackager, SelectedExtractivePackager
from v2.packagers.filtered_raw import FilteredRawPackager
from v2.packagers.guarded_structured import GuardedStructuredPackager
from v2.packagers.raw import RawPackager
from v2.packagers.selected_raw import SelectedRawPackager
from v2.packagers.sieve_v1 import SieveV1Packager
from v2.packagers.structured import StructuredPackager
from v2.packagers.structured_generic import StructuredGenericPackager
from v2.packagers.structured_quotes import StructuredQuotesPackager
from v2.packagers.summary import SummaryPackager

# Insertion order mirrors research/idea-v2.md's variant list (1..6) plus the v3
# generic-packager pilot styles (structured_generic, guarded_structured).
STYLE_REGISTRY: dict[str, Packager] = {
    RawPackager.style_name: RawPackager(),
    FilteredRawPackager.style_name: FilteredRawPackager(),
    ExtractivePackager.style_name: ExtractivePackager(),
    SelectedRawPackager.style_name: SelectedRawPackager(),
    SelectedExtractivePackager.style_name: SelectedExtractivePackager(),
    StructuredPackager.style_name: StructuredPackager(),
    StructuredQuotesPackager.style_name: StructuredQuotesPackager(),
    StructuredGenericPackager.style_name: StructuredGenericPackager(),
    GuardedStructuredPackager.style_name: GuardedStructuredPackager(),
    SummaryPackager.style_name: SummaryPackager(),
    SieveV1Packager.style_name: SieveV1Packager(),
}

# Styles that never need the compile-LLM (safe to run fully offline).
OFFLINE_STYLES: frozenset[str] = frozenset(
    {
        "raw",
        "filtered_raw",
        "extractive",
        "selected_raw",
        "selected_extractive",
        "structured",
        "structured_quotes",
        "structured_generic",
        "guarded_structured",
        "sieve_v1",
    }
)


def get_packager(style: str) -> Packager:
    """Return the packager for *style*, or raise ``KeyError`` with the options."""
    try:
        return STYLE_REGISTRY[style]
    except KeyError as exc:
        raise KeyError(
            f"Unknown evidence style {style!r}. "
            f"Known styles: {sorted(STYLE_REGISTRY)}."
        ) from exc


__all__ = [
    "Packager",
    "PackagerContext",
    "STYLE_REGISTRY",
    "OFFLINE_STYLES",
    "get_packager",
    "RawPackager",
    "FilteredRawPackager",
    "ExtractivePackager",
    "SelectedRawPackager",
    "SelectedExtractivePackager",
    "StructuredPackager",
    "StructuredQuotesPackager",
    "StructuredGenericPackager",
    "GuardedStructuredPackager",
    "SummaryPackager",
    "SieveV1Packager",
]
