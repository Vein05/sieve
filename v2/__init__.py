"""SIEVE v2: reader-adaptive evidence compilation (response-surface phase).

This package is additive over the frozen v1 pipeline. It builds a
counterfactual evidence lattice (styles x budgets) per query, runs every
reader against every variant, and maps the (reader, style, budget) ->
accuracy/cost response surface described in ``research/idea-v2.md``.

Phase 1 (this scaffold) delivers the packagers, the budget enforcement,
the variant cache, and the compile-once / replay-many runner. Phase 2
stubs (calibration probes, capability profiles, learned policy,
cross-encoder scorer) expose real interfaces with clearly marked
placeholder internals for the owner to author.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
