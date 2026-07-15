"""Calibration probe battery (Phase-2 stub with real interfaces).

The capability profile z_r is measured over six dimensions (research/idea-v2.md):

* ``distractor_tolerance``     - robustness to irrelevant retrieved turns
* ``multi_hop``               - multi-hop integration ability
* ``temporal``               - temporal reasoning (implicit vs explicit dates)
* ``paraphrase_robustness``   - robustness to paraphrased vs quoted evidence
* ``context_depth``          - context-utilisation depth
* ``abstention_calibration``  - correctly abstaining when evidence is absent

Each probe is a small QA item with a prompt template and a named scoring fn.
The battery content is intentionally not invented here. The probe-authoring
step remains explicit, and :func:`load_probe_battery` returns an empty battery.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

PROBE_DIMENSIONS: tuple[str, ...] = (
    "distractor_tolerance",
    "multi_hop",
    "temporal",
    "paraphrase_robustness",
    "context_depth",
    "abstention_calibration",
)


@dataclass(frozen=True)
class ProbeSpec:
    """One calibration probe.

    Attributes:
        probe_id: stable identifier.
        dimension: one of :data:`PROBE_DIMENSIONS`.
        prompt_template: a format string with ``{question}`` / ``{evidence}``
            placeholders, rendered at calibration time.
        scoring_fn_name: name of the scoring function (resolved by the owner's
            calibration harness) that maps a reader answer -> [0, 1].
        reference: expected answer / rubric for the scoring fn.
        meta: free-form metadata.
    """

    probe_id: str
    dimension: str
    prompt_template: str
    scoring_fn_name: str
    reference: str = ""
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dimension not in PROBE_DIMENSIONS:
            raise ValueError(
                f"ProbeSpec.dimension must be one of {PROBE_DIMENSIONS}; "
                f"got {self.dimension!r}."
            )


def load_probe_battery() -> list[ProbeSpec]:
    """Return the calibration probe battery.

    Phase-2 stub: returns an empty battery. The six-dimension probe *content*
    is authored by the owner (~50-100 items across PROBE_DIMENSIONS); this
    function is the stable load point the calibration harness will call.

    Populate this from an authored JSON asset under ``v2/configs/``; synthetic
    probe content should not be invented inside the loader.
    """
    return []
