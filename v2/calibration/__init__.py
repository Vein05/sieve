"""Reader capability calibration (Phase-2 stubs with real interfaces).

The capability profile ``z_r`` (``research/idea-v2.md``) is derived from a small
probe battery over six dimensions, not from model IDs. This subpackage exposes
the real interfaces (``ProbeSpec``, ``ReaderProfile``, ``compute_profile``,
``load_probe_battery``) so Phase 1 code can depend on stable signatures, while
the probe *content* and scoring internals are left for the owner to author.
"""

from __future__ import annotations

from v2.calibration.probes import PROBE_DIMENSIONS, ProbeSpec, load_probe_battery
from v2.calibration.profile import (
    ReaderProfile,
    compute_profile,
    load_profiles,
    save_profiles,
)

__all__ = [
    "PROBE_DIMENSIONS",
    "ProbeSpec",
    "load_probe_battery",
    "ReaderProfile",
    "compute_profile",
    "load_profiles",
    "save_profiles",
]
