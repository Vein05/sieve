"""Turn WikiContradict transfer cells.json into the decision-relevant deltas.

The paper question is one comparison: does deterministic conflict-marked
rendering (marked+standard) beat conflict-aware prompting (plain+aware), and do
they compose (marked+aware)? This reports that per reader and pooled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASELINE = "plain+standard"
PROMPTING = "plain+aware"
RENDERING = "marked+standard"
BOTH = "marked+aware"
CONDITION_ORDER = (BASELINE, PROMPTING, RENDERING, BOTH)
METRICS = ("det_correct", "both_present")


def _pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def _by_reader(cells: dict) -> dict[str, dict[str, dict]]:
    readers: dict[str, dict[str, dict]] = {}
    for key, cell in cells.items():
        condition, reader = key.split("|", 1)
        readers.setdefault(reader, {})[condition] = cell
    return readers


def _pooled(cells: dict) -> dict[str, dict]:
    pooled: dict[str, dict] = {}
    for key, cell in cells.items():
        condition = key.split("|", 1)[0]
        agg = pooled.setdefault(condition, {"n": 0, "det_correct": 0, "both_present": 0})
        for field in ("n", "det_correct", "both_present"):
            agg[field] += cell[field]
    return pooled


def _print_block(title: str, conditions: dict[str, dict]) -> None:
    print(f"\n== {title} ==")
    for metric in METRICS:
        print(f"  [{metric}]")
        for condition in CONDITION_ORDER:
            cell = conditions.get(condition)
            if not cell:
                continue
            print(f"    {condition:16s} {_pct(cell[metric], cell['n']):5.1f}%  ({cell[metric]}/{cell['n']})")
        rendering = conditions.get(RENDERING)
        prompting = conditions.get(PROMPTING)
        both = conditions.get(BOTH)
        if rendering and prompting:
            delta = _pct(rendering[metric], rendering["n"]) - _pct(prompting[metric], prompting["n"])
            print(f"    -> rendering vs prompting: {delta:+.1f}pp (the decision)")
        if both and prompting:
            delta = _pct(both[metric], both["n"]) - _pct(prompting[metric], prompting["n"])
            print(f"    -> compose(both) vs prompting: {delta:+.1f}pp")


def main() -> None:
    parser = argparse.ArgumentParser(description="WikiContradict transfer report")
    parser.add_argument("--cells", type=Path, required=True)
    args = parser.parse_args()

    cells = json.loads(args.cells.read_text())
    for reader, conditions in sorted(_by_reader(cells).items()):
        _print_block(reader, conditions)
    _print_block("POOLED (all readers)", _pooled(cells))


if __name__ == "__main__":
    main()
