"""Decompose the format-lever result: is it a universal lever or a model trait?

within-reader swing = max_format - min_format accuracy (format sensitivity).
universal component = spread of format means averaged across readers.
crossover = do readers disagree on their best format (model trait)?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.format_lever import FORMATS


def _pct(cell: dict) -> float:
    return 100.0 * cell["correct"] / cell["n"] if cell["n"] else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Format-lever report")
    parser.add_argument("--cells", type=Path, required=True)
    args = parser.parse_args()
    cells = json.loads(args.cells.read_text())

    readers = sorted(cells)
    header = "reader".ljust(28) + "".join(f"{f:>10s}" for f in FORMATS) + f"{'swing':>8s}{'best':>10s}"
    print(header)
    best_by_reader = {}
    for reader in readers:
        accs = {f: _pct(cells[reader].get(f, {"n": 0, "correct": 0})) for f in FORMATS}
        best = max(accs, key=accs.get)
        best_by_reader[reader] = best
        swing = max(accs.values()) - min(accs.values())
        row = reader.ljust(28) + "".join(f"{accs[f]:>10.1f}" for f in FORMATS)
        print(row + f"{swing:>8.1f}{best:>10s}")

    fmt_means = {f: sum(_pct(cells[r].get(f, {"n": 0, "correct": 0})) for r in readers) / len(readers) for f in FORMATS}
    universal_spread = max(fmt_means.values()) - min(fmt_means.values())
    distinct_best = set(best_by_reader.values())
    mean_swing = sum(
        max(_pct(cells[r].get(f, {"n": 0, "correct": 0})) for f in FORMATS)
        - min(_pct(cells[r].get(f, {"n": 0, "correct": 0})) for f in FORMATS)
        for r in readers
    ) / len(readers)

    print("\n-- decomposition --")
    print("format means (avg over readers):", {f: round(v, 1) for f, v in fmt_means.items()})
    print(f"universal spread (best vs worst format, reader-avg): {universal_spread:.1f}pp")
    print(f"mean within-reader swing: {mean_swing:.1f}pp")
    print(f"distinct best-formats across readers: {sorted(distinct_best)}")
    print(
        "VERDICT: "
        + (
            "crossover / model-trait (readers disagree on best format)"
            if len(distinct_best) > 1
            else "universal lever (all readers share one best format)"
        )
        + f"; effect is {'LARGE' if mean_swing >= 8 else 'small' if mean_swing >= 3 else 'FLAT'}"
    )


if __name__ == "__main__":
    main()
