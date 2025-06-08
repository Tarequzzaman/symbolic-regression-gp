from __future__ import annotations

from typing import List, Tuple

import csv
import pathlib

# Hard‑coded sample (replace with official Table 1 values)
_SAMPLE_DATA = [
    (-1.0, -1.0, 1.0),
    (0.0, 0.0, 0.0),
    (1.0, 2.0, 3.0),
    (2.0, 3.0, 5.0),
]


_DEF_CSV = pathlib.Path(__file__).with_suffix(".csv")  # gp/data.csv in same dir


def load_default_dataset() -> List[Tuple[float, float, float]]:
    """
    Load gp/data.csv (same folder) if it exists; otherwise return the fallback sample.
    Skips one header row if the first cell is not numeric.
    """
    if not _DEF_CSV.exists():
        return _SAMPLE_DATA

    data: List[Tuple[float, float, float]] = []
    with _DEF_CSV.open("r", newline="") as fh:
        rdr = csv.reader(fh)
        for row in rdr:
            # If row[0] is 'x' we’re on the header, so skip it once
            if row and row[0].strip().lower() == "x":
                continue
            data.append((float(row[0]), float(row[1]), float(row[2])))
    return data
