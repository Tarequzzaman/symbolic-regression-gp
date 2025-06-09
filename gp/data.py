from __future__ import annotations  # Enables forward references in type hints (Python 3.7+)

from typing import List, Tuple       # For type annotations of lists and tuples

import csv                          # To read CSV data
import pathlib                      # For filesystem path manipulation

# Hard‑coded sample (replace with official Table 1 values)
_SAMPLE_DATA = [                    # Fallback dataset if CSV not found
    (-1.0, -1.0, 1.0),              # Example data point (x, y, target)
    (0.0, 0.0, 0.0),
    (1.0, 2.0, 3.0),
    (2.0, 3.0, 5.0),
]

# Path to the default CSV file (same folder, named data.csv)
_DEF_CSV = pathlib.Path(__file__).with_suffix(".csv")  # gp/data.csv in same dir


def load_default_dataset() -> List[Tuple[float, float, float]]:
    """
    Load gp/data.csv (same folder) if it exists; otherwise return the fallback sample.
    Skips one header row if the first cell is not numeric.
    """
    if not _DEF_CSV.exists():       # If the CSV file does not exist
        return _SAMPLE_DATA         # Use built-in sample data as fallback

    data: List[Tuple[float, float, float]] = []  # Initialize dataset container
    with _DEF_CSV.open("r", newline="") as fh:   # Open the CSV file
        rdr = csv.reader(fh)                     # Create a CSV reader
        for row in rdr:                          # Read each row in the file
            # If row[0] is 'x' we’re on the header, so skip it once
            if row and row[0].strip().lower() == "x":  # Skip header if present
                continue
            data.append((float(row[0]), float(row[1]), float(row[2])))  # Parse and store tuple
    return data                                   # Return parsed dataset
