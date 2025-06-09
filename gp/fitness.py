from __future__ import annotations  # Enables forward references in type hints (Python 3.7+)
from typing import Iterable, List, Tuple  # Import type hints for iterables and tuples
import math                              # For math.inf
import numpy as np                       # For vectorized operations with arrays
from .nodes import Node                  # Import the Node class (base of GP trees)


def mean_squared_error(ind: Node, dataset: Iterable[Tuple[float, float, float]]) -> float:  # noqa: D401
    # Calculate MSE of the individual's predictions over the dataset
    sq_err = 0.0                                   # Initialize sum of squared errors
    for x, y, target in dataset:                   # Loop over each data point (x, y, true output)
        try:
            pred = ind.evaluate(x, y)              # Evaluate GP tree on (x, y)
        except (OverflowError, ZeroDivisionError, ValueError):  # Catch bad tree evaluations
            return math.inf                        # Return infinity to penalize invalid trees
        sq_err += (pred - target) ** 2             # Accumulate squared error
    return sq_err / len(list(dataset))             # Return mean squared error


# Vectorised fast version (numpy), optional
def mse_numpy(ind: Node, xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> float:  # noqa: D401
    # Calculate MSE using NumPy arrays for efficiency
    preds = np.fromiter(                           # Generate predictions from iterator
        (ind.evaluate(x, y) for x, y in zip(xs, ys)),  # Pair x, y inputs and evaluate the tree
        dtype=float,                                # Force result to float
        count=len(xs)                               # Number of expected predictions
    )
    return float(((preds - ts) ** 2).mean())        # Compute mean squared error using NumPy
