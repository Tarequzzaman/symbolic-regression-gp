from __future__ import annotations

from typing import Iterable, List, Tuple

import math
import numpy as np

from .nodes import Node


def mean_squared_error(ind: Node, dataset: Iterable[Tuple[float, float, float]]) -> float:  # noqa: D401
    sq_err = 0.0
    for x, y, target in dataset:
        try:
            pred = ind.evaluate(x, y)
        except (OverflowError, ZeroDivisionError, ValueError):
            return math.inf  # punish invalid trees immediately
        sq_err += (pred - target) ** 2
    return sq_err / len(list(dataset))


# Vectorised fast version (numpy), optional

def mse_numpy(ind: Node, xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> float:  # noqa: D401
    preds = np.fromiter((ind.evaluate(x, y) for x, y in zip(xs, ys)), dtype=float, count=len(xs))
    return float(((preds - ts) ** 2).mean())
