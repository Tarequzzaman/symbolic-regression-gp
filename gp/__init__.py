"""Top‑level re‑exports so users can simply `from gp import run_gp`"""
from .population import run_gp  # noqa: F401
from .data import load_default_dataset  # noqa: F401
from .visualisation import plot_fitness, plot_surface  # noqa: F401

__all__ = [
    "run_gp",
    "load_default_dataset",
    "plot_fitness",
    "plot_surface",
]
