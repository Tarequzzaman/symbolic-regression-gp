"""Top‑level re‑exports so users can simply `from gp import run_gp`"""

# Re-export the main Genetic Programming run function
from .population import run_gp  # noqa: F401

# Re-export the default dataset loader
from .data import load_default_dataset  # noqa: F401

# Re-export common plotting utilities (fitness over time and function surface)
from .visualisation import plot_fitness, plot_surface  # noqa: F401

# Define public API of the package when using `from gp import *`
__all__ = [
    "run_gp",                # Main GP algorithm runner
    "load_default_dataset",  # Utility to load example training data
    "plot_fitness",          # Visualize fitness over generations
    "plot_surface",          # Visualize function surface for symbolic regression
]
