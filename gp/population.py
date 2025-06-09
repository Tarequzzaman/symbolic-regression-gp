from __future__ import annotations  # Allow forward references in type hints (Python 3.7+)

import random                      # Random operations for selection, mutation, etc.
import time                        # (Optional: could be used for timing runs)
from typing import List, Tuple     # Type annotations for lists and tuples

from .data import load_default_dataset               # Function to load default training data
from .fitness import mean_squared_error              # MSE evaluation function
from .nodes import Node, generate_tree               # Node class and tree generator
from .operators import crossover, subtree_mutation, tournament_selection  # GP operators
from .visualisation import plot_fitness              # Plotting utility to show MSE trend


# Default GP hyper‑parameters (tweak via CLI args or kwargs in run_gp)
DEFAULTS = {
    "pop_size": 100,               # Number of individuals in population
    "max_depth": 6,                # Max tree depth allowed
    "tournament_k": 5,             # Tournament selection size
    "crossover_rate": 0.9,         # Probability of crossover
    "mutation_rate": 0.1,          # Probability of mutation
    "generations": 50,             # Number of generations to run
    "elitism": 1,                  # Number of top individuals to preserve
    "mse_threshold": 0.1,          # Stop early if this MSE or better is achieved
}


def generate_individual(max_depth: int = 6, full: bool = False) -> Node:
    """Return a brand-new random GP tree (wrapper for rubric)."""
    return generate_tree(max_depth, full=full)  # Use grow/full method to generate individual


def evaluate_individual(ind: Node, dataset) -> float:
    """Evaluate *ind* on *dataset* and return its MSE (wrapper for rubric)."""
    return mean_squared_error(ind, dataset)  # Compute MSE over dataset


def calculate_mse(ind: Node, dataset) -> float:
    """Calculate MSE of *ind* on *dataset* (wrapper for rubric)."""
    return evaluate_individual(ind, dataset)  # Alias for clarity


def _initial_population(size: int, max_depth: int) -> List[Node]:  # noqa: D401
    population = []                                                # List to hold population
    for i in range(size):                                          # For each individual
        # ramped half‑and‑half: alternate between full and grow trees, depths 2‑max_depth
        depth = 2 + (i % (max_depth - 1))                          # Cycle depth between 2 and max_depth
        full = (i // (max_depth - 1)) % 2 == 0                     # Alternate between grow and full method
        population.append(generate_individual(depth, full=full))  # Create and add individual
    return population                                              # Return initial population


def run_gp(**kwargs):  # noqa: D401
    """Run Genetic Programming and return (best_tree, history list)."""
    cfg = {**DEFAULTS, **kwargs}                                   # Merge default config with user config
    dataset = kwargs.get("dataset") or load_default_dataset()     # Use provided dataset or load default

    pop = _initial_population(cfg["pop_size"], cfg["max_depth"])  # Generate initial population
    fitnesses = [evaluate_individual(ind, dataset) for ind in pop]  # Evaluate each individual

    history = [min(fitnesses)]                                     # Track best fitness per generation
    best, best_fit = min(zip(pop, fitnesses), key=lambda t: t[1])  # Store best individual so far

    for gen in range(1, cfg["generations"] + 1):                   # For each generation
        new_pop: List[Node] = []                                   # New generation container

        # 1. Elitism (carry bests forward)
        elite_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[: cfg["elitism"]]
        for idx in elite_idx:
            new_pop.append(pop[idx].clone())                       # Copy top individuals directly

        # 2. Generate rest of population
        while len(new_pop) < cfg["pop_size"]:                      # Until new_pop is filled
            if random.random() < cfg["crossover_rate"]:            # Crossover branch
                p1 = tournament_selection(list(zip(pop, fitnesses)), k=cfg["tournament_k"])  # Select parent1
                p2 = tournament_selection(list(zip(pop, fitnesses)), k=cfg["tournament_k"])  # Select parent2
                c1, c2 = crossover(p1, p2, cfg["max_depth"])       # Perform subtree crossover
                new_pop.extend([c1, c2])                            # Add both offspring
            else:                                                  # Mutation branch
                p = tournament_selection(list(zip(pop, fitnesses)), k=cfg["tournament_k"])  # Select one parent
                m = subtree_mutation(p, cfg["max_depth"])          # Apply subtree mutation
                new_pop.append(m)                                  # Add mutant

        pop = new_pop[: cfg["pop_size"]]                           # Trim to exact pop size if exceeded
        fitnesses = [evaluate_individual(ind, dataset) for ind in pop]  # Recalculate fitnesses

        gen_best, gen_best_fit = min(zip(pop, fitnesses), key=lambda t: t[1])  # Best of this generation
        if gen_best_fit < best_fit:                               # If new best found
            best, best_fit = gen_best.clone(), gen_best_fit       # Update best tracker
        history.append(best_fit)                                  # Add to fitness history

        print(f"Gen {gen:03d}: best MSE = {best_fit:.5f}")        # Report progress
        if best_fit <= cfg["mse_threshold"]:                      # Early stopping check
            print("Early stop: fitness threshold reached")
            break

    # plot fitness curve
    plot_fitness(history)                                         # Show MSE curve over generations
    return best, history                                          # Return best solution and fitness history
