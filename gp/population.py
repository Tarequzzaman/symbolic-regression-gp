from __future__ import annotations

import random
import time
from typing import List, Tuple

from .data import load_default_dataset
from .fitness import mean_squared_error
from .nodes import Node, generate_tree
from .operators import crossover, subtree_mutation, tournament_selection
from .visualisation import plot_fitness


# Default GP hyper‑parameters (tweak via CLI args or kwargs in run_gp)
DEFAULTS = {
    "pop_size": 100,
    "max_depth": 6,
    "tournament_k": 5,
    "crossover_rate": 0.9,
    "mutation_rate": 0.1,
    "generations": 50,
    "elitism": 1,
    "mse_threshold": 0.1,
}


def _initial_population(size: int, max_depth: int) -> List[Node]:  # noqa: D401
    population = []
    for i in range(size):
        # ramped half‑and‑half: alternate between full and grow trees, depths 2‑max_depth
        depth = 2 + (i % (max_depth - 1))
        full = (i // (max_depth - 1)) % 2 == 0
        population.append(generate_tree(depth, full=full))
    return population


def run_gp(**kwargs):  # noqa: D401
    """Run Genetic Programming and return (best_tree, history list)."""
    cfg = {**DEFAULTS, **kwargs}
    dataset = kwargs.get("dataset") or load_default_dataset()

    pop = _initial_population(cfg["pop_size"], cfg["max_depth"])
    fitnesses = [mean_squared_error(ind, dataset) for ind in pop]

    history = [min(fitnesses)]
    best, best_fit = min(zip(pop, fitnesses), key=lambda t: t[1])

    for gen in range(1, cfg["generations"] + 1):
        new_pop: List[Node] = []
        # 1. Elitism (carry bests forward)
        elite_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[: cfg["elitism"]]
        for idx in elite_idx:
            new_pop.append(pop[idx].clone())

        # 2. Generate rest of population
        while len(new_pop) < cfg["pop_size"]:
            if random.random() < cfg["crossover_rate"]:
                p1 = tournament_selection(list(zip(pop, fitnesses)), k=cfg["tournament_k"])
                p2 = tournament_selection(list(zip(pop, fitnesses)), k=cfg["tournament_k"])
                c1, c2 = crossover(p1, p2, cfg["max_depth"])
                new_pop.extend([c1, c2])
            else:
                p = tournament_selection(list(zip(pop, fitnesses)), k=cfg["tournament_k"])
                m = subtree_mutation(p, cfg["max_depth"])
                new_pop.append(m)
        pop = new_pop[: cfg["pop_size"]]  # trim extras
        fitnesses = [mean_squared_error(ind, dataset) for ind in pop]

        gen_best, gen_best_fit = min(zip(pop, fitnesses), key=lambda t: t[1])
        if gen_best_fit < best_fit:
            best, best_fit = gen_best.clone(), gen_best_fit
        history.append(best_fit)

        print(f"Gen {gen:03d}: best MSE = {best_fit:.5f}")
        if best_fit <= cfg["mse_threshold"]:
            print("Early stop: fitness threshold reached")
            break

    # plot fitness curve
    plot_fitness(history)
    return best, history