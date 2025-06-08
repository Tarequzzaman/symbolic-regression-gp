from __future__ import annotations

import random
from typing import List, Tuple

from .nodes import Node, TerminalNode, generate_tree
from .utils import pick_random_subtree

# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ---------------------------------------------------------------------------


def tournament_selection(population: List[Tuple[Node, float]],
                         k: int = 3) -> Node:
    """Return a *clone* of the best among *k* randomly-sampled individuals."""
    contenders = random.sample(population, k)
    best = min(contenders, key=lambda t: t[1])   # lower MSE is better
    return best[0].clone()

# ─────────────────────────────────────────────────────────────────────────────
# Crossover
# ---------------------------------------------------------------------------


def crossover(parent1: Node, parent2: Node,
              max_depth: int) -> Tuple[Node, Node]:
    """
    Sub-tree crossover that respects *max_depth*.
    We pick crossover points **after cloning** so we never have to re-find
    the same object identity inside the copy.
    Returns two *new* individuals.
    """
    child1 = parent1.clone()
    child2 = parent2.clone()

    p1, attr1, subtree1 = pick_random_subtree(child1)
    p2, attr2, subtree2 = pick_random_subtree(child2)

    # swap the chosen sub-trees
    if p1 is None:
        child1 = subtree2.clone()
    else:
        p1.children[attr1] = subtree2.clone()

    if p2 is None:
        child2 = subtree1.clone()
    else:
        p2.children[attr2] = subtree1.clone()

    # depth safeguard
    if child1.depth() > max_depth:
        child1 = parent1.clone()
    if child2.depth() > max_depth:
        child2 = parent2.clone()

    return child1, child2

# ─────────────────────────────────────────────────────────────────────────────
# Mutation
# ---------------------------------------------------------------------------


def subtree_mutation(ind: Node,
                     max_depth: int,
                     mutation_depth: int = 2) -> Node:
    """
    Replace a random subtree with a freshly generated subtree.
    Mutation point is chosen on the **clone** to avoid identity issues.
    """
    mutant = ind.clone()
    parent, attr, _ = pick_random_subtree(mutant)
    new_subtree = generate_tree(mutation_depth, full=False)

    if parent is None:
        mutant = new_subtree           # mutated at root
    else:
        parent.children[attr] = new_subtree

    return mutant if mutant.depth() <= max_depth else ind.clone()


# Optional extra: point mutation on numeric terminals
# ---------------------------------------------------------------------------

def point_mutation(ind: Node, sigma: float = 0.3, prob: float = 0.1) -> Node:
    """
    Walk the tree; each numeric constant has *prob* chance of N(0, sigma) jitter.
    """
    mutant = ind.clone()

    def _walk(node: Node):
        if isinstance(node, TerminalNode) and isinstance(node.value, float):
            if random.random() < prob:
                node.value += random.gauss(0, sigma)
        elif hasattr(node, "children"):          # FunctionNode
            for child in node.children:
                _walk(child)

    _walk(mutant)
    return mutant
