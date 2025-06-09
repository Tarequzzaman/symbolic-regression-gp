from __future__ import annotations  # Postpone evaluation of annotations (for forward references)

import random                       # Used for selection and mutation randomness
from typing import List, Tuple     # Type hints for lists and tuples

from .nodes import Node, TerminalNode, generate_tree  # Import node types and tree generator
from .utils import pick_random_subtree                # Utility to pick random subtree from a tree


def tournament_selection(population: List[Tuple[Node, float]],
                         k: int = 3) -> Node:
    """Return a *clone* of the best among *k* randomly-sampled individuals."""
    contenders = random.sample(population, k)             # Randomly sample k individuals
    best = min(contenders, key=lambda t: t[1])            # Select the one with lowest fitness (MSE)
    return best[0].clone()                                # Return a clone of the best individual's tree


def crossover(parent1: Node, parent2: Node,
              max_depth: int) -> Tuple[Node, Node]:
    """
    Sub-tree crossover that respects *max_depth*.
    We pick crossover points **after cloning** so we never have to re-find
    the same object identity inside the copy.
    Returns two *new* individuals.
    """
    child1 = parent1.clone()                              # Clone parent1 to create first child
    child2 = parent2.clone()                              # Clone parent2 to create second child

    p1, attr1, subtree1 = pick_random_subtree(child1)     # Pick random subtree from child1
    p2, attr2, subtree2 = pick_random_subtree(child2)     # Pick random subtree from child2

    # swap the chosen sub-trees
    if p1 is None:                                        # If root was selected in child1
        child1 = subtree2.clone()                         # Replace entire tree with subtree2
    else:
        p1.children[attr1] = subtree2.clone()             # Replace subtree1 with a clone of subtree2

    if p2 is None:                                        # If root was selected in child2
        child2 = subtree1.clone()                         # Replace entire tree with subtree1
    else:
        p2.children[attr2] = subtree1.clone()             # Replace subtree2 with a clone of subtree1

    # depth safeguard
    if child1.depth() > max_depth:                        # If child1 exceeds depth, revert to original parent1
        child1 = parent1.clone()
    if child2.depth() > max_depth:                        # If child2 exceeds depth, revert to original parent2
        child2 = parent2.clone()

    return child1, child2                                  # Return the two offspring


def subtree_mutation(ind: Node,
                     max_depth: int,
                     mutation_depth: int = 2) -> Node:
    """
    Replace a random subtree with a freshly generated subtree.
    Mutation point is chosen on the **clone** to avoid identity issues.
    """
    mutant = ind.clone()                                  # Clone the individual to avoid modifying original
    parent, attr, _ = pick_random_subtree(mutant)         # Select a random subtree and its parent/position
    new_subtree = generate_tree(mutation_depth, full=False)  # Generate a new random subtree

    if parent is None:                                    # If root node is selected
        mutant = new_subtree                              # Replace entire tree with new subtree
    else:
        parent.children[attr] = new_subtree               # Replace selected subtree in-place

    return mutant if mutant.depth() <= max_depth else ind.clone()  # Return mutant or fallback if too deep


def point_mutation(ind: Node, sigma: float = 0.3, prob: float = 0.1) -> Node:
    """
    Walk the tree; each numeric constant has *prob* chance of N(0, sigma) jitter.
    """
    mutant = ind.clone()                                  # Clone the tree to preserve original

    def _walk(node: Node):                                # Inner recursive function to traverse tree
        if isinstance(node, TerminalNode) and isinstance(node.value, float):  # Only mutate constants
            if random.random() < prob:                    # With probability `prob`
                node.value += random.gauss(0, sigma)      # Add Gaussian noise (mean 0, std sigma)
        elif hasattr(node, "children"):                   # If node is a function node
            for child in node.children:                   # Recurse into children
                _walk(child)

    _walk(mutant)                                         # Begin traversal and mutation
    return mutant                                         # Return mutated tree
