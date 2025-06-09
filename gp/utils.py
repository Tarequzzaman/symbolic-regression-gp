from __future__ import annotations

import random
from typing import Tuple

from .nodes import Node


# To make experiments reproducible

def seed_everything(seed: int):  
    random.seed(seed)
    # numpy, torch etc. skipped for brevity


# Pick a random subtree from a tree (and optionally locate matching subtree)

def pick_random_subtree(tree: Node, *, target: Node | None = None):  
    """Return (parent, index_in_parent_or_attr, subtree). If *target* is provided
    we return the first match of that subtree in *tree* (used for mapping clones
    back to original nodes)."""

    for parent, attr, subtree in tree.iter_subtrees():
        if target is not None and subtree is not target:
            continue
        if target is None and random.random() > 0.3:  # ~70 % skip to randomise
            continue
        return parent, attr, subtree
    return None, None, tree  # fallback root