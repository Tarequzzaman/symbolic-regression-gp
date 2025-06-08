from __future__ import annotations

import math
import operator
import random
from typing import Callable, List, Sequence, Tuple, Union, Any

# ─────────────────────────────────────────────────────────────────────────────
# Helper: protected division (avoid division‑by‑zero explosions)


def _protected_div(x: float, y: float) -> float:  # noqa: D401
    """Return x / y but fallback to 1.0 if |y| < 1e‑6."""
    return x / y if abs(y) > 1e-6 else 1.0


# Map token → (callable, arity)
PRIMITIVES: dict[str, Tuple[Callable[..., float], int]] = {
    "+": (operator.add, 2),
    "‑": (operator.sub, 2),
    "*": (operator.mul, 2),
    "/": (_protected_div, 2),
}

TERMINALS = ("x", "y")  # constants are injected on‑the‑fly


class Node: 
    """Base class for all nodes in the expression tree."""

    
    def evaluate(self, x: float, y: float) -> float:  # noqa: D401
        raise NotImplementedError

    def clone(self) -> "Node":  # noqa: D401
        raise NotImplementedError

    def iter_subtrees(self):  # noqa: D401
        """Yield (parent, attribute_name, subtree) so we can swap later."""
        yield None, None, self  # root

    def depth(self) -> int:  # noqa: D401
        raise NotImplementedError

    def __str__(self):  # noqa: D401
        raise NotImplementedError


class FunctionNode(Node):  # noqa: D101
    def __init__(self, func_token: str, children: Sequence[Node]):
        """Create a function node with the given token and children."""

        self.func_token = func_token
        self.func, self.arity = PRIMITIVES[func_token]
        assert len(children) == self.arity, "Arity mismatch"
        self.children: List[Node] = list(children)

    def evaluate(self, x: float, y: float) -> float:  # noqa: D401
        """Evaluate the function node with given x and y values."""

        args = [child.evaluate(x, y) for child in self.children]
        return self.func(*args)

    def clone(self) -> "FunctionNode":
        """Return a deep clone of this function node."""
        return FunctionNode(self.func_token, [c.clone() for c in self.children])

    def depth(self) -> int:
        """Return the depth of this node in the tree."""
        return 1 + max(c.depth() for c in self.children)

    def iter_subtrees(self):
        """Yield (parent, index_in_parent, subtree) for each subtree."""
        for child_idx, child in enumerate(self.children):
            yield self, child_idx, child
            yield from child.iter_subtrees()

    def __str__(self):
        """Return a string representation of this function node."""
        if self.arity == 1:
            return f"{self.func_token}({self.children[0]})"
        left, right = self.children
        return f"({left} {self.func_token} {right})"


class TerminalNode(Node): 
    """A terminal node representing a variable or constant in the expression tree."""

    def __init__(self, value: Union[str, float]):
        """Create a terminal node with the given value."""
        self.value = value  # either "x", "y" or a constant

    def evaluate(self, x: float, y: float) -> float:  
        """Evaluate the terminal node with given x and y values."""
        if self.value == "x":
            return x
        if self.value == "y":
            return y
        return float(self.value)

    def clone(self) -> "TerminalNode":
        """Return a deep clone of this terminal node."""
        return TerminalNode(self.value)

    def depth(self) -> int:
        """Return the depth of this terminal node in the tree."""
        return 0

    def iter_subtrees(self):
        yield None, None, self

    def __str__(self):
        if isinstance(self.value, float):
            return f"{self.value:.3g}"
        return str(self.value)


# ─────────────────────────────────────────────────────────────────────────────
# Random tree generators


def _random_terminal(constant_range: Tuple[float, float]) -> TerminalNode:
    """Return a random terminal node, either a variable or a constant."""
    if random.random() < 0.5:  # 50 %: variable
        return TerminalNode(random.choice(TERMINALS))
    # else constant
    lo, hi = constant_range
    return TerminalNode(random.uniform(lo, hi))


def generate_tree(max_depth: int, *, full: bool, constant_range=(-5.0, 5.0)) -> Node:  # noqa: D401
    """Return a random expression tree via *grow* or *full* method (GP classics)."""
    if max_depth == 0:
        return _random_terminal(constant_range)
    if full:
        func_token = random.choice(list(PRIMITIVES))
        children = [generate_tree(max_depth-1, full=full, constant_range=constant_range) for _ in range(PRIMITIVES[func_token][1])]
        return FunctionNode(func_token, children)
    # grow method
    if random.random() < 0.5:  # choose function
        func_token = random.choice(list(PRIMITIVES))
        children = [generate_tree(max_depth-1, full=False, constant_range=constant_range) for _ in range(PRIMITIVES[func_token][1])]
        return FunctionNode(func_token, children)
    return _random_terminal(constant_range)

