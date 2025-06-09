from __future__ import annotations  # Enable postponed evaluation of type annotations (Python 3.7+)
import math                          # Math module (not directly used here)
import operator                     # Provides function equivalents for operators
import random                       # Used for stochastic decisions in tree generation
from typing import Callable, List, Sequence, Tuple, Union, Any  # Type hints


def _protected_div(x: float, y: float) -> float:  # noqa: D401
    """Return x / y but fallback to 1.0 if |y| < 1e‑6.""" 
    return x / y if abs(y) > 1e-6 else 1.0  # Avoid division by near-zero


# Map token → (callable, arity)
PRIMITIVES: dict[str, Tuple[Callable[..., float], int]] = {
    "+": (operator.add, 2),          # Addition function with 2 operands
    "-": (operator.sub, 2),          # Subtraction function with 2 operands
    "*": (operator.mul, 2),          # Multiplication function with 2 operands
    "/": (_protected_div, 2),        # Protected division function with 2 operands
}

TERMINALS = ("x", "y")              # Terminal variables; constants added dynamically


class Node:                         # Abstract base class for all expression tree nodes
    """Base class for all nodes in the expression tree."""

    def evaluate(self, x: float, y: float) -> float:  # noqa: D401
        raise NotImplementedError    # Must be implemented by child classes

    def clone(self) -> "Node":       # noqa: D401
        raise NotImplementedError    # Must be implemented to allow deep copying

    def iter_subtrees(self):        # noqa: D401
        """Yield (parent, attribute_name, subtree) so we can swap later."""
        yield None, None, self       # Base case: root node has no parent

    def depth(self) -> int:         # noqa: D401
        raise NotImplementedError    # Must be implemented to calculate tree depth

    def __str__(self):              # noqa: D401
        raise NotImplementedError    # Must be implemented for string conversion


class FunctionNode(Node):           # A node representing an operator with children
    def __init__(self, func_token: str, children: Sequence[Node]):
        """Create a function node with the given token and children."""
        self.func_token = func_token                           # Operator symbol
        self.func, self.arity = PRIMITIVES[func_token]        # Lookup function and its arity
        assert len(children) == self.arity, "Arity mismatch"  # Sanity check
        self.children: List[Node] = list(children)            # Store child nodes

    def evaluate(self, x: float, y: float) -> float:  # noqa: D401
        """Evaluate the function node with given x and y values."""
        args = [child.evaluate(x, y) for child in self.children]  # Recursively evaluate children
        return self.func(*args)                                   # Apply function

    def clone(self) -> "FunctionNode": 
        """Return a deep clone of this function node."""
        return FunctionNode(self.func_token, [c.clone() for c in self.children])  # Recursively clone

    def depth(self) -> int:
        """Return the depth of this node in the tree."""
        return 1 + max(c.depth() for c in self.children)  # Depth = 1 + max depth of children

    def iter_subtrees(self):
        """Yield (parent, index_in_parent, subtree) for each subtree."""
        for child_idx, child in enumerate(self.children):  # Loop through children
            yield self, child_idx, child                   # Yield current child
            yield from child.iter_subtrees()               # Recurse into child

    def __str__(self):
        """Return a string representation of this function node."""
        if self.arity == 1:                               # Unary function
            return f"{self.func_token}({self.children[0]})"
        left, right = self.children                       # Binary function: get both children
        return f"({left} {self.func_token} {right})"      # Infix representation


class TerminalNode(Node):            # A leaf node representing variable or constant
    """A terminal node representing a variable or constant in the expression tree."""

    def __init__(self, value: Union[str, float]):
        """Create a terminal node with the given value."""
        self.value = value         # Can be 'x', 'y', or a numeric constant

    def evaluate(self, x: float, y: float) -> float:
        """Evaluate the terminal node with given x and y values."""
        if self.value == "x": return x       # Return x if variable is 'x'
        if self.value == "y": return y       # Return y if variable is 'y'
        return float(self.value)             # Otherwise treat as constant

    def clone(self) -> "TerminalNode":
        """Return a deep clone of this terminal node."""
        return TerminalNode(self.value)      # Clone by copying value

    def depth(self) -> int:
        """Return the depth of this terminal node in the tree."""
        return 0                             # Leaf nodes have depth 0

    def iter_subtrees(self):
        yield None, None, self               # Terminal nodes yield themselves only

    def __str__(self):
        if isinstance(self.value, float):    # Format float nicely
            return f"{self.value:.3g}"
        return str(self.value)               # Return variable name as string


def _random_terminal(constant_range: Tuple[float, float]) -> TerminalNode:
    """Return a random terminal node, either a variable or a constant."""
    if random.random() < 0.5:                      # 50% chance: return variable
        return TerminalNode(random.choice(TERMINALS))
    lo, hi = constant_range                        # Otherwise generate constant
    return TerminalNode(random.uniform(lo, hi))


def generate_tree(max_depth: int, *, full: bool, constant_range=(-5.0, 5.0)) -> Node:  # noqa: D401
    """Return a random expression tree via *grow* or *full* method (GP classics)."""
    if max_depth == 0:
        return _random_terminal(constant_range)  # Base case: return terminal

    if full:                                     # Full method: always pick functions
        func_token = random.choice(list(PRIMITIVES))  # Choose a function symbol
        children = [
            generate_tree(max_depth-1, full=full, constant_range=constant_range)
            for _ in range(PRIMITIVES[func_token][1])
        ]
        return FunctionNode(func_token, children)  # Return composed node

    # Grow method: pick function or terminal probabilistically
    if random.random() < 0.5:                      # Pick a function node
        func_token = random.choice(list(PRIMITIVES))
        children = [
            generate_tree(max_depth-1, full=False, constant_range=constant_range)
            for _ in range(PRIMITIVES[func_token][1])
        ]
        return FunctionNode(func_token, children)

    return _random_terminal(constant_range)        # Otherwise return terminal
