from __future__ import annotations

from typing import List, Tuple

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from graphviz import Digraph
import os

import numpy as np

from .nodes import Node


plt.rcParams["figure.dpi"] = 120  # higher‑res by default



def plot_fitness(history: List[float]):  # noqa: D401
    plt.figure()
    plt.plot(history, linewidth=2)
    plt.title("GP Training Curve (MSE vs Generation)")
    plt.xlabel("Generation")
    plt.ylabel("Best MSE")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()



def plot_surface(ind: Node, dataset: List[Tuple[float, float, float]], *, grid_size: int = 30):  # noqa: D401
    """Plot a 3D surface of the GP's predictions vs. the dataset targets."""
   
    xs = np.linspace(min(d[0] for d in dataset), max(d[0] for d in dataset), grid_size)
    ys = np.linspace(min(d[1] for d in dataset), max(d[1] for d in dataset), grid_size)
    xx, yy = np.meshgrid(xs, ys)
    zz_pred = np.vectorize(lambda a, b: ind.evaluate(float(a), float(b)))(xx, yy)
    zz_true = np.vectorize(lambda a, b: _lookup_true(a, b, dataset))(xx, yy)


    fig = plt.figure(figsize=(10, 4))
    # predicted
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(xx, yy, zz_pred, alpha=0.9)
    ax1.set_title("GP‑Predicted Surface")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x, y)")
    # true
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(xx, yy, zz_true, alpha=0.9)
    ax2.set_title("True Surface (dataset interp.)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("target")
    plt.tight_layout()
    plt.show()


def _lookup_true(x: float, y: float, data):  # small helper for interp
    # nearest neighbour for simplicity
    nearest = min(data, key=lambda t: (t[0] - x) ** 2 + (t[1] - y) ** 2)
    return nearest[2]



def save_tree_graphviz(ind, filename: str = "gp_tree", out_dir: str = "output"):
    """Save a GP tree as a Graphviz .gv file and render it to PNG."""

    dot = Digraph(comment="GP Tree", format="png")
    dot.attr("node", shape="circle", fontname="Helvetica", fontsize="10")

    def _node_label(node):
        if hasattr(node, "value"):
            return str(node.value)
        for attr in ("op", "operator", "func", "symbol"):
            if hasattr(node, attr):
                return str(getattr(node, attr))
        return node.__class__.__name__

    def _children(node):
        # works for either .children list *or* .left/.right pair
        if hasattr(node, "children"):
            return list(node.children)
        if hasattr(node, "left") and hasattr(node, "right"):
            return [node.left, node.right]
        return []

    # depth-first traversal
    def _walk(node, uid):
        dot.node(uid, _node_label(node))
        for idx, child in enumerate(_children(node)):
            cid = f"{uid}_{idx}"
            dot.edge(uid, cid)
            _walk(child, cid)

    _walk(ind, "n0")

    os.makedirs(out_dir, exist_ok=True)
    gv_path = os.path.join(out_dir, f"{filename}.gv")
    png_path = dot.render(gv_path, view=False)
    print(f"Graphviz source saved to {gv_path}")
    print(f"Rendered PNG saved to  {png_path}")
