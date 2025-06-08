from __future__ import annotations
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from .nodes import Node
import networkx as nx



plt.rcParams["figure.dpi"] = 120

# ------------------------------------------------------------------
# where every graphic is saved
# ------------------------------------------------------------------
_OUT = "output"
os.makedirs(_OUT, exist_ok=True)

# ------------------------------------------------------------------
# 1) Fitness curve
# ------------------------------------------------------------------
def plot_fitness(history: List[float], *, save: bool = True):
    plt.figure()
    plt.plot(history, linewidth=2)
    plt.title("GP Training Curve (MSE vs Generation)")
    plt.xlabel("Generation")
    plt.ylabel("Best MSE")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(_OUT, "fitness.png"), dpi=150)
    plt.show()


# ------------------------------------------------------------------
# 2) 3-D surface comparison
# ------------------------------------------------------------------
def _lookup_true(x: float, y: float, data):
    """Nearest-neighbour lookup for target z."""
    return min(data, key=lambda t: (t[0]-x)**2 + (t[1]-y)**2)[2]


def plot_surface(ind: Node, dataset: List[Tuple[float, float, float]],
                 *, grid_size: int = 30, save: bool = True):
    xs = np.linspace(min(d[0] for d in dataset), max(d[0] for d in dataset), grid_size)
    ys = np.linspace(min(d[1] for d in dataset), max(d[1] for d in dataset), grid_size)
    xx, yy = np.meshgrid(xs, ys)
    zz_pred = np.vectorize(lambda a, b: ind.evaluate(float(a), float(b)))(xx, yy)
    zz_true = np.vectorize(lambda a, b: _lookup_true(a, b, dataset))(xx, yy)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(xx, yy, zz_pred, alpha=0.9)
    ax1.set_title("GP-Predicted Surface")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("f(x,y)")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(xx, yy, zz_true, alpha=0.9)
    ax2.set_title("True Surface")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("target")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(_OUT, "surface.png"), dpi=150)
    plt.show()


# ------------------------------------------------------------------
# 3) Flow-chart of GP life-cycle
# ------------------------------------------------------------------
def save_flowchart(path: str = os.path.join(_OUT, "flowchart_gp.png")):
    """
    Render a correct Genetic Programming (GP) flowchart and save it to *path*.
    Produces:
        Initialise → Evaluate → Select → Crossover/Mutation
                        ↑                           ↘
                  Replace ←── Early-Stop? ←──────────┘
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    ax.set_title("Genetic Programming Flowchart", fontsize=14, fontweight='bold', pad=20)


    # ── Box layout positions ─────────────────────────────────────────────
    nodes = {
        "init":    ("Initialise\nPopulation", (0.08, 0.6)),
        "eval":    ("Evaluate\nFitness",      (0.32, 0.6)),
        "select":  ("Select\nParents",        (0.56, 0.6)),
        "xover":   ("Crossover /\nMutation",  (0.80, 0.6)),
        "replace": ("Replace &\nElitism",     (0.32, 0.25)),
        "stop":    ("Early-Stop?\nMSE ≤ 0.1", (0.56, 0.25)),
    }

    # ── Draw boxes ───────────────────────────────────────────────────────
    for label, (text, (x, y)) in nodes.items():
        ax.text(x, y, text, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="#d0e1ff"))

    arrow = dict(arrowstyle="->", lw=1.4)

    # ── Top row arrows ───────────────────────────────────────────────────
    ax.annotate("", (0.24, 0.60), (0.16, 0.60), arrowprops=arrow)  # init → eval
    ax.annotate("", (0.48, 0.60), (0.40, 0.60), arrowprops=arrow)  # eval → select
    ax.annotate("", (0.72, 0.60), (0.64, 0.60), arrowprops=arrow)  # select → xover

    # ── Vertical arrows ─────────────────────────────────────────────────
    ax.annotate("", (0.32, 0.52), (0.32, 0.40), arrowprops=arrow)  # replace ↑ eval
    ax.annotate("", (0.56, 0.52), (0.56, 0.40), arrowprops=arrow)  # stop   ↑ select

    # ── Diagonal arrow from xover → stop ────────────────────────────────
    ax.annotate("", (0.58, 0.32), (0.80, 0.52), arrowprops=arrow)  # xover → stop

    # ── Bottom arrow (Replace → Stop) ───────────────────────────────────
    ax.annotate("", (0.50, 0.28), (0.38, 0.28), arrowprops=arrow)  # replace → stop

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()




# ------------------------------------------------------------------
# 4) Sub-tree crossover example
# ------------------------------------------------------------------
def _plot_tree_nx(node, uid="n0", pos=None, G=None):
    if G is None:
        G = nx.DiGraph(); pos = {}
    label = str(node)[:8]
    G.add_node(uid, label=label)
    pos[uid] = (len(pos), -node.depth())
    if hasattr(node, "children"):
        for i, child in enumerate(node.children):
            cid = f"{uid}_{i}"
            G.add_edge(uid, cid)
            _plot_tree_nx(child, cid, pos, G)
    return G, pos


def save_crossover_plt(p1, p2, c1, c2,
                           path: str = os.path.join(_OUT, "crossover.png")):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    titles = [("Parent 1", p1), ("Parent 2", p2),
              ("Child 1",  c1), ("Child 2",  c2)]
    for ax, (title, tree) in zip(axes.ravel(), titles):
        G, pos = _plot_tree_nx(tree)
        nx.draw(G, pos, ax=ax, arrows=False, with_labels=False,
                node_size=500, node_color="#e8ffe8", font_size=6)
        ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()


# ------------------------------------------------------------------
# 5) Graphviz tree exporter
# ------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def save_tree_graphviz(ind: Node, filename: str = "gp_tree"):
    dot = Digraph(comment="GP Tree", format="png")
    dot.attr("node", shape="circle", fontname="Helvetica", fontsize="10")

    def _label(node):
        return str(getattr(node, "value", getattr(node, "func_token", "?")))

    def _walk(node, uid: str):
        dot.node(uid, _label(node))
        if hasattr(node, "children"):
            for i, ch in enumerate(node.children):
                cid = f"{uid}_{i}"
                dot.edge(uid, cid)
                _walk(ch, cid)

    _walk(ind, "n0")

    # Save to file
    gv_path = os.path.join(_OUT, f"{filename}")
    dot.render(gv_path, view=False)  # Will save as .gv and .gv.png

    # Display the PNG
    img_path = f"{gv_path}.png"
    img = mpimg.imread(img_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Genetic Program Tree")
    plt.show()

    print(f"Saved Graphviz tree to {gv_path}.gv and .gv.png")


