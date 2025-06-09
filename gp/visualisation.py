from __future__ import annotations  # Enable forward references in type hints

import os                          # For file/directory operations
from typing import List, Tuple    # For list/tuple type annotations
import matplotlib.pyplot as plt   # For plotting graphs
import numpy as np                # For numerical computations
from graphviz import Digraph      # For rendering expression trees
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D plotting (required for 3D axes)
from .nodes import Node           # Import Node class representing GP trees
import networkx as nx             # For plotting trees using networkx
import matplotlib.pyplot as plt       # Re-import to support image display
import matplotlib.image as mpimg      # For displaying saved PNG image



plt.rcParams["figure.dpi"] = 120  # Set default DPI for better plot resolution

_OUT = "output"                   # Directory to save plots
os.makedirs(_OUT, exist_ok=True) # Create output directory if it doesn't exist


def plot_fitness(history: List[float], *, save: bool = True):
    plt.figure()                                        # Create new figure
    plt.plot(history, linewidth=2)                      # Plot fitness curve
    plt.title("GP Training Curve (MSE vs Generation)")  # Set plot title
    plt.xlabel("Generation")                            # X-axis label
    plt.ylabel("Best MSE")                              # Y-axis label
    plt.grid(True, linestyle=":")                       # Enable grid
    plt.tight_layout()                                  # Adjust spacing
    if save:
        plt.savefig(os.path.join(_OUT, "fitness.png"), dpi=150)  # Save figure
    plt.show()                                          # Display the plot


def _lookup_true(x: float, y: float, data):
    """Nearest-neighbour lookup for target z."""
    return min(data, key=lambda t: (t[0]-x)**2 + (t[1]-y)**2)[2]  # Return z value closest to (x,y)


def plot_surface(ind: Node, dataset: List[Tuple[float, float, float]],
                 *, grid_size: int = 30, save: bool = True):
    xs = np.linspace(min(d[0] for d in dataset), max(d[0] for d in dataset), grid_size)  # X-axis range
    ys = np.linspace(min(d[1] for d in dataset), max(d[1] for d in dataset), grid_size)  # Y-axis range
    xx, yy = np.meshgrid(xs, ys)                                                         # Create grid
    zz_pred = np.vectorize(lambda a, b: ind.evaluate(float(a), float(b)))(xx, yy)        # GP prediction
    zz_true = np.vectorize(lambda a, b: _lookup_true(a, b, dataset))(xx, yy)             # Ground truth

    fig = plt.figure(figsize=(10, 4))                  # Create new figure
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")     # Left subplot: GP surface
    ax1.plot_surface(xx, yy, zz_pred, alpha=0.9)        # Plot predicted surface
    ax1.set_title("GP-Predicted Surface")               # Title
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("f(x,y)")  # Axis labels

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")     # Right subplot: True surface
    ax2.plot_surface(xx, yy, zz_true, alpha=0.9)        # Plot true surface
    ax2.set_title("True Surface")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("target")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(_OUT, "surface.png"), dpi=150)  # Save the surface comparison
    plt.show()                                                   # Show plot


def save_flowchart(path: str = os.path.join(_OUT, "flowchart_gp.png")):
    """
    Render a correct Genetic Programming (GP) flowchart and save it to *path*.
    Produces:
        Initialise → Evaluate → Select → Crossover/Mutation
                        ↑                           ↘
                  Replace ←── Early-Stop? ←──────────┘
    """
    fig, ax = plt.subplots(figsize=(9, 4))                       # Create figure and axis
    ax.axis("off")                                               # Hide axes
    ax.set_title("Genetic Programming Flowchart", fontsize=14, fontweight='bold', pad=20)

    # Define node labels and positions
    nodes = {
        "init":    ("Initialise\nPopulation", (0.08, 0.6)),
        "eval":    ("Evaluate\nFitness",      (0.32, 0.6)),
        "select":  ("Select\nParents",        (0.56, 0.6)),
        "xover":   ("Crossover /\nMutation",  (0.80, 0.6)),
        "replace": ("Replace &\nElitism",     (0.32, 0.25)),
        "stop":    ("Early-Stop?\nMSE ≤ 0.1", (0.56, 0.25)),
    }

    # Draw nodes as boxes
    for label, (text, (x, y)) in nodes.items():
        ax.text(x, y, text, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="#d0e1ff"))

    arrow = dict(arrowstyle="->", lw=1.4)  # Arrow style

    # Draw flow arrows between nodes
    ax.annotate("", (0.24, 0.60), (0.16, 0.60), arrowprops=arrow)  # init → eval
    ax.annotate("", (0.48, 0.60), (0.40, 0.60), arrowprops=arrow)  # eval → select
    ax.annotate("", (0.72, 0.60), (0.64, 0.60), arrowprops=arrow)  # select → xover
    ax.annotate("", (0.32, 0.52), (0.32, 0.40), arrowprops=arrow)  # replace ↑ eval
    ax.annotate("", (0.56, 0.52), (0.56, 0.40), arrowprops=arrow)  # stop ↑ select
    ax.annotate("", (0.58, 0.32), (0.80, 0.52), arrowprops=arrow)  # xover → stop
    ax.annotate("", (0.50, 0.28), (0.38, 0.28), arrowprops=arrow)  # replace → stop

    plt.tight_layout()
    plt.savefig(path, dpi=150)  # Save image to file
    plt.show()
    plt.close()                 # Close figure


def _plot_tree_nx(node, uid="n0", pos=None, G=None):
    if G is None:
        G = nx.DiGraph(); pos = {}               # Initialize graph and position dictionary
    label = str(node)[:8]                        # Node label (truncated)
    G.add_node(uid, label=label)                 # Add node to graph
    pos[uid] = (len(pos), -node.depth())         # Position node vertically by depth
    if hasattr(node, "children"):                # Recurse if node has children
        for i, child in enumerate(node.children):
            cid = f"{uid}_{i}"                   # Unique child ID
            G.add_edge(uid, cid)                 # Add edge to graph
            _plot_tree_nx(child, cid, pos, G)    # Recurse
    return G, pos                                # Return graph and positions


def save_crossover_plt(p1, p2, c1, c2,
                       path: str = os.path.join(_OUT, "crossover.png")):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))               # Create 2x2 subplot
    titles = [("Parent 1", p1), ("Parent 2", p2), ("Child 1", c1), ("Child 2", c2)]
    for ax, (title, tree) in zip(axes.ravel(), titles):           # Plot each tree
        G, pos = _plot_tree_nx(tree)                              # Get graph layout
        nx.draw(G, pos, ax=ax, arrows=False, with_labels=False,   # Draw graph
                node_size=500, node_color="#e8ffe8", font_size=6)
        ax.set_title(title); ax.axis("off")                      # Set title and hide axes
    plt.tight_layout()
    plt.savefig(path, dpi=150)                                   # Save figure
    plt.show()
    plt.close()                                                  # Close plot




def save_tree_graphviz(ind: Node, filename: str = "gp_tree"):
    dot = Digraph(comment="GP Tree", format="png")                    # Create Graphviz Digraph
    dot.attr("node", shape="circle", fontname="Helvetica", fontsize="10")  # Node style

    def _label(node):                                                 # Get node label
        return str(getattr(node, "value", getattr(node, "func_token", "?")))

    def _walk(node, uid: str):                                        # Recursively add nodes/edges
        dot.node(uid, _label(node))                                   # Add current node
        if hasattr(node, "children"):                                 # Add children if any
            for i, ch in enumerate(node.children):
                cid = f"{uid}_{i}"
                dot.edge(uid, cid)                                    # Add edge
                _walk(ch, cid)                                        # Recurse

    _walk(ind, "n0")                                                  # Start tree traversal

    gv_path = os.path.join(_OUT, f"{filename}")                       # File output path
    dot.render(gv_path, view=False)                                   # Render graph and save .png/.gv

    img_path = f"{gv_path}.png"                                       # Image path
    img = mpimg.imread(img_path)                                     # Read image from file
    plt.figure(figsize=(8, 8))
    plt.imshow(img)                                                  # Show image
    plt.axis('off')                                                  # Hide axes
    plt.title("Genetic Program Tree")                                # Set title
    plt.show()

    print(f"Saved Graphviz tree to {gv_path}.gv and .gv.png")        # Log file output
