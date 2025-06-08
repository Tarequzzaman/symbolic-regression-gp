
"""Convenience entry‑point: `python main.py` runs GP with defaults."""
import argparse
from gp import run_gp, load_default_dataset, plot_surface
from gp.visualisation import save_tree_graphviz

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Run GP symbolic regression")
    parser.add_argument("--pop_size",      type=int,   default=100)
    parser.add_argument("--generations",   type=int,   default=50)
    parser.add_argument("--max_depth",     type=int,   default=6)
    parser.add_argument("--mutation_rate", type=float, default=0.1)
    parser.add_argument("--crossover_rate",type=float, default=0.9)
    parser.add_argument("--dataset",       type=str,   help="Path to CSV dataset")
    args = parser.parse_args()

    kwargs = vars(args)
    # Remove None dataset so run_gp() loads default automatically
    if kwargs["dataset"] is None:
        kwargs.pop("dataset")

    best, history = run_gp(**kwargs)


    print("\nBest evolved expression:")
    print(best)


    dataset = kwargs.get("dataset") or load_default_dataset()
    if len(dataset) > 8:
        plot_surface(best, dataset)

        
    save_tree_graphviz(best, filename="best_tree")
