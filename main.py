import argparse  # For parsing command-line arguments
from gp import run_gp, load_default_dataset, plot_surface  # Import core GP functions
from gp.visualisation import (                             # Import visualisation utilities
    save_flowchart,
    save_tree_graphviz,
    save_crossover_plt,
    plot_surface,
)


if __name__ == "__main__":  # Main entry point for CLI execution

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Run GP symbolic regression")

    # Add configurable arguments
    parser.add_argument("--pop_size",      type=int,   default=100)   # Population size
    parser.add_argument("--generations",   type=int,   default=50)    # Number of generations
    parser.add_argument("--max_depth",     type=int,   default=6)     # Max tree depth
    parser.add_argument("--mutation_rate", type=float, default=0.1)   # Probability of mutation
    parser.add_argument("--crossover_rate",type=float, default=0.9)   # Probability of crossover
    parser.add_argument("--dataset",       type=str,   help="Path to CSV dataset")  # Optional dataset path

    args = parser.parse_args()   # Parse all arguments

    kwargs = vars(args)          # Convert parsed arguments to a dictionary

    # Remove None dataset so run_gp() loads default automatically
    if kwargs["dataset"] is None:
        kwargs.pop("dataset")    # If no dataset specified, remove it from kwargs

    best, history = run_gp(**kwargs)  # Run Genetic Programming with provided parameters

    print("\nBest evolved expression:")  # Display result
    print(best)

    dataset = kwargs.get("dataset") or load_default_dataset()  # Load dataset for plotting
    if len(dataset) > 8:                                       # Only plot surface if dataset is large enough
        plot_surface(best, dataset)                            # Show predicted vs true surface

    save_flowchart()                                           # Save GP process flowchart
    save_tree_graphviz(best, filename="best_tree")            # Save tree structure of best individual
