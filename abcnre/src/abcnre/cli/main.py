#!/usr/bin/env python3
"""
ABC-NRE Command Line Interface

A comprehensive CLI for ABC-NRE training and evaluation workflows.
Provides modular commands for different stages of the ABC-NRE pipeline.
"""

import argparse
import sys
from pathlib import Path

# Import command modules
from .create_simulator import setup_create_simulator_parser
from .create_estimator import setup_create_estimator_parser
from .create_simulator_and_estimator import setup_create_simulator_and_estimator_parser
from .train_summary_stats import setup_train_summary_stats_parser
from .train_nre import setup_train_nre_parser

from .plot_grid_posterior_comparison import setup_plot_posterior_comparison_parser
from .plot_mcmc_posterior_comparison import setup_plot_mcmc_posterior_comparison_parser
from .plot_mcmc_output import setup_plot_mcmc_output_parser
from .compute_metrics import setup_compute_metrics_parser
from .run_sbc import setup_run_sbc_parser
from .run_mcmc import setup_run_mcmc_parser
from .interactive import setup_interactive_parsers


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="abcnre",
        description="""
        ABC-NRE: Approximate Bayesian Computation with Neural Ratio Estimation
        
        A modular command-line interface for the complete ABC-NRE workflow:
        1. Individual steps: create_simulator, train_summary_stats, create_estimator, train_nre
        2. Combined workflows: create_simulator_and_estimator, train_full_pipeline
        3. Evaluation pipeline: plot_posterior_comparison, compute_metrics, run_sbc
        
        Each command can be run independently with appropriate inputs.
        """,
        epilog="""
        Examples:
        
        Step-by-step workflow:
        abcnre create_simulator --model-name gauss_gauss ./sim
        abcnre train_summary_stats ./sim/simulator.yaml templates/regressor_configs/deepset_fast.yaml
        abcnre create_estimator ./sim/simulator.yaml ./est
        abcnre train_nre ./est/estimator.yaml templates/classifier_configs/mlp_fast.yaml
        
        Full pipeline (recommended):
        abcnre train_full_pipeline --model-name gauss_gauss --with-summary-stats ./output
        
        Combined workflow:
        abcnre create_simulator_and_estimator --model-name gauss_gauss ./output
        
        Evaluation workflow:
        abcnre plot_posterior_comparison ./output/estimator.yaml
        abcnre compute_metrics ./output/estimator.yaml
        abcnre run_mcmc ./output/estimator.yaml ./output/mcmc_results
        abcnre run_sbc ./output/estimator.yaml --n_samples 100
        
        MCMC sampling examples:
        abcnre run_mcmc ./estimator.yaml ./mcmc_results --n-samples-chain 50000
        abcnre run_mcmc ./estimator.yaml ./mcmc_results --no-true --output-format npy
        
        For detailed help on any command: abcnre <command> --help
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        title="Commands",
        description="Choose a command to run",
    )

    # Setup command parsers
    setup_create_simulator_parser(subparsers)
    setup_create_estimator_parser(subparsers)
    setup_create_simulator_and_estimator_parser(subparsers)
    setup_train_summary_stats_parser(subparsers)
    setup_train_nre_parser(subparsers)

    setup_plot_posterior_comparison_parser(subparsers)
    setup_plot_mcmc_posterior_comparison_parser(subparsers)
    setup_plot_mcmc_output_parser(subparsers)
    setup_compute_metrics_parser(subparsers)
    setup_run_sbc_parser(subparsers)
    setup_run_mcmc_parser(subparsers)
    setup_interactive_parsers(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # If no command provided, show help
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Run the appropriate command
    try:
        args.func(args)
    except Exception as e:
        print(f"Error running command '{args.command}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
