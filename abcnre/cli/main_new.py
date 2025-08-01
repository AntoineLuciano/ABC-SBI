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
from .plot_posterior_comparison import setup_plot_posterior_comparison_parser
from .compute_metrics import setup_compute_metrics_parser
from .run_sbc import setup_run_sbc_parser


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="abcnre",
        description="""
        ABC-NRE: Approximate Bayesian Computation with Neural Ratio Estimation
        
        A modular command-line interface for the complete ABC-NRE workflow:
        1. Training pipeline: create_simulator → create_estimator
        2. Evaluation pipeline: plot_posterior_comparison, compute_metrics, run_sbc
        
        Each command can be run independently with appropriate inputs.
        """,
        epilog="""
        Examples:
        
        Training workflow:
        abcnre create_simulator gauss_gauss_1d_default ./sim --with-summary-stats
        abcnre create_estimator ./sim/simulator.yaml ./est --template mlp_fast
        
        Evaluation workflow:
        abcnre plot_posterior_comparison ./est/estimator.yaml --include-corrected
        abcnre compute_metrics ./est/estimator.yaml --include-corrected --include-abc
        abcnre run_sbc ./est/estimator.yaml --num-sbc-rounds 1000
        
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
    setup_plot_posterior_comparison_parser(subparsers)
    setup_compute_metrics_parser(subparsers)
    setup_run_sbc_parser(subparsers)

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
