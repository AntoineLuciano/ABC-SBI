#!/usr/bin/env python

import argparse
import jax
import jax.numpy as jnp
from pathlib import Path
import logging
import numpy as np

from abcnre.inference import load_estimator_from_yaml
from abcnre.diagnostics.posterior import (
    get_unnormalized_nre_pdf,
    get_unnormalized_corrected_nre_pdf,
    get_normalized_pdf
)

from abcnre.cli.utils import handle_output_path, add_boolean_flag, get_default_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_grid_command(args):
    """
    Main command function for running optimal Grid searching.

    Args:
        args: Parsed command line arguments
    """
    logger.info("=== ABC-NRE Run Optimal Grid Finding ===")
    n_grid = args.n_grid
    # Setup paths
    estimator_path = Path(args.estimator_path)
    if not estimator_path.exists():
        raise FileNotFoundError(f"Estimator not found: {estimator_path}")

    # Handle output path with automatic filename completion
    output_file = handle_output_path(args.output_path, get_default_filename("grid"))

    # Load estimator
    logger.info(f"Loading estimator from: {estimator_path}")
    estimator = load_estimator_from_yaml(estimator_path)
    simulator = estimator.simulator
    model = simulator.model

    # Check if estimator is trained
    if not estimator.is_trained:
        raise ValueError("Estimator must be trained before running grid")

    # Get observed data
    if simulator.observed_data is None:
        raise ValueError("Simulator must have observed data for grid sampling")

    x_obs = simulator.observed_data

    grid_results = {}
    abc_phi_samples = estimator.stored_phis
    # min_phi, max_phi = np.min(abc_phi_samples), np.max(abc_phi_samples)
    min_phi, max_phi = -100., 100.
    # abc_phi_grid = np.linspace(min_phi, max_phi, n_grid)
    if args.nre:
        logger.info("Running grid search for NRE standard posterior...")
        unnormalized_pdf_func = get_unnormalized_nre_pdf(estimator)
        logger.info(f"Using grid bounds: [{min_phi}, {max_phi}]")
        nre_grid, nre_pdf = get_normalized_pdf(unnormalized_pdf_func, 
                                            initial_bounds=(min_phi, max_phi), 
                                                n_grid_points=n_grid)
        grid_results["nre_standard"]= {"grid": nre_grid, "pdf": nre_pdf}
        logger.info(f"Done! Max = {np.max(nre_pdf)} reach for phi = {nre_grid[np.argmax(nre_pdf)]}")


    if args.true:
        logger.info("Running grid search for true posterior...")
        true_grid, true_pdf = get_normalized_pdf(estimator.simulator.model.get_posterior_phi_pdf(estimator.simulator.observed_data), initial_bounds=(min_phi, max_phi))
        grid_results["true_posterior"] = {"grid": true_grid, "pdf": true_pdf}
        logger.info(f"Done! Max = {np.max(true_pdf)} reach for phi = {true_grid[np.argmax(true_pdf)]}")

        
    if args.corrected_nre:
        logger.info("Running grid search for corrected NRE posterior...")
        unnormalized_corrected_pdf_func = get_unnormalized_corrected_nre_pdf(estimator)
        nre_corrected_grid, nre_corrected_pdf = get_normalized_pdf(
                                            unnormalized_pdf_func=unnormalized_corrected_pdf_func,
                                            initial_bounds=(min_phi, max_phi),
                                            n_grid_points=n_grid
        )
        grid_results["nre_corrected"] = {"grid": nre_corrected_grid, "pdf": nre_corrected_pdf}
        logger.info(f"Done! Max = {np.max(nre_corrected_pdf)} reach for phi = {nre_corrected_grid[np.argmax(nre_corrected_pdf)]}")

    if args.save:
        np.savez(output_file, grid_results)
        logger.info(f"Grid results saved to: {output_file}")





def setup_run_grid_parser(subparsers):
    """Setup argument parser for run_grid command."""
    parser = subparsers.add_parser(
        "run_grid",
        help="Run grid search for various posterior distributions",
        description="""
        Run optimal grid searching on NRE standard, corrected NRE, and/or true posteriors.
        
        This command uses adaptive Metropolis-Hastings sampling with automatic tuning
        to generate samples from the specified posterior distributions.
        """,
    )

    # Required arguments
    parser.add_argument(
        "estimator_path", type=str, help="Path to trained estimator YAML file"
    )

    parser.add_argument(
        "output_path",
        type=str,
        help=f"Output path for grid results. If directory provided, will save as '{get_default_filename('grid')}' in that directory",
    )

    parser.add_argument(
        "--nre",
        action="store_true",
        default=True,
        help="Run grid search for NRE standard posterior (default: True)",
    )

    parser.add_argument(
        "--no-nre",
        dest="nre",
        action="store_false",
        help="Disable NRE standard posterior sampling",
    )

    parser.add_argument(
        "--corrected-nre",
        action="store_true",
        default=True,
        help="Run grid search for corrected NRE posterior (default: True)",
    )

    parser.add_argument(
        "--no-corrected-nre",
        dest="corrected_nre",
        action="store_false",
        help="Disable corrected NRE posterior sampling",
    )

    parser.add_argument(
        "--true",
        dest="true",
        action="store_true",
        help="Run grid search for true posterior if available (default: True)",
    )

    parser.add_argument(
        "--no-true",
        dest="true",
        action="store_false",
        help="Disable true posterior sampling",
    )

    # adaptive_metropolis_sampler arguments
    parser.add_argument(
        "--n-grid",
        type=int,
        default=10000,
        help="Number of grid points (default: 10000)",
    )


    # Add standardized boolean flags
    add_boolean_flag(
        parser, "save", default=True, help_text="Save grid results to file"
    )

    add_boolean_flag(parser, "verbose", default=True, help_text="Enable verbose output")

    parser.set_defaults(func=run_grid_command)

    return parser


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser(description="Run grid search")
    subparsers = parser.add_subparsers(dest="command")
    setup_run_grid_parser(subparsers)
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
