#!/usr/bin/env python3
"""
CLI command for plotting posterior comparisons.

This command reproduces the posterior comparison plotting functionality
from the gauss_gauss_load.ipynb notebook.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import logging

from abcnre.inference import load_estimator_from_yaml
from abcnre.diagnostics.posterior import (
    get_unnormalized_nre_pdf,
    get_normalized_pdf,
    get_unnormalized_corrected_nre_pdf,
)
from abcnre.diagnostics.viz import plot_posterior_comparison

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_posterior_comparison_command(args):
    """
    Main command function for plotting posterior comparisons.

    Args:
        args: Parsed command line arguments
    """
    logger.info("=== ABC-NRE Plot Posterior Comparison ===")

    # Setup paths
    estimator_path = Path(args.estimator_path)
    output_dir = (
        Path(args.output_dir) if args.output_dir else estimator_path.parent / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load estimator
    logger.info(f"Loading estimator from: {estimator_path}")
    estimator = load_estimator_from_yaml(estimator_path)
    simulator = estimator.simulator

    # Check if estimator has stored phis
    if not hasattr(estimator, "stored_phis") or estimator.stored_phis is None:
        raise ValueError(
            "Estimator does not have stored phi samples. Please train first."
        )

    # Get phi samples and bounds
    abc_phi_samples = estimator.stored_phis
    min_phi, max_phi = np.min(abc_phi_samples), np.max(abc_phi_samples)

    # Add padding to bounds if requested
    if args.bounds_padding > 0:
        phi_range = max_phi - min_phi
        padding = phi_range * args.bounds_padding
        min_phi -= padding
        max_phi += padding

    logger.info(f"Phi bounds: [{min_phi:.3f}, {max_phi:.3f}]")

    # Create grid for posterior calculations
    n_grid = args.grid_points
    abc_phi_grid = np.linspace(min_phi, max_phi, n_grid)

    # Get NRE posterior distribution
    logger.info("Computing NRE posterior...")
    unnormalized_pdf_func = get_unnormalized_nre_pdf(estimator)
    phi_grid, nre_pdf = get_normalized_pdf(
        unnormalized_pdf_func, initial_bounds=(min_phi, max_phi), n_grid_points=n_grid
    )

    # Get prior for plotting
    logger.info("Computing prior distribution...")
    prior_pdf_values = simulator.model.prior_phi_pdf(abc_phi_grid)

    # Get true posterior if analytical posterior is available
    true_distribution = None
    if _has_analytical_posterior(simulator.model):
        logger.info("Computing true analytical posterior...")
        true_grid, true_pdf = get_normalized_pdf(
            simulator.model.get_posterior_phi_distribution(simulator.observed_data).pdf,
            initial_bounds=(min_phi, max_phi),
        )
        true_distribution = (true_grid, true_pdf)
    else:
        logger.info("No analytical posterior available for this model")

    # Get corrected NRE posterior if requested
    corrected_distribution = None
    if args.include_corrected:
        logger.info("Computing corrected NRE posterior...")
        unnormalized_corrected_pdf_func = get_unnormalized_corrected_nre_pdf(estimator)
        phi_corrected_grid, corrected_pdf = get_normalized_pdf(
            unnormalized_pdf_func=unnormalized_corrected_pdf_func,
            initial_bounds=(min_phi, max_phi),
            n_grid_points=n_grid,
        )
        corrected_distribution = (phi_corrected_grid, corrected_pdf)

    # Prepare distributions dictionary
    distributions = {
        "NRE Posterior": (phi_grid, nre_pdf),
        "ABC Posterior": abc_phi_samples.flatten(),  # Flatten to 1D array
    }

    if true_distribution is not None:
        distributions["True Posterior"] = true_distribution

    if corrected_distribution is not None:
        distributions["Corrected NRE Posterior"] = corrected_distribution

    # Set xlim if provided
    xlim = None
    if args.xlim:
        xlim = tuple(args.xlim)

    # Generate the comparison plot
    output_path = output_dir / args.output_filename
    logger.info(f"Generating posterior comparison plot...")

    plot_posterior_comparison(
        distributions=distributions,
        prior_pdf=(abc_phi_grid, prior_pdf_values),
        xlim=xlim,
        save_path=output_path,
    )

    logger.info(f"Posterior comparison plot saved to: {output_path}")

    # Save a summary report
    _save_plotting_report(args, estimator, output_dir, distributions)


def _has_analytical_posterior(model) -> bool:
    """Check if model has analytical posterior support."""
    return (
        hasattr(model, "has_analytical_posterior") and model.has_analytical_posterior()
    ) or (
        hasattr(model, "get_model_args")
        and model.get_model_args()
        .get("metadata", {})
        .get("analytical_posterior", False)
    )


def _save_plotting_report(args, estimator, output_dir: Path, distributions: dict):
    """Save a summary report of the plotting operation."""
    report = {
        "command": "plot_posterior_comparison",
        "estimator_path": str(args.estimator_path),
        "output_directory": str(output_dir),
        "parameters": {
            "grid_points": args.grid_points,
            "bounds_padding": args.bounds_padding,
            "include_corrected": args.include_corrected,
            "xlim": args.xlim,
            "output_filename": args.output_filename,
        },
        "distributions_included": list(distributions.keys()),
        "model_info": {
            "model_type": type(estimator.simulator.model).__name__,
            "has_analytical_posterior": _has_analytical_posterior(
                estimator.simulator.model
            ),
            "summary_as_input": estimator.summary_as_input,
            "is_trained": estimator.is_trained,
        },
    }

    report_path = output_dir / "posterior_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to: {report_path}")


def setup_plot_posterior_comparison_parser(subparsers):
    """Setup argument parser for plot_posterior_comparison command."""
    parser = subparsers.add_parser(
        "plot_posterior_comparison",
        help="Generate posterior comparison plots from trained estimator",
        description="""
        Create posterior comparison plots showing NRE posterior, ABC samples, 
        true analytical posterior (if available), and optionally corrected NRE posterior.
        
        This command reproduces the plotting functionality from gauss_gauss_load.ipynb.
        """,
    )

    # Required arguments
    parser.add_argument(
        "estimator_path", type=str, help="Path to trained estimator YAML file"
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for plots (default: estimator_dir/plots)",
    )

    parser.add_argument(
        "--grid-points",
        type=int,
        default=10000,
        help="Number of grid points for posterior calculation (default: 10000)",
    )

    parser.add_argument(
        "--bounds-padding",
        type=float,
        default=0.1,
        help="Padding factor for bounds as fraction of range (default: 0.1)",
    )

    parser.add_argument(
        "--include-corrected",
        action="store_true",
        help="Include corrected NRE posterior in comparison",
    )

    parser.add_argument(
        "--xlim", type=float, nargs=2, help="X-axis limits for plot (e.g., --xlim 3 7)"
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default="posterior_comparison.png",
        help="Output filename for plot (default: posterior_comparison.png)",
    )

    parser.set_defaults(func=plot_posterior_comparison_command)

    return parser


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser(description="Plot posterior comparisons")
    setup_plot_posterior_comparison_parser(parser)
    args = parser.parse_args()
    args.func(args)
