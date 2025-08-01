#!/usr/bin/env python3
"""
CLI command for running Simulation-Based Calibration (SBC).

This command reproduces the SBC functionality from the gauss_gauss_load.ipynb notebook.
"""

import argparse
import json
from pathlib import Path
import logging
import jax

from abcnre.inference import load_estimator_from_yaml
from abcnre.diagnostics.calibration import run_abc_sbc
from abcnre.diagnostics.viz import plot_sbc_ranks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_sbc_command(args):
    """
    Main command function for running SBC.

    Args:
        args: Parsed command line arguments
    """
    logger.info("=== ABC-NRE Simulation-Based Calibration ===")

    # Setup paths
    estimator_path = Path(args.estimator_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else estimator_path.parent / "diagnostics"
    )
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load estimator
    logger.info(f"Loading estimator from: {estimator_path}")
    estimator = load_estimator_from_yaml(estimator_path)

    # Check if estimator is trained
    if not estimator.is_trained:
        raise ValueError("Estimator must be trained before running SBC")

    # Run SBC
    logger.info(f"Running SBC with {args.num_sbc_rounds} rounds...")
    logger.info(f"Using {args.num_posterior_samples} posterior samples per round")

    key = jax.random.PRNGKey(args.random_seed)

    sbc_results = run_abc_sbc(
        key=key,
        estimator=estimator,
        num_sbc_rounds=args.num_sbc_rounds,
        num_posterior_samples=args.num_posterior_samples,
    )

    logger.info("✅ SBC computation complete")

    # Generate SBC plot
    plot_path = plots_dir / args.plot_filename
    logger.info(f"Generating SBC ranks plot...")

    plot_sbc_ranks(sbc_results, save_path=plot_path)
    logger.info(f"✅ SBC plot saved to: {plot_path}")

    # Save SBC results
    results_path = output_dir / args.results_filename
    logger.info(f"Saving SBC results...")

    save_sbc_results = sbc_results.save(
        results_path,
        save_samples=args.save_samples,
        save_ranks=args.save_ranks,
        save_posterior_samples=args.save_posterior_samples,
    )

    logger.info(f"✅ SBC results saved to: {results_path}")

    # Analyze SBC results
    _analyze_sbc_results(sbc_results, args)

    # Save summary report
    _save_sbc_report(args, estimator, sbc_results, output_dir, plot_path, results_path)


def _analyze_sbc_results(sbc_results, args):
    """Analyze and report SBC results."""
    logger.info("\n=== SBC Analysis ===")

    # Get ranks for analysis
    ranks = sbc_results.ranks

    # Basic statistics
    n_rounds = len(ranks)
    expected_mean = args.num_posterior_samples / 2
    actual_mean = ranks.mean()

    logger.info(f"Number of SBC rounds: {n_rounds}")
    logger.info(f"Expected rank mean: {expected_mean:.2f}")
    logger.info(f"Actual rank mean: {actual_mean:.2f}")
    logger.info(f"Rank mean deviation: {abs(actual_mean - expected_mean):.2f}")

    # Check for uniformity (rough)
    rank_std = ranks.std()
    expected_std = (
        args.num_posterior_samples**2 - 1
    ) / 12  # Uniform distribution variance
    expected_std = expected_std**0.5

    logger.info(f"Expected rank std: {expected_std:.2f}")
    logger.info(f"Actual rank std: {rank_std:.2f}")

    # Simple calibration assessment
    deviation_threshold = 0.1 * expected_mean  # 10% threshold
    is_well_calibrated = abs(actual_mean - expected_mean) < deviation_threshold

    logger.info(
        f"Well calibrated (mean within {deviation_threshold:.1f}): {is_well_calibrated}"
    )

    if is_well_calibrated:
        logger.info("✅ SBC suggests good calibration")
    else:
        logger.warning("⚠️  SBC suggests potential calibration issues")

    return {
        "n_rounds": n_rounds,
        "expected_mean": expected_mean,
        "actual_mean": float(actual_mean),
        "rank_std": float(rank_std),
        "is_well_calibrated": is_well_calibrated,
        "deviation_threshold": deviation_threshold,
    }


def _save_sbc_report(
    args, estimator, sbc_results, output_dir: Path, plot_path: Path, results_path: Path
):
    """Save a comprehensive SBC report."""

    # Get analysis results
    analysis = _analyze_sbc_results(sbc_results, args)

    report = {
        "command": "run_sbc",
        "estimator_path": str(args.estimator_path),
        "output_directory": str(output_dir),
        "parameters": {
            "num_sbc_rounds": args.num_sbc_rounds,
            "num_posterior_samples": args.num_posterior_samples,
            "random_seed": args.random_seed,
            "plot_filename": args.plot_filename,
            "results_filename": args.results_filename,
            "save_samples": args.save_samples,
            "save_ranks": args.save_ranks,
            "save_posterior_samples": args.save_posterior_samples,
        },
        "files_created": {"plot": str(plot_path), "results": str(results_path)},
        "model_info": {
            "model_type": type(estimator.simulator.model).__name__,
            "summary_as_input": estimator.summary_as_input,
            "is_trained": estimator.is_trained,
        },
        "sbc_analysis": analysis,
    }

    report_path = output_dir / "sbc_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"SBC report saved to: {report_path}")


def setup_run_sbc_parser(subparsers):
    """Setup argument parser for run_sbc command."""
    parser = subparsers.add_parser(
        "run_sbc",
        help="Run Simulation-Based Calibration for trained estimator",
        description="""
        Run Simulation-Based Calibration (SBC) to assess the calibration quality
        of a trained neural ratio estimator.
        
        This command reproduces the SBC functionality from gauss_gauss_load.ipynb.
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
        help="Output directory for SBC results (default: estimator_dir/diagnostics)",
    )

    parser.add_argument(
        "--num-sbc-rounds",
        type=int,
        default=1000,
        help="Number of SBC rounds to run (default: 1000)",
    )

    parser.add_argument(
        "--num-posterior-samples",
        type=int,
        default=128,
        help="Number of posterior samples per SBC round (default: 128)",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42)",
    )

    parser.add_argument(
        "--plot-filename",
        type=str,
        default="sbc.png",
        help="Filename for SBC plot (default: sbc.png)",
    )

    parser.add_argument(
        "--results-filename",
        type=str,
        default="sbc_results.json",
        help="Filename for SBC results (default: sbc_results.json)",
    )

    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save all generated samples in results file",
    )

    parser.add_argument(
        "--save-ranks",
        action="store_true",
        default=True,
        help="Save rank statistics in results file (default: True)",
    )

    parser.add_argument(
        "--save-posterior-samples",
        action="store_true",
        help="Save posterior samples in results file",
    )

    parser.set_defaults(func=run_sbc_command)

    return parser


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser(description="Run Simulation-Based Calibration")
    setup_run_sbc_parser(parser)
    args = parser.parse_args()
    args.func(args)
