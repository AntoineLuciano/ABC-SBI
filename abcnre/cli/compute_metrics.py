#!/usr/bin/env python3
"""
CLI command for computing diagnostic metrics.

This command reproduces the metrics computation functionality
from the gauss_gauss_load.ipynb notebook.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import logging
import jax

from abcnre.inference import load_estimator_from_yaml
from abcnre.diagnostics.posterior import (
    get_unnormalized_nre_pdf,
    get_normalized_pdf,
    get_unnormalized_corrected_nre_pdf,
    get_sampler_from_pdf,
)
from abcnre.diagnostics.metrics import (
    generate_and_evaluate_metrics,
    save_metrics_to_csv,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics_command(args):
    """
    Main command function for computing diagnostic metrics.

    Args:
        args: Parsed command line arguments
    """
    logger.info("=== ABC-NRE Compute Metrics ===")

    # Setup paths
    estimator_path = Path(args.estimator_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else estimator_path.parent / "diagnostics"
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

    # Get phi samples and compute bounds
    abc_phi_samples = estimator.stored_phis
    min_phi = np.min(abc_phi_samples) - args.bounds_padding
    max_phi = np.max(abc_phi_samples) + args.bounds_padding

    logger.info(f"Using bounds: [{min_phi:.3f}, {max_phi:.3f}]")

    # Create samplers
    logger.info("Creating samplers...")
    approx_samplers_dict = {}

    # NRE Standard sampler
    logger.info("Creating NRE standard sampler...")
    unorm_nre_pdf = get_unnormalized_nre_pdf(estimator)
    nre_sampler = get_sampler_from_pdf(
        unnormalized_pdf_func=unorm_nre_pdf, initial_bounds=(min_phi, max_phi)
    )
    approx_samplers_dict["NRE_Standard"] = nre_sampler

    # NRE Corrected sampler (if requested)
    if args.include_corrected:
        logger.info("Creating NRE corrected sampler...")
        unorm_corrected_nre_pdf = get_unnormalized_corrected_nre_pdf(
            estimator, phi_samples=abc_phi_samples
        )
        corrected_nre_sampler = get_sampler_from_pdf(
            unnormalized_pdf_func=unorm_corrected_nre_pdf,
            initial_bounds=(min_phi, max_phi),
        )
        approx_samplers_dict["NRE_Corrected"] = corrected_nre_sampler

    # ABC sampler (using stored samples)
    if args.include_abc:
        logger.info("Creating ABC sampler from stored samples...")

        def abc_sampler(key, n_samples):
            # Resample from stored ABC samples
            if len(abc_phi_samples.flatten()) < n_samples:
                # If we don't have enough samples, sample with replacement
                indices = jax.random.choice(
                    key,
                    len(abc_phi_samples.flatten()),
                    shape=(n_samples,),
                    replace=True,
                )
                return abc_phi_samples.flatten()[indices]
            else:
                # Sample without replacement
                indices = jax.random.choice(
                    key,
                    len(abc_phi_samples.flatten()),
                    shape=(n_samples,),
                    replace=False,
                )
                return abc_phi_samples.flatten()[indices]

        approx_samplers_dict["ABC"] = abc_sampler

    # Get true sampler if analytical posterior is available
    true_sampler = None
    if _has_analytical_posterior(simulator.model):
        logger.info("Using analytical posterior as true sampler...")
        true_sampler = simulator.get_true_posterior_samples
    else:
        logger.warning(
            "No analytical posterior available - cannot compute metrics against truth"
        )
        if not args.force_without_truth:
            raise ValueError(
                "No analytical posterior available for metrics computation. "
                "Use --force-without-truth to skip truth-based metrics."
            )

    # Run metrics computation
    if true_sampler is not None:
        logger.info(
            f"Computing metrics with {len(approx_samplers_dict)} approximate samplers..."
        )
        key = jax.random.PRNGKey(args.random_seed)

        all_metrics_results = generate_and_evaluate_metrics(
            key=key,
            true_sampler=true_sampler,
            approx_samplers_dict=approx_samplers_dict,
            n_samples=args.n_samples,
        )

        # Display results
        logger.info("\n=== Quantitative Results ===")
        print(json.dumps(all_metrics_results, indent=2))

        # Save to CSV
        csv_path = output_dir / args.csv_filename
        save_metrics_to_csv(all_metrics_results, csv_path)
        logger.info(f"Metrics saved to: {csv_path}")

        # Save detailed JSON report
        _save_metrics_report(
            args, estimator, output_dir, all_metrics_results, approx_samplers_dict
        )

    else:
        logger.info("Skipping metrics computation due to missing analytical posterior")
        all_metrics_results = None
        _save_metrics_report(args, estimator, output_dir, None, approx_samplers_dict)


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


def _save_metrics_report(
    args, estimator, output_dir: Path, metrics_results: dict, samplers_dict: dict
):
    """Save a comprehensive metrics computation report."""
    report = {
        "command": "compute_metrics",
        "estimator_path": str(args.estimator_path),
        "output_directory": str(output_dir),
        "parameters": {
            "n_samples": args.n_samples,
            "bounds_padding": args.bounds_padding,
            "include_corrected": args.include_corrected,
            "include_abc": args.include_abc,
            "random_seed": args.random_seed,
            "csv_filename": args.csv_filename,
        },
        "samplers_created": list(samplers_dict.keys()),
        "model_info": {
            "model_type": type(estimator.simulator.model).__name__,
            "has_analytical_posterior": _has_analytical_posterior(
                estimator.simulator.model
            ),
            "summary_as_input": estimator.summary_as_input,
            "is_trained": estimator.is_trained,
        },
        "metrics_computed": metrics_results is not None,
    }

    if metrics_results is not None:
        report["metrics_summary"] = _summarize_metrics(metrics_results)

    report_path = output_dir / "metrics_computation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to: {report_path}")


def _summarize_metrics(metrics_results: dict) -> dict:
    """Create a summary of metrics results."""
    summary = {}

    for method, metrics in metrics_results.items():
        summary[method] = {}
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                summary[method][metric_name] = value
            elif isinstance(value, np.ndarray) and value.size == 1:
                summary[method][metric_name] = float(value)
            else:
                summary[method][metric_name] = str(type(value))

    return summary


def setup_compute_metrics_parser(subparsers):
    """Setup argument parser for compute_metrics command."""
    parser = subparsers.add_parser(
        "compute_metrics",
        help="Compute diagnostic metrics for trained estimator",
        description="""
        Compute comprehensive diagnostic metrics comparing different posterior
        approximation methods (NRE, corrected NRE, ABC) against analytical truth.
        
        This command reproduces the metrics computation from gauss_gauss_load.ipynb.
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
        help="Output directory for metrics (default: estimator_dir/diagnostics)",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of samples for metrics computation (default: 5000)",
    )

    parser.add_argument(
        "--bounds-padding",
        type=float,
        default=1.0,
        help="Padding for sampling bounds (default: 1.0)",
    )

    parser.add_argument(
        "--include-corrected",
        action="store_true",
        help="Include corrected NRE in metrics comparison",
    )

    parser.add_argument(
        "--include-abc",
        action="store_true",
        help="Include ABC samples in metrics comparison",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=123,
        help="Random seed for reproducible results (default: 123)",
    )

    parser.add_argument(
        "--csv-filename",
        type=str,
        default="diagnostics_metrics.csv",
        help="Output CSV filename (default: diagnostics_metrics.csv)",
    )

    parser.add_argument(
        "--force-without-truth",
        action="store_true",
        help="Allow running without analytical posterior (will skip truth-based metrics)",
    )

    parser.set_defaults(func=compute_metrics_command)

    return parser


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser(description="Compute diagnostic metrics")
    setup_compute_metrics_parser(parser)
    args = parser.parse_args()
    args.func(args)
