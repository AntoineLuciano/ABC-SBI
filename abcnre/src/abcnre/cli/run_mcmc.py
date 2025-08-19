#!/usr/bin/env python3
"""
Run MCMC Sampling Command - Run MCMC sampling for NRE, corrected NRE, and true posteriors.

This command performs MCMC sampling on various posterior distributions from a trained estimator
and saves the results for further analysis.
"""

import argparse
import jax
import jax.numpy as jnp
from pathlib import Path
import logging
import numpy as np

from abcnre.inference import load_estimator_from_yaml
from abcnre.diagnostics.posterior import (
    get_unnormalized_nre_logpdf,
    get_unnormalized_corrected_nre_logpdf,
)
from abcnre.diagnostics.mcmc import (
    adaptive_metropolis_sampler,
    save_mcmc_samples,
    MCMCResults,
)
from abcnre.cli.utils import handle_output_path, add_boolean_flag, get_default_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_mcmc_command(args):
    """
    Main command function for running MCMC sampling.

    Args:
        args: Parsed command line arguments
    """
    logger.info("=== ABC-NRE Run MCMC Sampling ===")

    # Setup paths
    estimator_path = Path(args.estimator_path)
    if not estimator_path.exists():
        raise FileNotFoundError(f"Estimator not found: {estimator_path}")

    # Handle output path with automatic filename completion
    output_file = handle_output_path(args.output_path, get_default_filename("mcmc"))

    # Load estimator
    logger.info(f"Loading estimator from: {estimator_path}")
    estimator = load_estimator_from_yaml(estimator_path)
    simulator = estimator.simulator
    model = simulator.model

    # Check if estimator is trained
    if not estimator.is_trained:
        raise ValueError("Estimator must be trained before running MCMC")

    # Get observed data
    if simulator.observed_data is None:
        raise ValueError("Simulator must have observed data for MCMC sampling")

    x_obs = simulator.observed_data

    # Initialize JAX key
    key = jax.random.PRNGKey(args.seed)

    # Setup initial state (use zeros or model-specific initialization)
    key, key_init = jax.random.split(key)
    initial_state = model.sample_phi(key_init)
    print(f"Initial state shape: {initial_state.shape}")
    # Container for results
    mcmc_results = {}
    print("PHI DIM", model.phi_dim)
    # Run NRE Standard MCMC
    if args.nre:
        logger.info("Running MCMC for NRE Standard posterior...")
        key, subkey = jax.random.split(key)

        nre_logpdf = get_unnormalized_nre_logpdf(estimator)

        nre_results = adaptive_metropolis_sampler(
            key=subkey,
            logpdf_unnorm=nre_logpdf,
            initial_state=initial_state,
            n_samples_chain=args.n_samples_chain,
            n_samples_tuning=args.n_samples_tuning,
            burnin=args.burnin,
            verbose=args.verbose,
        )

        mcmc_results["nre_standard"] = nre_results
        logger.info(
            f"NRE Standard MCMC completed. Acceptance rate: {nre_results.acceptance_rate:.3f}"
        )

    # Run Corrected NRE MCMC
    if args.corrected_nre:
        logger.info("Running MCMC for NRE Corrected posterior...")
        key, subkey = jax.random.split(key)

        corrected_nre_logpdf = get_unnormalized_corrected_nre_logpdf(estimator)

        corrected_results = adaptive_metropolis_sampler(
            key=subkey,
            logpdf_unnorm=corrected_nre_logpdf,
            initial_state=initial_state,
            n_samples_chain=args.n_samples_chain,
            n_samples_tuning=args.n_samples_tuning,
            burnin=args.burnin,
            verbose=args.verbose,
        )

        mcmc_results["nre_corrected"] = corrected_results
        logger.info(
            f"NRE Corrected MCMC completed. Acceptance rate: {corrected_results.acceptance_rate:.3f}"
        )

    # Run True Posterior MCMC (if available)
    if args.true and (hasattr(model, "get_posterior_logpdf") or hasattr(model, "get_posterior_phi_logpdf")):
        logger.info("Running MCMC for True posterior...")
        key, subkey = jax.random.split(key)

        # Create JAX-compatible true logpdf
        if hasattr(model, "get_posterior_logpdf"):
            true_logpdf = model.get_posterior_logpdf(x_obs)
        elif hasattr(model, "get_posterior_phi_logpdf"):
            true_logpdf = model.get_posterior_phi_logpdf(x_obs)

        true_results = adaptive_metropolis_sampler(
            key=subkey,
            logpdf_unnorm=true_logpdf,
            initial_state=initial_state,
            n_samples_chain=args.n_samples_chain,
            n_samples_tuning=args.n_samples_tuning,
            burnin=args.burnin,
            verbose=args.verbose,
        )

        mcmc_results["true_posterior"] = true_results
        logger.info(
            f"True Posterior MCMC completed. Acceptance rate: {true_results.acceptance_rate:.3f}"
        )

    elif args.true:
        logger.warning(
            "True posterior MCMC requested but model doesn't support analytical posterior"
        )

    # Save results
    if not mcmc_results:
        raise ValueError(
            "No MCMC runs were performed. Enable at least one method with --nre, --corrected-nre, or --true"
        )
    if args.save:
        save_data = {}
        for method_name, results in mcmc_results.items():
            save_data[method_name] = {
                "log_probs": np.array(results.log_probs),
                "samples": np.array(results.samples),
                "acceptance_rate": results.acceptance_rate,
                "n_accepted": results.n_accepted,
                "n_total": results.n_total,
            }
        np.savez(output_file, **save_data)
        logger.info(f"MCMC results saved to: {output_file}")

    # Print summary
    logger.info("\n=== MCMC Summary ===")
    for method_name, results in mcmc_results.items():
        logger.info(f"{method_name}:")
        logger.info(f"  Samples: {results.samples.shape}")
        logger.info(f"  Acceptance rate: {results.acceptance_rate:.3f}")
        logger.info(f"  Sample mean: {np.mean(results.samples, axis=0)}")
        logger.info(f"  Sample std: {np.std(results.samples, axis=0)}")
    logger.info("===================")


def setup_run_mcmc_parser(subparsers):
    """Setup argument parser for run_mcmc command."""
    parser = subparsers.add_parser(
        "run_mcmc",
        help="Run MCMC sampling for various posterior distributions",
        description="""
        Run MCMC sampling on NRE standard, corrected NRE, and/or true posteriors.
        
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
        help=f"Output path for MCMC samples. If directory provided, will save as '{get_default_filename('mcmc')}' in that directory",
    )

    # MCMC method selection
    parser.add_argument(
        "--nre",
        action="store_true",
        default=True,
        help="Run MCMC for NRE standard posterior (default: True)",
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
        help="Run MCMC for corrected NRE posterior (default: True)",
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
        help="Run MCMC for true posterior if available (default: True)",
    )

    parser.add_argument(
        "--no-true",
        dest="true",
        action="store_false",
        help="Disable true posterior sampling",
    )

    # adaptive_metropolis_sampler arguments
    parser.add_argument(
        "--n-samples-chain",
        type=int,
        default=10000,
        help="Number of MCMC samples per chain (default: 50000)",
    )

    parser.add_argument(
        "--n-samples-tuning",
        type=int,
        default=10000,
        help="Number of samples for covariance tuning (default: 10000)",
    )

    parser.add_argument(
        "--burnin",
        type=int,
        default=None,
        help="Number of burnin samples to discard (default: max(100, n_samples_chain//10))",
    )

    # General arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility (default: 123)",
    )

    # Add standardized boolean flags
    add_boolean_flag(
        parser, "save", default=True, help_text="Save MCMC results to file"
    )

    add_boolean_flag(parser, "verbose", default=True, help_text="Enable verbose output")

    parser.set_defaults(func=run_mcmc_command)

    return parser


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser(description="Run MCMC sampling")
    subparsers = parser.add_subparsers(dest="command")
    setup_run_mcmc_parser(subparsers)
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
