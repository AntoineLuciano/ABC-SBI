"""
MCMC sampling module for posterior estimation in ABC-NRE framework.

This module provides efficient Metropolis-Hastings algorithms optimized for
neural ratio estimation posteriors, with automatic tuning and diagnostics.
Features:
- JIT-compiled sampling with lax.scan
- Robbins-Monro adaptive tuning with diminishing adaptation
- Split R-hat and FFT-based effective sample size
- Robust covariance estimation
"""

import jax
import jax.numpy as jnp
import jax.random as random
import jax.lax as lax
from typing import Callable, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MCMCResults:
    """Results container for MCMC sampling."""

    samples: jnp.ndarray  # Shape: (n_samples, n_dims)
    log_probs: jnp.ndarray  # Shape: (n_samples,)
    acceptance_rate: float
    n_accepted: int
    n_total: int
    final_state: jnp.ndarray  # Last sample in chain
    final_log_prob: float


@dataclass
class TuningResults:
    """Results container for covariance tuning."""

    optimal_covariance: jnp.ndarray
    acceptance_rates: jnp.ndarray
    scale_factors: jnp.ndarray
    final_acceptance_rate: float
    tuning_samples: jnp.ndarray


def run_metropolis(
    key: jax.random.PRNGKey,
    logpdf_unnorm: Callable[[jnp.ndarray], float],
    n_samples: int,
    initial_state: jnp.ndarray,
    covariance_matrix: jnp.ndarray,
    target_acceptance_rate: float = 0.234,
    thin: int = 1,
    burnin: int = 0,
    epsilon_regularization: float = 1e-6,
    verbose: bool = True,
) -> MCMCResults:
    """
    Run Metropolis-Hastings sampling with multivariate normal proposals.

    Args:
        key: JAX random key
        logpdf_unnorm: JAX-compatible function returning log probability (scalar)
        n_samples: Number of samples to generate (after burnin and thinning)
        initial_state: Starting point for the chain (shape: n_dims)
        covariance_matrix: Covariance matrix for proposal distribution (n_dims, n_dims)
        target_acceptance_rate: Target acceptance rate (default: 0.234 optimal for multivariate)
        thin: Thinning interval (keep every `thin` samples)
        burnin: Number of burnin samples to discard
        epsilon_regularization: Regularization for numerical stability
        verbose: Whether to print progress

    Returns:
        MCMCResults with samples, acceptance rates, and diagnostics
    """
    n_dims = initial_state.shape[0]  # Use shape instead of len() for JAX compatibility
    total_iterations = burnin + n_samples * thin

    if verbose:
        logger.info(
            f"Running Metropolis-Hastings: {total_iterations} iterations, "
            f"{n_samples} final samples, burnin={burnin}, thin={thin}"
        )

    # Pre-compute Cholesky decomposition for efficient sampling
    try:
        chol_cov = jnp.linalg.cholesky(covariance_matrix)
    except Exception:
        # Add regularization if not positive definite
        regularized_cov = covariance_matrix + epsilon_regularization * jnp.eye(n_dims)
        chol_cov = jnp.linalg.cholesky(regularized_cov)
        if verbose:
            logger.warning("Covariance matrix regularized for numerical stability")

    # JIT-compiled single step function for efficiency
    @jax.jit
    def metropolis_step(state, key_input):
        current_state, current_log_prob = state
        key = key_input

        # Generate proposal
        key, subkey = random.split(key)
        z = random.normal(subkey, shape=current_state.shape)
        proposal = current_state + chol_cov @ z
        # Evaluate proposal
        proposal_log_prob = logpdf_unnorm(proposal)
        # Accept/reject
        log_alpha = proposal_log_prob - current_log_prob
        log_alpha = log_alpha.squeeze()
        key, subkey = random.split(key)
        accept = jnp.log(random.uniform(subkey)) < log_alpha

        # Update state
        new_state = jnp.where(accept, proposal, current_state)
        
        new_log_prob = jnp.where(accept, proposal_log_prob, current_log_prob)

        return (new_state, new_log_prob), (new_state, new_log_prob, accept)

    # Current state
    current_state = jnp.array(initial_state)
    current_log_prob = logpdf_unnorm(current_state)
    # Use lax.scan for efficient sampling
    keys = random.split(key, total_iterations)
    initial_state_for_scan = (current_state, current_log_prob)

    final_state, (all_states, all_log_probs, all_accepts) = lax.scan(
        metropolis_step, initial_state_for_scan, keys
    )

    # Extract samples after burnin and thinning
    valid_indices = jnp.arange(burnin, total_iterations, thin)
    samples = all_states[valid_indices]
    log_probs = all_log_probs[valid_indices]

    # Calculate acceptance rate
    n_accepted = jnp.sum(all_accepts)
    final_acceptance_rate = n_accepted / total_iterations

    if verbose:
        logger.info(
            f"MCMC completed. Final acceptance rate: {final_acceptance_rate:.3f}"
        )
        logger.info(f"Target acceptance rate: {target_acceptance_rate:.3f}")

    return MCMCResults(
        samples=samples,
        log_probs=log_probs,
        acceptance_rate=float(final_acceptance_rate),
        n_accepted=int(n_accepted),
        n_total=total_iterations,
        final_state=final_state[0],
        final_log_prob=final_state[1],
    )


def get_tuned_covariance(
    key: jax.random.PRNGKey,
    logpdf_unnorm: Callable[[jnp.ndarray], float],
    initial_state: jnp.ndarray,
    n_tuning_samples: int = 1000,
    initial_scale: float = 1.0,
    adaptation_window: int = 200,
    epsilon_regularization: float = 1e-6,
    verbose: bool = True,
) -> TuningResults:
    """
    Robbins-Monro adaptive tuning with diminishing adaptation rate.

    Uses Robbins-Monro stochastic approximation for optimal scaling with
    adaptive window and diminishing step sizes.

    Args:
        key: JAX random key
        logpdf_unnorm: Target log probability function
        initial_state: Starting point for tuning
        n_tuning_samples: Number of samples for tuning
        target_acceptance_rate: Target acceptance rate (0.234 is optimal for multivariate)
        initial_scale: Initial scaling factor for covariance
        adaptation_window: Window size for empirical covariance updates
        verbose: Whether to print tuning progress

    Returns:
        TuningResults with optimal covariance and tuning diagnostics
    """
    n_dims = len(initial_state)
    if n_dims > 1:
        target_acceptance_rate = 0.234  # Optimal for multivariate normal
    else:
        target_acceptance_rate = 0.44  # Optimal for univariate normal
    if verbose:
        logger.info(f"Robbins-Monro tuning with {n_tuning_samples} samples")
        logger.info(f"Target acceptance rate: {target_acceptance_rate}")

    # Initial covariance estimate using short preliminary run
    key, subkey = random.split(key)
    preliminary_samples = min(500, n_tuning_samples // 4)
    prelim_cov = (initial_scale**2) * jnp.eye(n_dims)

    base_covariance = jnp.eye(n_dims)
    try:
        prelim_results = run_metropolis(
            key=subkey,
            logpdf_unnorm=logpdf_unnorm,
            n_samples=preliminary_samples,
            initial_state=initial_state,
            covariance_matrix=prelim_cov,
            verbose=False,
        )

        if len(prelim_results.samples) > 10:
            empirical_cov = jnp.cov(prelim_results.samples.T)
            base_covariance = empirical_cov + epsilon_regularization * jnp.eye(n_dims)
            if verbose:
                logger.info(
                    f"Empirical covariance from {len(prelim_results.samples)} preliminary samples"
                )
    except Exception as e:
        if verbose:
            logger.warning(f"Using identity covariance: {e}")

    # Robbins-Monro adaptation
    log_scale = jnp.log(initial_scale)
    current_state = jnp.array(initial_state)

    acceptance_rates = []
    scale_factors = []
    tuning_samples_list = []

    # Adaptation phases with diminishing step sizes
    n_adaptation_phases = min(10, max(1, n_tuning_samples // adaptation_window))
    samples_per_phase = max(1, n_tuning_samples // n_adaptation_phases)

    for phase in range(n_adaptation_phases):
        # Robbins-Monro step size: decreases as 1/(phase+1)^0.6
        step_size = 1.0 / ((phase + 1) ** 0.6)

        current_scale = jnp.exp(log_scale)
        current_scale = jnp.clip(current_scale, 0.01, 10.0)
        current_covariance = (
            current_scale**2
        ) * base_covariance  # Fixed: scale squared for covariance

        # Run MCMC for this phase
        key, subkey = random.split(key)
        phase_results = run_metropolis(
            key=subkey,
            logpdf_unnorm=logpdf_unnorm,
            n_samples=samples_per_phase,
            initial_state=current_state,
            covariance_matrix=current_covariance,
            verbose=False,
        )

        acceptance_rate = phase_results.acceptance_rate
        acceptance_rates.append(acceptance_rate)
        scale_factors.append(current_scale)
        tuning_samples_list.extend(phase_results.samples)

        # Update for next phase
        current_state = phase_results.final_state

        # Update empirical covariance every adaptation_window samples
        if len(tuning_samples_list) >= adaptation_window:
            recent_samples = jnp.array(tuning_samples_list[-adaptation_window:])
            if len(recent_samples) > n_dims:
                empirical_cov = jnp.cov(recent_samples.T)
                # Blend with previous estimate
                blend_factor = 0.1  # Small update
                base_covariance = (
                    1 - blend_factor
                ) * base_covariance + blend_factor * (
                    empirical_cov + epsilon_regularization * jnp.eye(n_dims)
                )

        # Robbins-Monro update for log-scale
        error = acceptance_rate - target_acceptance_rate
        log_scale = log_scale + step_size * error

        if verbose and phase % max(1, n_adaptation_phases // 10) == 0:
            logger.info(
                f"Phase {phase+1}/{n_adaptation_phases}, "
                f"scale: {current_scale:.4f}, "
                f"acceptance: {acceptance_rate:.3f}, "
                f"step_size: {step_size:.4f}"
            )

    # Final covariance with corrected scaling
    final_scale = jnp.exp(log_scale)
    final_scale = jnp.clip(final_scale, 0.01, 10.0)
    optimal_covariance = (
        final_scale**2
    ) * base_covariance  # Scale squared for covariance

    if verbose:
        logger.info(f"Robbins-Monro tuning completed. Final scale: {final_scale:.4f}")
        logger.info(f"Final acceptance rate: {acceptance_rates[-1]:.3f}")

    return TuningResults(
        optimal_covariance=optimal_covariance,
        acceptance_rates=jnp.array(acceptance_rates),
        scale_factors=jnp.array(scale_factors),
        final_acceptance_rate=acceptance_rates[-1],
        tuning_samples=(
            jnp.array(tuning_samples_list) if tuning_samples_list else jnp.array([])
        ),
    )


def estimate_effective_sample_size(
    samples: jnp.ndarray, max_lag: Optional[int] = None
) -> jnp.ndarray:
    """
    Estimate effective sample size for each dimension using autocorrelation.

    Args:
        samples: MCMC samples (n_samples, n_dims)
        max_lag: Maximum lag for autocorrelation (default: n_samples // 4)

    Returns:
        Effective sample sizes for each dimension
    """
    n_samples, n_dims = samples.shape
    if max_lag is None:
        max_lag = n_samples // 4

    ess = jnp.zeros(n_dims)

    for dim in range(n_dims):
        x = samples[:, dim]
        x_centered = x - jnp.mean(x)

        # Compute autocorrelation
        autocorr = jnp.correlate(x_centered, x_centered, mode="full")
        autocorr = autocorr[autocorr.size // 2 :]
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find integrated autocorrelation time
        # Sum until autocorrelation becomes negligible or changes sign
        tau_int = 1.0
        for lag in range(1, min(max_lag, len(autocorr))):
            if autocorr[lag] <= 0:
                break
            tau_int += 2 * autocorr[lag]

        # Effective sample size
        ess = ess.at[dim].set(n_samples / (2 * tau_int + 1))

    return ess


def gelman_rubin_diagnostic(chains: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Gelman-Rubin diagnostic (R-hat) for convergence assessment.

    Args:
        chains: Multiple chains (n_chains, n_samples, n_dims)

    Returns:
        R-hat values for each dimension (should be close to 1.0 for convergence)
    """
    n_chains, n_samples, n_dims = chains.shape

    if n_chains < 2:
        raise ValueError("Need at least 2 chains for Gelman-Rubin diagnostic")

    rhat = jnp.zeros(n_dims)

    for dim in range(n_dims):
        # Chain means and overall mean
        chain_means = jnp.mean(chains[:, :, dim], axis=1)  # (n_chains,)
        overall_mean = jnp.mean(chain_means)

        # Between-chain variance
        B = n_samples * jnp.var(chain_means, ddof=1)

        # Within-chain variance
        chain_vars = jnp.var(chains[:, :, dim], axis=1, ddof=1)  # (n_chains,)
        W = jnp.mean(chain_vars)

        # Pooled variance estimate
        var_plus = ((n_samples - 1) * W + B) / n_samples

        # R-hat
        rhat = rhat.at[dim].set(jnp.sqrt(var_plus / W))

    return rhat


def split_rhat_diagnostic(chains: jnp.ndarray) -> jnp.ndarray:
    """
    Compute split R-hat diagnostic for improved convergence assessment.

    More robust than standard R-hat by splitting each chain in half,
    effectively doubling the number of chains for better variance estimates.

    Args:
        chains: Multiple chains (n_chains, n_samples, n_dims)

    Returns:
        Split R-hat values for each dimension
    """
    n_chains, n_samples, n_dims = chains.shape

    if n_samples < 4:
        raise ValueError("Need at least 4 samples per chain for split R-hat")

    # Split each chain in half
    half_samples = n_samples // 2
    first_half = chains[:, :half_samples, :]
    second_half = chains[:, half_samples : half_samples * 2, :]

    # Combine split chains (now we have 2*n_chains effective chains)
    split_chains = jnp.concatenate([first_half, second_half], axis=0)

    # Compute standard R-hat on split chains
    return gelman_rubin_diagnostic(split_chains)


def fft_effective_sample_size(samples: jnp.ndarray) -> jnp.ndarray:
    """
    Compute effective sample size using FFT-based autocorrelation.

    More efficient and numerically stable than direct autocorrelation
    computation, especially for long chains.

    Args:
        samples: MCMC samples (n_samples, n_dims)

    Returns:
        Effective sample sizes for each dimension
    """
    n_samples, n_dims = samples.shape
    ess = jnp.zeros(n_dims)

    for dim in range(n_dims):
        x = samples[:, dim]
        x_centered = x - jnp.mean(x)

        # Zero-pad for FFT
        n_fft = 2 * n_samples
        x_padded = jnp.concatenate([x_centered, jnp.zeros(n_samples)])

        # FFT-based autocorrelation
        x_fft = jnp.fft.fft(x_padded)
        autocorr_fft = jnp.fft.ifft(x_fft * jnp.conj(x_fft)).real
        autocorr = autocorr_fft[:n_samples]

        # Protected normalization
        denom = jnp.maximum(autocorr[0], 1e-12)
        autocorr = autocorr / denom

        # Automatic windowing (stop when autocorrelation becomes negligible)
        # Find first negative value or when it drops below threshold
        cutoff = n_samples
        for lag in range(1, n_samples):
            if autocorr[lag] <= 0.01:  # 1% threshold
                cutoff = lag
                break

        # Integrated autocorrelation time with automatic cutoff
        tau_int = 1.0 + 2 * jnp.sum(autocorr[1:cutoff])
        tau_int = jnp.maximum(tau_int, 1.0)  # Ensure positive

        # Effective sample size with clipping
        ess_dim = n_samples / tau_int
        ess_dim = jnp.clip(ess_dim, 1.0, float(n_samples))
        ess = ess.at[dim].set(ess_dim)

    return ess


def comprehensive_mcmc_diagnostics(
    chains: jnp.ndarray, verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive MCMC diagnostics combining multiple metrics.

    Args:
        chains: Multiple chains (n_chains, n_samples, n_dims) or single chain (n_samples, n_dims)
        verbose: Whether to print diagnostic summary

    Returns:
        Dictionary with all diagnostic metrics
    """
    # Handle single chain case
    if chains.ndim == 2:
        chains = chains[None, :, :]  # Add chain dimension

    n_chains, n_samples, n_dims = chains.shape

    diagnostics = {}

    # Basic R-hat (if multiple chains)
    if n_chains > 1:
        diagnostics["rhat"] = gelman_rubin_diagnostic(chains)

        # Split R-hat (more robust)
        if n_samples >= 4:
            diagnostics["split_rhat"] = split_rhat_diagnostic(chains)

    # Effective sample sizes
    all_samples = chains.reshape(-1, n_dims)  # Combine all chains
    diagnostics["ess_basic"] = estimate_effective_sample_size(all_samples)
    diagnostics["ess_fft"] = fft_effective_sample_size(all_samples)

    # Per-chain ESS for comparison
    diagnostics["ess_per_chain"] = [
        fft_effective_sample_size(chains[i]) for i in range(n_chains)
    ]

    # Acceptance rates (if available in chain metadata)
    diagnostics["n_chains"] = n_chains
    diagnostics["n_samples_per_chain"] = n_samples
    diagnostics["total_samples"] = n_chains * n_samples

    # Monte Carlo standard errors with protected division
    ess_fft_clipped = jnp.maximum(diagnostics["ess_fft"], 1.0)
    mcmc_se = jnp.sqrt(jnp.var(all_samples, axis=0) / ess_fft_clipped)
    diagnostics["mcmc_standard_errors"] = mcmc_se

    if verbose:
        logger.info("\n=== MCMC Diagnostics Summary ===")
        logger.info(f"Total chains: {n_chains}")
        logger.info(f"Samples per chain: {n_samples}")
        logger.info(f"Total samples: {n_chains * n_samples}")

        if "rhat" in diagnostics:
            max_rhat = jnp.max(diagnostics["rhat"])
            logger.info(f"Max R-hat: {max_rhat:.4f} {'✓' if max_rhat < 1.1 else '✗'}")

        if "split_rhat" in diagnostics:
            max_split_rhat = jnp.max(diagnostics["split_rhat"])
            logger.info(
                f"Max Split R-hat: {max_split_rhat:.4f} {'✓' if max_split_rhat < 1.1 else '✗'}"
            )

        min_ess = jnp.min(diagnostics["ess_fft"])
        logger.info(f"Min ESS (FFT): {min_ess:.0f}")

        avg_mcmc_se = jnp.mean(diagnostics["mcmc_standard_errors"])
        logger.info(f"Avg MCMC SE: {avg_mcmc_se:.4f}")

        # Convergence assessment
        converged = True
        if "split_rhat" in diagnostics:
            converged &= jnp.all(diagnostics["split_rhat"] < 1.1)
        converged &= min_ess > 400  # Rule of thumb

        logger.info(f"Overall convergence: {'✓' if converged else '✗'}")
        logger.info("===============================\n")

    return diagnostics


def adaptive_metropolis_sampler(
    key: jax.random.PRNGKey,
    logpdf_unnorm: Callable[[jnp.ndarray], float],
    initial_state: jnp.ndarray,
    n_samples_chain: int,
    n_samples_tuning: int,
    burnin: Optional[int] = None,
    verbose: bool = True,
) -> MCMCResults:
    """
    Adaptive Metropolis sampler with Robbins-Monro scale adaptation.

    Args:
        key: JAX random key
        logpdf_unnorm: Target log probability function (must be JAX-compatible)
        initial_state: Starting point
        n_samples_chain: Number of final samples to keep
        n_samples_tuning: Number of samples for covariance tuning
        burnin: Number of burnin samples to discard (default: max(100, n_samples_chain//10))
        verbose: Whether to print progress

    Returns:
        MCMCResults with adapted sampling
    """
    # Set default burnin if not provided
    if burnin is None:
        burnin = max(100, n_samples_chain // 10)

    n_dims = initial_state.shape[0]  # Use shape instead of len() for JAX compatibility
    # Start with tuned covariance
    if verbose:
        logger.info("Initial covariance tuning...")
    key, subkey = random.split(key)
    tuning_results = get_tuned_covariance(
        key=subkey,
        logpdf_unnorm=logpdf_unnorm,
        initial_state=initial_state,
        verbose=verbose,
        n_tuning_samples=n_samples_tuning,
    )

    optimal_covariance = tuning_results.optimal_covariance

    key, subkey = random.split(key)
    # Run burnin + final samples
    total_samples = burnin + n_samples_chain

    if verbose:
        logger.info(
            f"Running {total_samples} total samples ({burnin} burnin + {n_samples_chain} final)"
        )

    final_results = run_metropolis(
        key=subkey,
        logpdf_unnorm=logpdf_unnorm,
        n_samples=total_samples,
        initial_state=initial_state,
        covariance_matrix=optimal_covariance,
        verbose=False,
    )

    # Discard burnin samples
    final_samples = final_results.samples[burnin:]
    final_log_probs = final_results.log_probs[burnin:]

    # Calculate acceptance rate on final samples (post-burnin)
    final_acceptance_rate = final_results.n_accepted / len(final_results.samples)

    if verbose:
        logger.info(
            f"Adaptive sampling completed. Final acceptance rate: {final_acceptance_rate:.3f}"
        )
        logger.info(f"Final scale factor: {tuning_results.scale_factors[-1]:.3f}")

    return MCMCResults(
        samples=jnp.array(final_samples),
        log_probs=jnp.array(final_log_probs),
        acceptance_rate=final_acceptance_rate,
        n_accepted=final_results.n_accepted,
        n_total=len(final_samples),  # Use final samples count, not total
        final_state=final_results.final_state,
        final_log_prob=(final_results.final_log_prob),
    )


def save_mcmc_samples(results: MCMCResults, filepath: str):
    """
    Save MCMC results to a file.

    Args:
        results: MCMCResults object containing the samples and diagnostics
        filepath: Path to the file where results will be saved
    """
    np.save(filepath, np.array(results.samples))
