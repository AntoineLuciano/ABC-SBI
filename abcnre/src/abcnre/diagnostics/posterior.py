# src/abcnre/diagnostics/posterior.py

import jax.numpy as jnp
import numpy as np
from scipy.integrate import trapz
from typing import Callable, Tuple
import jax 
# To avoid circular imports, we use TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..inference.estimator import NeuralRatioEstimator
    from ..simulation.simulator import ABCSimulator


def _find_optimal_grid(
    pdf_func: Callable,
    initial_bounds: Tuple[float, float],
    n_grid_points: int = 500,
    threshold_factor: float = 1e-4,
    expansion_factor: float = 0.2
) -> np.ndarray:
    """
    Finds a grid where the PDF is non-negligible.

    This function starts with initial bounds and expands them until the PDF
    values at the edges are below a certain threshold of the maximum value.

    Args:
        pdf_func: A function that takes a grid of points and returns the PDF values.
        initial_bounds: A tuple (min, max) for the initial grid search.
        n_grid_points: The number of points to evaluate on the grid.
        threshold_factor: The fraction of the max PDF value to use as a threshold.
        expansion_factor: The factor by which to expand the grid range.

    Returns:
        A NumPy array representing the optimal grid.
    """
    min_bound, max_bound = initial_bounds
    
    for _ in range(10): # Limit expansions to avoid infinite loops
        grid = np.linspace(min_bound, max_bound, n_grid_points)
        pdf_values = np.array(pdf_func(grid))
        
        max_pdf = np.max(pdf_values)
        if max_pdf == 0: # If PDF is zero everywhere, expand aggressively
            range_span = max_bound - min_bound
            min_bound -= range_span
            max_bound += range_span
            continue

        threshold = threshold_factor * max_pdf
        
        on_left_edge = pdf_values[0] > threshold
        on_right_edge = pdf_values[-1] > threshold
        
        if not on_left_edge and not on_right_edge:
            break # Grid is sufficient

        range_span = max_bound - min_bound
        if on_left_edge:
            min_bound -= expansion_factor * range_span
        if on_right_edge:
            max_bound += expansion_factor * range_span
    
    # Final grid refinement
    significant_mask = pdf_values > threshold
    if np.any(significant_mask):
        min_grid_opt = grid[significant_mask].min()
        max_grid_opt = grid[significant_mask].max()
        range_span = max_grid_opt - min_grid_opt
        final_min = min_grid_opt - expansion_factor * range_span
        final_max = max_grid_opt + expansion_factor * range_span
        return np.linspace(final_min, final_max, n_grid_points)
    else: # Fallback to the last computed grid if no points are significant
        return grid


def get_unnormalized_nre_pdf(
    estimator: 'NeuralRatioEstimator',
    simulator: 'ABCSimulator'
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Creates a function for the unnormalized NRE posterior PDF.

    The returned function computes: p(phi|x_obs) ∝ exp(log_r(phi, x_obs)) * p(phi)

    Args:
        estimator: The trained NeuralRatioEstimator.
        simulator: The ABCSimulator containing the model and observed data.

    Returns:
        A callable function that evaluates the unnormalized posterior at given phi values.
    """
    model = simulator.model
    obs_summary = simulator.observed_summary_stats
    
    def unnormalized_pdf(phi_values: jnp.ndarray) -> jnp.ndarray:
        # Ensure phi_values is 2D for concatenation
        phi_features = phi_values[:, None] if phi_values.ndim == 1 else phi_values
        
        # Repeat observed summary statistic for each phi
        z_features = np.repeat(obs_summary[None, :], len(phi_values), axis=0)

        # Create features [phi, z_obs]
        features = np.concatenate([phi_features, z_features], axis=1)

        # Get log-ratio from the NRE
        log_ratios = estimator.log_ratio(features)
        
        # Get prior probability
        # Note: This assumes model.prior_logpdf exists. Add it to your models.
        prior_log_pdf = model.prior_logpdf(phi_values)

        # p(phi|x) ∝ exp(log_r) * p(phi) => log(p(phi|x)) = log_r + log(p(phi))
        unnormalized_log_pdf = -log_ratios + prior_log_pdf
        
        return jnp.exp(unnormalized_log_pdf)
    
    return unnormalized_pdf


def get_unormalized_corrected_nre_pdf(
    estimator: 'NeuralRatioEstimator',
    simulator: 'ABCSimulator'
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Creates a function for the unnormalized corrected NRE posterior PDF.

    The returned function computes: p(phi|x_obs) ∝ exp(log_r(phi, x_obs)) * p(phi) / p(x_obs)

    Args:
        estimator: The trained NeuralRatioEstimator.
        simulator: The ABCSimulator containing the model and observed data.

    Returns:
        A callable function that evaluates the unnormalized posterior at given phi values.
    """
    model = simulator.model
    obs_summary = simulator.observed_summary_stats
    phi_stored = simulator.phi_stored
    
    def unnormalized_pdf(phi_values: jnp.ndarray) -> jnp.ndarray:
        # Ensure phi_values is 2D for concatenation
        phi_features = phi_values[:, None] if phi_values.ndim == 1 else phi_values
        
        # Repeat observed summary statistic for each phi
        z_features = np.repeat(obs_summary[None, :], len(phi_values), axis=0)

        # Create features [phi, z_obs]
        features = np.concatenate([phi_features, z_features], axis=1)

        # Get log-ratio from the NRE
        log_ratios = estimator.log_ratio(features)
        
        # Get prior probability
        prior_log_pdf = model.prior_logpdf(phi_values)

        # Get likelihood p(x_obs | phi)
        likelihood_log_pdf = model.likelihood_logpdf(obs_summary, phi_values)

        # p(phi|x) ∝ exp(log_r) * p(phi) / p(x_obs) => log(p(phi|x)) = log_r + log(p(phi)) - log(p(x_obs))
        unnormalized_log_pdf = -log_ratios + prior_log_pdf - likelihood_log_pdf
        
        return jnp.exp(unnormalized_log_pdf)
    
    return unnormalized_pdf


def get_normalized_posterior(
    unnormalized_pdf_func: Callable,
    initial_bounds: Tuple[float, float],
    n_grid_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the normalized posterior PDF on an optimal grid.

    Args:
        unnormalized_pdf_func: A function that computes the unnormalized PDF.
        initial_bounds: A tuple (min, max) to start the grid search.
        n_grid_points: The number of points for the final grid.

    Returns:
        A tuple (grid, normalized_pdf) containing the grid points and the
        corresponding normalized probability density values.
    """
    # 1. Find the optimal grid where the posterior is significant
    grid = _find_optimal_grid(unnormalized_pdf_func, initial_bounds, n_grid_points)

    # 2. Evaluate the unnormalized PDF on this grid
    unnormalized_pdf_values = unnormalized_pdf_func(grid)

    # 3. Compute the normalization constant (area under the curve)
    normalization_constant = trapz(unnormalized_pdf_values, grid)
    
    if normalization_constant == 0:
        return grid, unnormalized_pdf_values # Avoid division by zero

    # 4. Normalize the PDF
    normalized_pdf = unnormalized_pdf_values / normalization_constant
    
    return grid, normalized_pdf

from scipy.integrate import cumulative_trapezoid

def sample_from_posterior(
    grid: np.ndarray,
    normalized_pdf: np.ndarray,
    n_samples: int,
    key: 'jax.random.PRNGKey'
) -> jnp.ndarray:
    """
    Draws samples from a posterior distribution defined on a grid.

    This function uses the inverse transform sampling method. It first computes
    the Cumulative Distribution Function (CDF) from the PDF, and then uses
    uniform random numbers to sample from the inverse CDF.

    Args:
        grid: The array of parameter values (phi).
        normalized_pdf: The corresponding normalized probability density values.
        n_samples: The number of samples to draw.
        key: A JAX random key for reproducibility.

    Returns:
        A JAX array of samples drawn from the posterior distribution.
    """
    # 1. Compute the CDF from the PDF using numerical integration
    # We add an initial 0 to the CDF to start at the beginning of the grid
    cdf = cumulative_trapezoid(normalized_pdf, grid, initial=0)
    
    # Ensure the CDF ends exactly at 1.0 for numerical stability
    cdf = cdf / cdf[-1]

    # 2. Generate uniform random numbers
    uniform_samples = jax.random.uniform(key, shape=(n_samples,))

    # 3. Use interpolation to find the inverse of the CDF
    # This is the core of the inverse transform sampling method
    posterior_samples = jnp.interp(uniform_samples, cdf, grid)
    
    return posterior_samples


