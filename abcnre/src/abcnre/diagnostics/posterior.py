# src/abcnre/diagnostics/posterior.py

import jax.numpy as jnp
import numpy as np
from scipy.integrate import trapz
from typing import Callable, Tuple, Optional
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
import logging
import jax

# To avoid circular imports, we use TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..inference.estimator import NeuralRatioEstimator
    from ..simulation.samplers import ABCSimulator

# Configure logging
logger = logging.getLogger(__name__)


def _find_optimal_grid(
    pdf_func: Callable,
    initial_bounds: Tuple[float, float],
    n_grid_points: int = 500,
    threshold_factor: float = 1e-4,
    expansion_factor: float = 0.2,
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

    for _ in range(10):  # Limit expansions to avoid infinite loops
        grid = np.linspace(min_bound, max_bound, n_grid_points)
        pdf_values = np.array(pdf_func(grid))

        max_pdf = np.max(pdf_values)
        if max_pdf == 0:  # If PDF is zero everywhere, expand aggressively
            range_span = max_bound - min_bound
            min_bound -= range_span
            max_bound += range_span
            continue

        threshold = threshold_factor * max_pdf

        on_left_edge = pdf_values[0] > threshold
        on_right_edge = pdf_values[-1] > threshold

        if not on_left_edge and not on_right_edge:
            break  # Grid is sufficient

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
    else:  # Fallback to the last computed grid if no points are significant
        return grid


def get_unnormalized_nre_pdf(
    estimator: "NeuralRatioEstimator", x: Optional[jnp.ndarray] = None
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
    simulator = estimator.simulator
    model = simulator.model

    if x is None:
        x = simulator.observed_data
        s_x = simulator.observed_summary_stats
    else:
        s_x = simulator.summary_stat_fn(x)

    def unnormalized_pdf(phi_values: jnp.ndarray) -> jnp.ndarray:
        # Ensure phi_values is 2D for concatenation

        n_batch = phi_values.shape[0]

        x_batch = jnp.repeat(x[None, :], n_batch, axis=0)
        s_x_batch = jnp.repeat(s_x[None, :], n_batch, axis=0)

        if s_x_batch.ndim == 1:
            s_x_batch = s_x_batch[:, jnp.newaxis]

        if estimator.summary_as_input:
            log_ratios = estimator.log_ratio_fn(
                phi=phi_values, x=x_batch, s_x=s_x_batch
            )
        else:
            log_ratios = estimator.log_ratio_fn(phi=phi_values, x=x_batch)
        # Note: This assumes model.prior_logpdf exists. Add it to your models.
        prior_log_pdf = model.prior_phi_logpdf(phi_values)

        # p(phi|x) ∝ exp(log_r) * p(phi) => log(p(phi|x)) = log_r + log(p(phi))
        unnormalized_log_pdf = log_ratios + prior_log_pdf

        return jnp.exp(unnormalized_log_pdf)

    return unnormalized_pdf


def get_unnormalized_corrected_nre_pdf(
    estimator: "NeuralRatioEstimator",
    x: Optional[jnp.ndarray] = None,
    s_x: Optional[jnp.ndarray] = None,
    phi_samples: Optional[jnp.ndarray] = None,
    kde_approximation: Optional[Callable] = None,
    num_samples_for_kde: Optional[int] = 1000,
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
    simulator = estimator.simulator
    obs_summary = simulator.observed_summary_stats
    if kde_approximation is not None:
        kde_func = kde_approximation
    elif phi_samples is not None:
        kde_func = gaussian_kde(phi_samples.flatten())
    elif estimator.stored_phis is not None:
        kde_func = gaussian_kde(estimator.stored_phis.flatten())
    else:
        key = jax.random.PRNGKey(42)  # Use a fixed key for reproducibility
        phi_samples = simulator.get_phi_samples(num_samples_for_kde)
        kde_func = gaussian_kde(phi_samples.flatten())

    if x is None:
        x = simulator.observed_data
        s_x = simulator.observed_summary_stats
    else:
        s_x = simulator.summary_stat_fn(x)

    def unnormalized_pdf(phi_values: jnp.ndarray) -> jnp.ndarray:
        # Ensure phi_values is 2D for concatenation

        n_batch = phi_values.shape[0]

        x_batch = jnp.repeat(x[None, :], n_batch, axis=0)
        s_x_batch = jnp.repeat(s_x[None, :], n_batch, axis=0)
        print("DEBUG: s_x_batch shape:", s_x_batch.shape)
        print("DEBUG: x_batch shape:", x_batch.shape)
        print("DEBUG: phi_values shape:", phi_values.shape)
        if s_x_batch.ndim == 1:
            s_x_batch = s_x_batch[:, jnp.newaxis]
            
        if estimator.summary_as_input:
            log_ratios = estimator.log_ratio_fn(
                phi=phi_values, x=x_batch, s_x=s_x_batch
            )
        else:
            log_ratios = estimator.log_ratio_fn(phi=phi_values, x=x_batch)

        # Note: This assumes model.prior_logpdf exists. Add it to your models.
        pseudo_posterior_pdf = kde_func(phi_values.flatten())

        # p(phi|x) ∝ exp(log_r) * p(phi) => log(p(phi|x)) = log_r + log(p(phi))
        unnormalized_pdf = jnp.exp(log_ratios) * pseudo_posterior_pdf

        return unnormalized_pdf

    return unnormalized_pdf


def get_normalized_pdf(
    unnormalized_pdf_func: Callable,
    initial_bounds: Tuple[float, float],
    n_grid_points: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the normalized PDF on an optimal grid.

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
        return grid, unnormalized_pdf_values  # Avoid division by zero

    # 4. Normalize the PDF
    normalized_pdf = unnormalized_pdf_values / normalization_constant

    return grid, normalized_pdf


def sample_from_pdf(
    grid: np.ndarray,
    normalized_pdf: np.ndarray,
    n_samples: int,
    key: "jax.random.PRNGKey",
) -> jnp.ndarray:
    """
    Draws samples from a probability distribution defined on a grid.

    This function uses the inverse transform sampling method. It first computes
    the Cumulative Distribution Function (CDF) from the PDF, and then uses
    uniform random numbers to sample from the inverse CDF.

    Args:
        grid: The array of parameter values (phi).
        normalized_pdf: The corresponding normalized probability density values.
        n_samples: The number of samples to draw.
        key: A JAX random key for reproducibility.

    Returns:
        A JAX array of samples drawn from the probability distribution.
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


def get_sampler_from_pdf(
    unnormalized_pdf_func: Callable,
    initial_bounds: Tuple[float, float],
    n_grid_points: int = 1000,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Creates a sampler function from the unnormalized PDF.

    This function combines the steps of finding the optimal grid, normalizing
    the PDF, and sampling from it.

    Args:
        unnormalized_pdf_func: A function that computes the unnormalized PDF.
        initial_bounds: A tuple (min, max) to start the grid search.
        n_grid_points: The number of points for the final grid.

    Returns:
        A callable function that takes a JAX random key and returns samples.
    """
    grid, normalized_pdf = get_normalized_pdf(
        unnormalized_pdf_func, initial_bounds, n_grid_points
    )

    def sampler(key: "jax.random.PRNGKey", num_samples: int) -> jnp.ndarray:
        return sample_from_pdf(grid, normalized_pdf, num_samples, key)

    return sampler
