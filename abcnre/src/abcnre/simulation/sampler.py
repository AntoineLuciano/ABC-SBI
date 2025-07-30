"""
ABC sampling methods.

This module contains the implementation of ABC rejection sampling
and other sampling strategies, migrated from the original sampling.py.
"""

from jax import random, jit, vmap, lax, tree_util
import jax.numpy as jnp
from typing import Callable, Tuple, Optional
from functools import partial
from dataclasses import dataclass

from .base import BaseSampler, ABCSampleResult, ABCTrainingResult, ABCSingleResult


@dataclass
class RejectionState:
    """JAX-compatible state for rejection sampling loop."""

    key: random.PRNGKey
    data: jnp.ndarray
    theta: jnp.ndarray
    summary_stat: Optional[jnp.ndarray]
    distance: float
    count: int
    phi: Optional[jnp.ndarray] = None


# Register the dataclass as a JAX pytree
tree_util.register_dataclass(
    RejectionState,
    data_fields=[
        "key",
        "data",
        "theta",
        "summary_stat",
        "distance",
        "count",
        "phi",
    ],
    meta_fields=[],
)


# Core ABC sampling function (JIT compiled)
@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def get_abc_sample(
    key: random.PRNGKey,
    prior_simulator: Callable,
    data_simulator: Callable,
    discrepancy_fn: Callable,
    summary_stat_fn: Optional[Callable],
    transform_fn: Optional[Callable],
    epsilon: float,
    observed_data: jnp.ndarray,
    observed_summary_stats: Optional[jnp.ndarray],
) -> Tuple[
    jnp.ndarray, jnp.ndarray, float, Optional[jnp.ndarray], Optional[jnp.ndarray]
]:
    """
    Sample single ABC draw using rejection sampling with summary statistics.

    Args:
        key: JAX random key
        prior_simulator: Function to sample from prior
        data_simulator: Function to simulate data given parameters
        discrepancy_fn: Function to compute distance between datasets
        summary_stat_fn: Optional function to compute summary statistics
        transform_fn: Optional function to transform theta to phi
        epsilon: ABC tolerance threshold
        observed_data: Observed dataset
        observed_summary_stats: Pre-computed observed summary statistics

    Returns:
        Tuple of (simulated_data, theta, distance, summary_statistics, phi)
    """

    # Determine what to compare against
    if summary_stat_fn is not None and observed_summary_stats is not None:
        comparison_target = observed_summary_stats
    else:
        comparison_target = observed_data

    def should_continue(state: RejectionState) -> bool:
        return state.distance >= epsilon

    def rejection_step(state: RejectionState) -> RejectionState:
        """
        JAX-compatible rejection step using typed dataclass.
        Much safer than manual tuple unpacking!
        """
        key, key_theta, key_data = random.split(state.key, 3)
        theta_proposal = prior_simulator(key_theta)
        data_proposal = data_simulator(key_data, theta_proposal)

        # Transform theta to phi if transform function is provided
        if transform_fn is not None:
            phi_proposal = transform_fn(theta_proposal)
        else:
            phi_proposal = None

        # Use summary statistics if provided, otherwise raw data
        if summary_stat_fn is not None:
            summary_proposal = summary_stat_fn(data_proposal)
            distance = discrepancy_fn(summary_proposal, comparison_target)
        else:
            summary_proposal = state.summary_stat  # Keep the same structure
            distance = discrepancy_fn(data_proposal, comparison_target)

        return RejectionState(
            key=key,
            data=data_proposal,
            theta=theta_proposal,
            summary_stat=summary_proposal,
            distance=distance,
            count=state.count + 1,
            phi=phi_proposal,
        )

    # Initialize with dummy values using typed state
    key, key_theta = random.split(key)
    initial_theta = prior_simulator(key_theta)
    initial_data = jnp.zeros_like(observed_data).astype(float)

    # Initialize phi with transform function if provided
    if transform_fn is not None:
        initial_phi = transform_fn(initial_theta)
    else:
        initial_phi = None

    # Initialize summary with consistent structure
    if summary_stat_fn is not None:
        # Create dummy summary with same shape as expected output
        dummy_summary = summary_stat_fn(initial_data)
        initial_summary = jnp.zeros_like(dummy_summary)
    else:
        initial_summary = None

    initial_state = RejectionState(
        key=key,
        data=initial_data,
        theta=initial_theta,
        summary_stat=initial_summary,
        distance=epsilon + 1.0,
        count=0,
        phi=initial_phi,
    )

    final_state = lax.while_loop(should_continue, rejection_step, initial_state)

    return (
        final_state.data,
        final_state.theta,
        final_state.distance,
        final_state.summary_stat,
        final_state.phi,
        final_state.count,
    )


class RejectionSampler(BaseSampler):
    """
    ABC rejection sampler implementation.

    This class encapsulates the ABC rejection sampling algorithm
    with support for summary statistics and JIT compilation.
    """

    def __init__(
        self,
        prior_simulator: Callable,
        data_simulator: Callable,
        discrepancy_fn: Callable,
        summary_stat_fn: Optional[Callable] = None,
        transform_fn: Optional[Callable] = None,
        epsilon: float = 0.1,
        observed_data: Optional[jnp.ndarray] = None,
        observed_summary_stats: Optional[jnp.ndarray] = None,
    ):
        """
        Initialize rejection sampler.

        Args:
            prior_simulator: Function to sample from prior
            data_simulator: Function to simulate data given parameters
            discrepancy_fn: Function to compute distance between datasets
            summary_stat_fn: Optional function to compute summary statistics
            transform_fn: Optional function to transform theta to phi
            epsilon: ABC tolerance threshold
            observed_data: Observed dataset
            observed_summary_stats: Pre-computed observed summary statistics
        """
        self.prior_simulator = prior_simulator
        self.data_simulator = data_simulator
        self.discrepancy_fn = discrepancy_fn
        self.summary_stat_fn = summary_stat_fn
        self.transform_fn = transform_fn
        self.epsilon = epsilon
        self.observed_data = observed_data
        self.observed_summary_stats = observed_summary_stats

    def sample_single(self, key: random.PRNGKey) -> ABCSingleResult:
        """
        Generate single ABC sample using rejection sampling.

        Args:
            key: JAX random key

        Returns:
            ABCSingleResult with data, theta, distance, summary_stat, and phi
        """
        data, theta, distance, summary_stat, phi, count = get_abc_sample(
            key,
            self.prior_simulator,
            self.data_simulator,
            self.discrepancy_fn,
            self.summary_stat_fn,
            self.transform_fn,
            self.epsilon,
            self.observed_data,
            self.observed_summary_stats,
        )

        return ABCSingleResult(
            data=data,
            theta=theta,
            distance=distance,
            summary_stat=summary_stat,
            phi=phi,
        )

    def sample(self, key: random.PRNGKey, n_samples: int) -> ABCSampleResult:
        """
        Generate multiple ABC samples using vectorized sampling.

        Args:
            key: JAX random key
            n_samples: Number of samples to generate

        Returns:
            ABCSampleResult with all sampling results including phi_samples
        """
        keys = random.split(key, n_samples + 1)

        # Vectorize the single sample function
        vectorized_sampler = vmap(
            get_abc_sample, in_axes=(0, None, None, None, None, None, None, None, None)
        )

        (
            data,
            theta_samples,
            distances,
            summary_stats,
            phi_samples,
            rejection_count,
        ) = vectorized_sampler(
            keys[1:],
            self.prior_simulator,
            self.data_simulator,
            self.discrepancy_fn,
            self.summary_stat_fn,
            self.transform_fn,
            self.epsilon,
            self.observed_data,
            self.observed_summary_stats,
        )

        # Ensure phi_samples is 2D for consistency
        if phi_samples is not None and phi_samples.ndim == 1:
            phi_samples = phi_samples[:, None]

        return ABCSampleResult(
            data=data,
            theta=theta_samples,
            distances=distances,
            summary_stats=summary_stats,
            key=keys[0],
            simulation_count=rejection_count,
            phi=phi_samples,
        )

    def generate_training_samples(
        self, key: random.PRNGKey, n_samples: int
    ) -> ABCTrainingResult:
        """
        Generates a training dataset for Neural Ratio Estimation.

        Uses the sampler's built-in transform_fn (from the model) for theta to phi transformation.

        Args:
            key: A JAX random key for reproducibility.
            n_samples: The total number of samples to generate for the training
                set. This will be split equally between class 0 and class 1.

        Returns:
            An ABCTrainingResult named tuple with training features, labels, and metadata.
        """
        half_samples = n_samples // 2

        # Generate base ABC samples from the approximate posterior
        abc_result = self.sample(key, half_samples)
        key = abc_result.key  # Use the updated key from the result
        
        
        key, perm_key = random.split(key)
        data = abc_result.data
        permutation_indices = random.permutation(perm_key, jnp.arange(half_samples))
        
        phi = abc_result.phi
        theta = abc_result.theta
        phi_permuted = abc_result.phi[permutation_indices] if abc_result.phi is not None else None
        theta_permuted = abc_result.theta[permutation_indices]

        summary_stats = abc_result.summary_stats if abc_result.summary_stats is not None else None
        
        distances = abc_result.distances
        
        training_data = jnp.concatenate( [data, data], axis=0)
        training_summary_stats = jnp.concatenate(
            [summary_stats, summary_stats], axis=0
        ) if summary_stats is not None else None
        
        training_phi = jnp.concatenate([phi_permuted, phi], axis=0) if phi_permuted is not None else None
        training_theta = jnp.concatenate([theta_permuted, theta], axis=0)
        training_distances = jnp.concatenate([distances, distances], axis=0)
        
        training_labels = jnp.concatenate(
            [jnp.zeros(half_samples, dtype=int), jnp.ones(half_samples, dtype=int)]
        )
        
        
        return ABCTrainingResult(
            labels = training_labels,
            data = training_data,
            distances = training_distances,
            summary_stats = training_summary_stats,
            key = key,
            theta = training_theta,
            phi = training_phi,
            total_sim_count=int(abc_result.simulation_count.sum()),
        )
        
        
        
   

    def update_epsilon(self, new_epsilon: float):
        """Update epsilon value."""
        self.epsilon = new_epsilon

    def update_observed_data(self, new_observed_data: jnp.ndarray):
        """Update observed data and recompute summary statistics if needed."""
        self.observed_data = new_observed_data

        if self.summary_stat_fn is not None:
            self.observed_summary_stats = self.summary_stat_fn(new_observed_data)

    def get_epsilon_quantile(
        self, key: random.PRNGKey, alpha: float, n_samples: int = 10000
    ) -> Tuple[float, jnp.ndarray, random.PRNGKey]:
        """
        Get epsilon value corresponding to alpha quantile of distance distribution.

        Args:
            key: JAX random key
            alpha: Quantile level (e.g., 0.1 for 10th percentile)
            n_samples: Number of samples to estimate distance distribution

        Returns:
            Tuple of (epsilon_quantile, all_distances, updated_key)
        """
        # Temporarily set epsilon to infinity to get full distance distribution
        original_epsilon = self.epsilon
        self.epsilon = jnp.inf

        # Sample with infinite epsilon
        result = self.sample(key, n_samples)

        # Restore original epsilon
        self.epsilon = original_epsilon

        # Compute alpha-quantile
        epsilon_quantile = float(jnp.quantile(result.distances, alpha))

        return epsilon_quantile, result.distances, result.key


# # Backward compatibility functions
# def get_abc_samples_vectorized(
#     key: random.PRNGKey,
#     n_samples: int,
#     prior_simulator: Callable,
#     data_simulator: Callable,
#     discrepancy_fn: Callable,
#     summary_stat_fn: Optional[Callable],
#     epsilon: float,
#     observed_data: jnp.ndarray,
#     observed_summary_stats: Optional[jnp.ndarray] = None
# ) -> ABCSampleResult:
#     """
#     Backward compatibility function for get_abc_samples_vectorized.
#     """
#     sampler = RejectionSampler(
#         prior_simulator, data_simulator, discrepancy_fn,
#         summary_stat_fn, epsilon, observed_data, observed_summary_stats
#     )
#     return sampler.sample(key, n_samples)


# def get_training_samples(
#     key: random.PRNGKey,
#     n_samples: int,
#     prior_simulator: Callable,
#     data_simulator: Callable,
#     discrepancy_fn: Callable,
#     summary_stat_fn: Optional[Callable],
#     epsilon: float,
#     observed_data: jnp.ndarray,
#     observed_summary_stats: Optional[jnp.ndarray] = None,
#     marginal_index: int = 0
# ) -> ABCTrainingResult:
#     """
#     Backward compatibility function for get_training_samples.
#     """
#     sampler = RejectionSampler(
#         prior_simulator, data_simulator, discrepancy_fn,
#         summary_stat_fn, epsilon, observed_data, observed_summary_stats
#     )
#     return sampler.generate_training_samples(key, n_samples, marginal_index)


# Export all functions
__all__ = [
    "RejectionSampler",
    "get_abc_sample",
    "get_abc_samples_vectorized",
    "get_training_samples",
]
