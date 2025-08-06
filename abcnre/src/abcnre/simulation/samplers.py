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

from typing import NamedTuple, Optional, List, Dict
from .base import ABCSampleResult, ABCTrainingResult, ABCSingleResult
from .models.base import StatisticalModel


@dataclass
class RejectionState:
    """JAX-compatible state for rejection sampling loop."""

    key: random.PRNGKey
    data: jnp.ndarray
    theta: jnp.ndarray
    summary_stat: Optional[jnp.ndarray]
    distance: float
    count: int


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
    ],
    meta_fields=[],
)

def get_epsilon_quantile(
    key: random.PRNGKey,
    sample_theta_x: Callable,
    discrepancy_fn: Callable,
    alpha: float,
    n_samples: int = 10000) -> Tuple[float, jnp.ndarray, random.PRNGKey]:
    """
    Get epsilon value corresponding to alpha quantile of distance distribution.

    Returns:
        Tuple of (epsilon_quantile, all_distances, updated_key)
    """

    if alpha < 0 or alpha > 1:
        raise ValueError(f'alpha must be in [0,1] (got {alpha})')

    # Sample with infinite epsilon
    _, x_draws = sample_theta_x(key, n_samples)
    _, epsilons = vmap(discrepancy_fn)(x_draws)

    # Compute alpha-quantile
    epsilon_quantile = float(jnp.quantile(epsilons, alpha))

    return epsilon_quantile, epsilons


# Core ABC sampling function (JIT compiled) - Fixed version
@partial(jit, static_argnums=(1, 2, 3))
def get_abc_sample(
    key: random.PRNGKey,
    sample_theta_x: Callable,
    # data_simulator: Callable,
    discrepancy_fn: Callable,
    epsilon: float,
    # summary_stat_fn: Optional[Callable],
    # transform_fn: Optional[Callable],
    # observed_data: jnp.ndarray,
    # observed_summary_stats: Optional[jnp.ndarray],
) -> Tuple[
    jnp.ndarray, jnp.ndarray, float, Optional[jnp.ndarray], Optional[jnp.ndarray], int
]:
    """
    Sample single ABC draw using rejection sampling with summary statistics.

    Args:
        key: JAX random key
        simulator: Function to sample from prior and data
        discrepancy_fn: Function to return a summary state and scalar discrepancy
        summary_stat_fn: Optional function to compute summary statistics
        epsilon: ABC tolerance threshold

    Returns:
        Tuple of (simulated_data, theta, distance, summary_statistics, phi, count)
    """

    # Determine what to compare against outside the loop for better JIT performance
    # use_summary_stats = (
    #     summary_stat_fn is not None and observed_summary_stats is not None
    # )
    # comparison_target = observed_summary_stats if use_summary_stats else observed_data

    def should_continue(state: RejectionState) -> bool:
        return state.distance >= epsilon

    def rejection_step(state: RejectionState) -> RejectionState:
        """
        JAX-compatible rejection step using typed dataclass.
        Optimized to minimize conditionals in the loop.
        """
        # key, key_theta, key_data = random.split(state.key, 3)
        key, key_sim = random.split(state.key, 2)
        theta_proposal, data_proposal = sample_theta_x(key)
        # theta_proposal = prior_simulator(key_theta)
        # data_proposal = data_simulator(key_data, theta_proposal)

        # Transform theta to phi if transform function is provided
        # RG: Don't transform here, just keep the draws that are given.
        # phi_proposal = (
        #     transform_fn(theta_proposal) if transform_fn is not None else None
        # )

        # Optimized: choose computation path based on pre-determined flag
        # RG: Just use a user-provided discrepancy function, don't do the
        # summary stat computation here.
        # if use_summary_stats:
        #     summary_proposal = summary_stat_fn(data_proposal)
        #     distance = discrepancy_fn(summary_proposal, comparison_target)
        # else:
        #     summary_proposal = state.summary_stat  # Keep the same structure
        #     distance = discrepancy_fn(data_proposal, comparison_target)

        summary_stat, distance = discrepancy_fn(data_proposal)

        return RejectionState(
            key=key,
            data=data_proposal,
            theta=theta_proposal,
            summary_stat=summary_stat,
            distance=distance,
            count=state.count + 1
        )

    # Initialize with dummy values using typed state
    key, key_theta = random.split(key)
    initial_theta, initial_data = sample_theta_x(key_theta)
    #initial_data = jnp.zeros_like(observed_data).astype(float) # Why?

    # Initialize phi with transform function if provided
    # RG: Don't transform here, just keep the draws that are given.
    # if transform_fn is not None:
    #     initial_phi = transform_fn(initial_theta)
    # else:
    #     initial_phi = None

    # Initialize summary with consistent structure
    # if summary_stat_fn is not None:
    #     # Create dummy summary with same shape as expected output
    #     dummy_summary = summary_stat_fn(initial_data)
    #     initial_summary = jnp.zeros_like(dummy_summary)
    # else:
    #     initial_summary = None
    summary_stat, distance = discrepancy_fn(initial_data)

    initial_state = RejectionState(
        key=key,
        data=initial_data,
        theta=initial_theta,
        summary_stat=summary_stat,
        distance=distance,
        count=0,
    )

    #RG: This should be a standalone function with all the checking done
    #    beforehand.
    final_state = lax.while_loop(should_continue, rejection_step, initial_state)

    # RG: Can't you return a RejectionState object?
    return (
        final_state.data,
        final_state.theta,
        final_state.distance,
        final_state.summary_stat,
        final_state.count
    )


class RejectionSamplerMetadata(NamedTuple):
    """Result structure for rejection sampling operations."""
    distances: jnp.ndarray
    summary_stats: jnp.ndarray
    rejection_count: jnp.ndarray
    key: random.PRNGKey
    n_samples: int

# Inherit StatisticalModel instaed
class RejectionSampler(StatisticalModel):
    """
    ABC rejection sampler implementation.

    This class encapsulates the ABC rejection sampling algorithm
    with support for summary statistics and JIT compilation.
    """

    # RG: This class is a mostly unnecessary wrapper around get_abc_sample.

    def __init__(
        self,
        model: StatisticalModel,
        discrepancy_fn: Callable,
        epsilon: float):
        """
        Initialize rejection sampler.

        Args:
            model: Function to sample from prior and data
            discrepancy_fn: Function to compute summary statistic and distance between datasets
            epsilon: ABC tolerance threshold
        """

        # TODO: call super().__init__ for cacheing support?

        self.model = model
        self.discrepancy_fn = discrepancy_fn
        self.set_epsilon(epsilon)

        self.clear_cache()

        # self.observed_data = observed_data
        # self.observed_summary_stats = observed_summary_stats

        # Precompute observed summary statistics if summary_stat_fn is provided
        # if self.summary_stat_fn is not None and self.observed_data is not None:
        #     if self.observed_summary_stats is None:
        #         self.observed_summary_stats = self.summary_stat_fn(self.observed_data)

    def clear_cache(self):
        self._cache = RejectionSamplerMetadata(
            distances=jnp.array([]),
            summary_stats=jnp.array([]),
            rejection_count=jnp.array([]),
            key=None,
            n_samples=None)

    def get_cache(self, key=None, n_samples=None):
        """
        Query a cached value from a call to sample_theta_x_multiple.
        Optionally pass in the key and n_samples with with sample_theta_x_multiple
        was called to ensure that you are getting the cache for the call you expect.
        """
        if (key is not None) and (not jnp.array_equal(key, self._cache.key)):
            raise ValueError('Called get_cache with a non-matching key')

        if (n_samples is not None) and (n_samples != self._cache.n_samples):
            raise ValueError('Called get_cache with a non-matching n_samples')

        return self._cache

    def get_model_args(self):
        # TODO: fill this out for serialization
        return dict()

    def get_prior_sample(self, key: random.PRNGKey, n_samples: int) -> jnp.ndarray:
        raise NotImplementedError("get_prior_samples not implemented for RejectionSampler")

    def simulate_data(self, key: random.PRNGKey, theta: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("simulate_data not implemented for RejectionSampler")

    def set_epsilon(self, epsilon):
        if epsilon <= 0:
            raise ValueError(f'epsilon must be strictly positive (got {epsilon})')
        self._epsilon = epsilon

    def sample_theta_x(self, key: random.PRNGKey):
        data, theta, distance, summary_stat, count = get_abc_sample(
            key,
            self.model.sample_theta_x,
            # self.data_simulator,
            self.discrepancy_fn,
            # self.summary_stat_fn,
            # self.transform_fn,
            self._epsilon,
            # self.observed_data,
            # self.observed_summary_stats,
        )
        return theta, data

    def sample_theta_x_multiple(self, key: random.PRNGKey, n_samples: int, cache=False):
        """
        Generate multiple ABC samples using vectorized sampling.

        Args:
            key: JAX random key
            n_samples: Number of samples to generate
        """
        keys = random.split(key, n_samples + 1)

        # Vectorize the single sample function
        vectorized_sampler = vmap(get_abc_sample, in_axes=(0, None, None, None))

        (data,
         theta_samples,
         distances,
         summary_stats,
         rejection_count) = \
            vectorized_sampler(
                keys[1:],
                self.model.sample_theta_x,
                self.discrepancy_fn,
                self._epsilon)

        # # Ensure phi_samples is 2D for consistency
        # if phi_samples is not None and phi_samples.ndim == 1:
        #     phi_samples = phi_samples[:, None]

        if cache:
            self._cache = RejectionSamplerMetadata(
                distances=distances,
                summary_stats=summary_stats,
                rejection_count=rejection_count,
                key=key,
                n_samples=n_samples)

        return theta_samples, data
        # return ABCSampleResult(
        #     data=data,
        #     theta=theta_samples,
        #     distances=distances,
        #     summary_stats=summary_stats,
        #     key=keys[0],
        #     simulation_count=rejection_count,
        #     phi=phi_samples,
        # )

    # def sample_single(self, key: random.PRNGKey) -> ABCSingleResult:
    #     """
    #     Generate single ABC sample using rejection sampling.

    #     Args:
    #         key: JAX random key

    #     Returns:
    #         ABCSingleResult with data, theta, distance, summary_stat, and phi
    #     """
    #     data, theta, distance, summary_stat, phi, count = get_abc_sample(
    #         key,
    #         self.prior_simulator,
    #         self.data_simulator,
    #         self.discrepancy_fn,
    #         self.summary_stat_fn,
    #         self.transform_fn,
    #         self.epsilon,
    #         self.observed_data,
    #         self.observed_summary_stats,
    #     )

    #     return ABCSingleResult(
    #         sim_data=data,
    #         theta=theta,
    #         distance=distance,
    #         summary_stat=summary_stat,
    #         phi=phi,
    #     )

    # def sample(self, key: random.PRNGKey, n_samples: int) -> ABCSampleResult:
    #     """
    #     Generate multiple ABC samples using vectorized sampling.

    #     Args:
    #         key: JAX random key
    #         n_samples: Number of samples to generate

    #     Returns:
    #         ABCSampleResult with all sampling results including phi_samples
    #     """
    #     keys = random.split(key, n_samples + 1)

    #     # Vectorize the single sample function
    #     vectorized_sampler = vmap(
    #         get_abc_sample, in_axes=(0, None, None, None, None, None, None, None, None)
    #     )

    #     (
    #         data,
    #         theta_samples,
    #         distances,
    #         summary_stats,
    #         phi_samples,
    #         rejection_count,
    #     ) = vectorized_sampler(
    #         keys[1:],
    #         self.prior_simulator,
    #         self.data_simulator,
    #         self.discrepancy_fn,
    #         self.summary_stat_fn,
    #         self.transform_fn,
    #         self.epsilon,
    #         self.observed_data,
    #         self.observed_summary_stats,
    #     )

    #     # Ensure phi_samples is 2D for consistency
    #     if phi_samples is not None and phi_samples.ndim == 1:
    #         phi_samples = phi_samples[:, None]

    #     return ABCSampleResult(
    #         data=data,
    #         theta=theta_samples,
    #         distances=distances,
    #         summary_stats=summary_stats,
    #         key=keys[0],
    #         simulation_count=rejection_count,
    #         phi=phi_samples,
    #     )



def generate_nre_training_samples(
        sample_theta_x_multiple: Callable,
        key: random.PRNGKey, n_samples: int):
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

    # RG: TODO: change this so it's not a method of the rejection sampler
    half_samples = n_samples // 2

    # Generate base ABC samples from the approximate posterior
    abc_result = self.sample(key, half_samples)
    key = abc_result.key  # Use the updated key from the result

    key, perm_key = random.split(key)
    data = abc_result.data
    permutation_indices = random.permutation(perm_key, jnp.arange(half_samples))

    phi = abc_result.phi
    theta = abc_result.theta
    phi_permuted = (
        abc_result.phi[permutation_indices] if abc_result.phi is not None else None
    )
    theta_permuted = abc_result.theta[permutation_indices]

    summary_stats = (
        abc_result.summary_stats if abc_result.summary_stats is not None else None
    )

    distances = abc_result.distances

    training_data = jnp.concatenate([data, data], axis=0)
    training_summary_stats = (
        jnp.concatenate([summary_stats, summary_stats], axis=0)
        if summary_stats is not None
        else None
    )

    training_phi = (
        jnp.concatenate([phi_permuted, phi], axis=0)
        if phi_permuted is not None
        else None
    )
    training_theta = jnp.concatenate([theta_permuted, theta], axis=0)
    training_distances = jnp.concatenate([distances, distances], axis=0)

    training_labels = jnp.concatenate(
        [jnp.zeros(half_samples, dtype=int), jnp.ones(half_samples, dtype=int)]
    )

    return ABCTrainingResult(
        labels=training_labels,
        data=training_data,
        distances=training_distances,
        summary_stats=training_summary_stats,
        key=key,
        theta=training_theta,
        phi=training_phi,
        total_sim_count=int(abc_result.simulation_count.sum()),
    )

# def update_epsilon(self, new_epsilon: float):
#     """Update epsilon value."""
#     self.epsilon = new_epsilon

# def update_observed_data(self, new_observed_data: jnp.ndarray):
#     """Update observed data and recompute summary statistics if needed."""
#     self.observed_data = new_observed_data

#     if self.summary_stat_fn is not None:
#         self.observed_summary_stats = self.summary_stat_fn(new_observed_data)


# Export all functions
__all__ = [
    "RejectionSampler",
    "get_abc_sample",
]
