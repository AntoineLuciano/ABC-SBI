"""
ABC sampling methods.

This module contains the implementation of ABC rejection sampling
and other sampling strategies, migrated from the original sampling.py.
"""

from jax import random, jit, vmap, lax, tree_util
import jax.numpy as jnp
from typing import Callable, Tuple, Optional
from functools import partial, cached_property
from dataclasses import dataclass

from typing import NamedTuple, Optional, List, Dict
#from .base import ABCSampleResult, ABCTrainingResult, ABCSingleResult
from .models.base import StatisticalModel



######################################################################
#--------------------------------------------------------------------#
######################################################################




class SummarizedStatisticalModel(StatisticalModel):
    """
    Draw from a statistical model but using a summary of the parameter
    given by summary_fn.
    """
    def __init__(
            self,
            model: StatisticalModel,
            summary_fn: Callable[[jnp.ndarray], jnp.ndarray]):

        self.model = model
        self.summary_fn = summary_fn

    def get_prior_sample(self, key: random.PRNGKey) -> jnp.ndarray:
        phi = self.summary_fn(self.model.get_prior_sample(key))
        return phi

    def get_prior_samples(self, key: random.PRNGKey, n_samples: int) -> jnp.ndarray:
        theta = self.model.get_prior_samples(key, n_samples)
        phi = vmap(self.summary_fn)(theta)
        return phi
        # Cannot use the super because sometimes the model uses different
        # drawing schemes for a single draw and for multiple draws
        #return super().get_prior_samples(key, n_samples)

    def sample_theta_x(self, key: random.PRNGKey) -> jnp.ndarray:
        theta, x = self.model.sample_theta_x(key)
        return self.summary_fn(theta), x

    def sample_theta_x_multiple(
            self, key: random.PRNGKey, n_samples: int) -> jnp.ndarray:
        # Cannot use the super because sometimes the model uses different
        # drawing schemes for a single draw and for multiple draws
        # return super().sample_theta_x_multiple(key, n_samples)
        theta, x = self.model.sample_theta_x_multiple(key, n_samples)
        phi = vmap(self.summary_fn)(theta)
        return phi, x

    def simulate_data(self, key: random.PRNGKey, theta: jnp.ndarray) -> jnp.ndarray:
        self.model.simulate_data(key, theta)

    def simulate_datas(self, key: random.PRNGKey, theta: jnp.ndarray) -> jnp.ndarray:
        self.model.simulate_datas(key, theta)

    def get_model_args(self):
        # TODO: annotate the summary function, too
        return self.model.get_model_args()





######################################################################
#--------------------------------------------------------------------#
######################################################################



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


# Core ABC sampling function (JIT compiled) - Fixed version
@partial(jit, static_argnums=(1, 2, 3))
def get_abc_sample(
    key: random.PRNGKey,
    sample_theta_x: Callable,
    discrepancy_fn: Callable,
    epsilon: float,
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
        key, key_sim = random.split(state.key, 2)
        theta_proposal, data_proposal = sample_theta_x(key)

        summary_stat, distance = discrepancy_fn(data_proposal)

        return RejectionState(
            key=key,
            data=data_proposal,
            theta=theta_proposal,
            summary_stat=summary_stat,
            distance=distance,
            count=state.count + 1
        )

    key, key_theta = random.split(key)
    initial_theta, initial_data = sample_theta_x(key_theta)
    summary_stat, distance = discrepancy_fn(initial_data)

    initial_state = RejectionState(
        key=key,
        data=initial_data,
        theta=initial_theta,
        summary_stat=summary_stat,
        distance=distance,
        count=0,
    )

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
    empty: bool

class RejectionSampler(StatisticalModel):
    """
    ABC rejection sampler implementation.

    This class encapsulates the ABC rejection sampling algorithm
    with support for summary statistics and JIT compilation.
    """

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

    def clear_cache(self):
        self._cache = RejectionSamplerMetadata(
            distances=jnp.array([]),
            summary_stats=jnp.array([]),
            rejection_count=jnp.array([]),
            key=None,
            n_samples=None,
            empty=True)

    def get_cache(self, key=None, n_samples=None):
        """
        Query a cached value from a call to sample_theta_x_multiple.
        Optionally pass in the key and n_samples with with sample_theta_x_multiple
        was called to ensure that you are getting the cache for the call you expect.
        """
        if self._cache.empty:
            raise ValueError('The cache is empty.  Did you set cache=True in the sampler?')

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
            self.discrepancy_fn,
            self._epsilon,
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
                n_samples=n_samples,
                empty=False)

        return theta_samples, data




######################################################################
#--------------------------------------------------------------------#
######################################################################


# RG: I'm not sure what the ABCSimulator class is adding beyond RejectionSampler
# What is the difference between a simulator and a sampler?


# RG: I am pretty sure this class is unnecessary, or at least
# overly complicated.
class ABCSimulator:
    """
    Main class for ABC simulation and data generation.

    This class handles:
    - ABC rejection sampling through a RejectionSampler.
    - Epsilon management and quantile computation.
    - Provides a high-level interface for generating simulation data and persistence.
    """

    def __init__(self):
        raise NotImplementedError()



# Export all functions
__all__ = [
    "RejectionSampler",
    "SummarizedStatisticalModel",
    "ABCSimulator", # TODO: deprecate
]

