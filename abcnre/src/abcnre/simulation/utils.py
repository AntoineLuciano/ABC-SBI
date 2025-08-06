"""
Utilities for working with rejection samplers and NNs

"""

import jax.numpy as jnp
from jax import random
from typing import Callable, Tuple, Any, Dict, Optional, Union
from jax import numpy as jnp
from pathlib import Path
import yaml

from .samplers import RejectionSampler
#from .base import ABCSampleResult, ABCTrainingResult, ABCSingleResult
from functools import cached_property
from .io import generate_sampler_hash
from ..utils.comparison import are_simulators_equivalent
import jax
import flax.linen as nn

from ..training import (
    NNConfig,
    get_nn_config,
    train_regressor,
    TrainingConfig,
    NetworkConfig,
)


def create_summary_stats_fn(
    network: nn.Module,
    params: Dict[str, Any]
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Create summary statistics function adapted to different network architectures.

    Args:
        network: Trained summary statistics network
        params: Trained parameters
        network_type: Type of network ("deepset", "conditioned_deepset", "MLP")

    Returns:
        Summary statistics function that takes observations and returns summary stats
    """

    def summary_stats_fn(x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute summary statistics S(x) from observations.

        Args:
            x: Observations, shape (batch, n, d) or (batch, d)

        Returns:
            Summary statistics, shape (batch, summary_dim)
        """
        if x.ndim == 2:
            x = x[None, :, :]  # (1, n, d)
        summary_stats = network.apply(params, x, training=False)

        return summary_stats  # (batch, summary_dim)

    return summary_stats_fn





# RG: TODO: refactor this with the new framework
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

    # return ABCTrainingResult(
    #     labels=training_labels,
    #     data=training_data,
    #     distances=training_distances,
    #     summary_stats=training_summary_stats,
    #     key=key,
    #     theta=training_theta,
    #     phi=training_phi,
    #     total_sim_count=int(abc_result.simulation_count.sum()),
    # )




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
    _, epsilons = jax.vmap(discrepancy_fn)(x_draws)

    # Compute alpha-quantile
    epsilon_quantile = float(jnp.quantile(epsilons, alpha))

    return epsilon_quantile, epsilons



# Create data generator that matches the expected interface for neural networks.
def get_io_generator(sample_theta_x_multiple: Callable):
    def io_generator(key: random.PRNGKey, batch_size: int):
        """Adapter for the unified training interface."""
        phi, x = sample_theta_x_multiple(key, batch_size)
        return {"input": x, "output": phi, "n_simulations": batch_size}
    return io_generator



__all__ = [
    "get_io_generator",
    "get_epsilon_quantile",
    "create_summary_stats_fn"
]
