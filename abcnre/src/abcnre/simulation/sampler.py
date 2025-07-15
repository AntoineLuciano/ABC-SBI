"""
ABC sampling methods.

This module contains the implementation of ABC rejection sampling
and other sampling strategies, migrated from the original sampling.py.
"""

from jax import random, jit, vmap, lax
import jax.numpy as jnp
from typing import Callable, Tuple, Optional
from functools import partial

from .base import BaseSampler, ABCSampleResult, ABCTrainingResult, ABCSingleResult


# Core ABC sampling function (JIT compiled)  
@partial(jit, static_argnums=(1, 2, 3, 4, 8))
def get_abc_sample(
    key: random.PRNGKey,
    prior_simulator: Callable,
    data_simulator: Callable,
    discrepancy_fn: Callable,
    summary_stat_fn: Optional[Callable],
    epsilon: float,
    observed_data: jnp.ndarray,
    observed_summary_stats: Optional[jnp.ndarray],
    n_obs: int
) -> Tuple[jnp.ndarray, jnp.ndarray, float, Optional[jnp.ndarray]]:
    """
    Sample single ABC draw using rejection sampling with summary statistics.
    
    Args:
        key: JAX random key
        prior_simulator: Function to sample from prior
        data_simulator: Function to simulate data given parameters  
        discrepancy_fn: Function to compute distance between datasets
        summary_stat_fn: Optional function to compute summary statistics
        epsilon: ABC tolerance threshold
        observed_data: Observed dataset
        observed_summary_stats: Pre-computed observed summary statistics
        n_obs: Number of observations
        
    Returns:
        Tuple of (simulated_data, theta, distance, summary_statistics)
    """
    
    # Determine what to compare against
    if summary_stat_fn is not None and observed_summary_stats is not None:
        comparison_target = observed_summary_stats
    else:
        comparison_target = observed_data
    
    def should_continue(state):
        *_, distance, _ = state
        return distance >= epsilon
    
    def rejection_step(state):
        #! This is error prone, better to use a named tuple for state
        key, sim_data, theta, summary_stat, distance, count = state # Add count
        
        key, key_theta, key_data = random.split(key, 3)
        
        theta_proposal = prior_simulator(key_theta)
        data_proposal = data_simulator(key_data, theta_proposal, n_obs)
        
        # Use summary statistics if provided, otherwise raw data
        if summary_stat_fn is not None:
            summary_proposal = summary_stat_fn(data_proposal)
            distance = discrepancy_fn(summary_proposal, comparison_target)
        else:
            summary_proposal = summary_stat  # Keep the same structure (None or dummy)
            distance = discrepancy_fn(data_proposal, comparison_target)
        count += 1 # Increment count
        return key, data_proposal, theta_proposal, summary_proposal, distance, count
    
    # Initialize with dummy values - ensure consistent structure
    key, key_theta = random.split(key)
    initial_theta = prior_simulator(key_theta)
    initial_data = jnp.zeros_like(observed_data).astype(float)
    
    # Initialize summary with consistent structure
    if summary_stat_fn is not None:
        # Create dummy summary with same shape as expected output
        dummy_summary = summary_stat_fn(initial_data)
        initial_summary = jnp.zeros_like(dummy_summary)
    else:
        initial_summary = None
        
    initial_distance = epsilon + 1.0
    initial_count = 0
    
    final_state = lax.while_loop(
        should_continue,
        rejection_step,
        (key, initial_data, initial_theta, initial_summary, initial_distance, initial_count) # Add count
    )
    
    _, final_data, final_theta, final_summary, final_distance, final_count = final_state
    return final_data, final_theta, final_distance, final_summary, final_count # Return count


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
        epsilon: float = 0.1,
        observed_data: Optional[jnp.ndarray] = None,
        observed_summary_stats: Optional[jnp.ndarray] = None
    ):
        """
        Initialize rejection sampler.
        
        Args:
            prior_simulator: Function to sample from prior
            data_simulator: Function to simulate data given parameters  
            discrepancy_fn: Function to compute distance between datasets
            summary_stat_fn: Optional function to compute summary statistics
            epsilon: ABC tolerance threshold
            observed_data: Observed dataset
            observed_summary_stats: Pre-computed observed summary statistics
        """
        self.prior_simulator = prior_simulator
        self.data_simulator = data_simulator
        self.discrepancy_fn = discrepancy_fn
        self.summary_stat_fn = summary_stat_fn
        self.epsilon = epsilon
        self.observed_data = observed_data
        self.observed_summary_stats = observed_summary_stats
    
    def sample_single(self, key: random.PRNGKey) -> ABCSingleResult:
        """
        Generate single ABC sample using rejection sampling.
        
        Args:
            key: JAX random key
            
        Returns:
            ABCSingleResult with sim_data, theta, distance, and summary_stat
        """
        n_obs = self.observed_data.shape[0]
        sim_data, theta, distance, summary_stat = get_abc_sample(
            key, self.prior_simulator, self.data_simulator,
            self.discrepancy_fn, self.summary_stat_fn, self.epsilon,
            self.observed_data, self.observed_summary_stats, n_obs
        )
        
        return ABCSingleResult(
            sim_data=sim_data,
            theta=theta,
            distance=distance,
            summary_stat=summary_stat
        )
    
    def sample(self, key: random.PRNGKey, n_samples: int) -> ABCSampleResult:
        """
        Generate multiple ABC samples using vectorized sampling.
        
        Args:
            key: JAX random key
            n_samples: Number of samples to generate
            
        Returns:
            ABCSampleResult with all sampling results
        """
        keys = random.split(key, n_samples + 1)
        n_obs = self.observed_data.shape[0]
        
        # Vectorize the single sample function
        vectorized_sampler = vmap(
            get_abc_sample,
            in_axes=(0, None, None, None, None, None, None, None, None)
        )
        
        sim_data, theta_samples, distances, summary_stats, rejection_count = vectorized_sampler(
            keys[1:], self.prior_simulator, self.data_simulator, 
            self.discrepancy_fn, self.summary_stat_fn, self.epsilon,
            self.observed_data, self.observed_summary_stats, n_obs
        )
        
        return ABCSampleResult(
            sim_data=sim_data,
            theta_samples=theta_samples,
            distances=distances,
            summary_stats=summary_stats,
            key=keys[0],
            rejection_count=rejection_count
        )
    
    def generate_training_samples(
        self,
        key: random.PRNGKey,
        n_samples: int,
        transform_fn: Callable[[jnp.ndarray], jnp.ndarray]
    ) -> ABCTrainingResult:
        """
        Generates a training dataset for Neural Ratio Estimation.

        This method creates a balanced dataset for NRE by:
        1.  Sampling `theta` and summary statistics `z` from the ABC posterior.
        2.  Applying a transformation to `theta` to get the parameter of interest, `phi`.
        3.  Creating "class 0" samples (phi, z) from the joint distribution.
        4.  Creating "class 1" samples by permuting `phi` to break the dependency,
            sampling from the product of the marginals.

        Args:
            key: A JAX random key for reproducibility.
            n_samples: The total number of samples to generate for the training
                set. This will be split equally between class 0 and class 1.
            transform_fn: A function that transforms the raw model parameters `theta`
                          to the parameter(s) of interest `phi`.

        Returns:
            An ABCTrainingResult named tuple with training features, labels, and metadata.
        """
        half_samples = n_samples // 2

        # Generate base ABC samples from the approximate posterior
        abc_result = self.sample(key, half_samples)
        key = abc_result.key  # Use the updated key from the result

        # Determine the features `z` (either summary stats or raw data)
        if abc_result.summary_stats is not None:
            features_z = abc_result.summary_stats
        else:
            # Reshape data to be a flat feature vector per sample
            features_z = abc_result.sim_data.reshape(half_samples, -1)
        
        # Apply the transformation to get the parameter of interest, phi
        phi_samples = transform_fn(abc_result.theta_samples)
        
        # Ensure phi_samples is 2D for concatenation, e.g., (n_samples, n_phi)
        if phi_samples.ndim == 1:
            phi_samples = phi_samples[:, None]

        total_sim_count = abc_result.rejection_count.sum()
        
        # Create the marginal samples by permuting the phi samples
        key, perm_key = random.split(key)
        phi_permuted = phi_samples[
            random.permutation(perm_key, phi_samples.shape[0])
        ]

        # Combine joint and marginal samples
        # Features for the classifier are [phi, z]
        features_joint = jnp.concatenate([phi_samples, features_z], axis=1)
        features_marginal = jnp.concatenate([phi_permuted, features_z], axis=1)
        
        all_features = jnp.concatenate([features_joint, features_marginal], axis=0)
        labels = jnp.concatenate([
            jnp.zeros(half_samples, dtype=int),
            jnp.ones(half_samples, dtype=int)
        ])

        return ABCTrainingResult(
            features=all_features,
            labels=labels,
            distances=abc_result.distances,
            summary_stats=abc_result.summary_stats,
            key=key,
            phi_samples=phi_samples, 
            total_sim_count=int(total_sim_count) 
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
        self,
        key: random.PRNGKey,
        alpha: float,
        n_samples: int = 10000
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


# Backward compatibility functions
def get_abc_samples_vectorized(
    key: random.PRNGKey,
    n_samples: int,
    prior_simulator: Callable,
    data_simulator: Callable,
    discrepancy_fn: Callable,
    summary_stat_fn: Optional[Callable],
    epsilon: float,
    observed_data: jnp.ndarray,
    observed_summary_stats: Optional[jnp.ndarray] = None
) -> ABCSampleResult:
    """
    Backward compatibility function for get_abc_samples_vectorized.
    """
    sampler = RejectionSampler(
        prior_simulator, data_simulator, discrepancy_fn,
        summary_stat_fn, epsilon, observed_data, observed_summary_stats
    )
    return sampler.sample(key, n_samples)


def get_training_samples(
    key: random.PRNGKey,
    n_samples: int,
    prior_simulator: Callable,
    data_simulator: Callable,
    discrepancy_fn: Callable,
    summary_stat_fn: Optional[Callable],
    epsilon: float,
    observed_data: jnp.ndarray,
    observed_summary_stats: Optional[jnp.ndarray] = None,
    marginal_index: int = 0
) -> ABCTrainingResult:
    """
    Backward compatibility function for get_training_samples.
    """
    sampler = RejectionSampler(
        prior_simulator, data_simulator, discrepancy_fn,
        summary_stat_fn, epsilon, observed_data, observed_summary_stats
    )
    return sampler.generate_training_samples(key, n_samples, marginal_index)


# Export all functions
__all__ = [
    "RejectionSampler",
    "get_abc_sample",
    "get_abc_samples_vectorized", 
    "get_training_samples"
]