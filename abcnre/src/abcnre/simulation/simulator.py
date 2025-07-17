"""
Main ABC simulator class.

This module contains the ABCSimulator class which is the main interface
for ABC simulation.
"""

import jax.numpy as jnp
from jax import random
from typing import Callable, Tuple, Any, Dict, Optional, Union
from jax import numpy as jnp
from pathlib import Path

from .sampler import RejectionSampler
from .base import ABCSampleResult, ABCTrainingResult, ABCSingleResult
from . import utils 
from functools import cached_property
from .utils import generate_sampler_hash
import jax
class ABCSimulator:
    """
    Main class for ABC simulation and data generation.
    
    This class handles:
    - ABC rejection sampling through a RejectionSampler.
    - Epsilon management and quantile computation.
    - Provides a high-level interface for generating simulation data and persistence.
    """
    
    def __init__(
        self,
        model=None,
        observed_data: Optional[jnp.ndarray] = None,
        epsilon: Optional[float] = None,
        quantile_distance: Optional[float] = None,
        config: Optional[Dict] = None
    ):
        """
        Initializes the ABC simulator.
        
        Args:
            model: A statistical model instance with a defined interface.
            observed_data: The observed dataset to match against.
            epsilon: The ABC tolerance threshold. Defaults to infinity.
            quantile_distance: If provided (0,1), automatically computes epsilon 
                             as this quantile of the distance distribution.
            config: An optional dictionary for advanced ABC parameters.
        """
        self.model = model
        self.observed_data = observed_data
        
        # Set default configuration and override with user-provided config
        self.config = {
            'quantile_n_samples': 10000,
            'verbose': True,
            'summary_stats_enabled': False,
            **(config if config is not None else {})
        }
        
        self.quantile_distance = quantile_distance
        self.epsilon = jnp.inf if epsilon is None else epsilon
        
        # Initialize observed summary statistics if applicable
        self._init_summary_stats()
            
        # Initialize the underlying sampler
        self.sampler = None
        if self.model is not None and self.observed_data is not None:
            self._initialize_sampler()
            self._init_epsilon()

    
    @cached_property
    def sampler_id(self) -> str:
        """A unique, deterministic identifier for this simulator configuration."""
        if self.model is None or self.observed_data is None:
            raise ValueError("Model and observed_data must be set to generate a sampler ID.")
        
        model_config = getattr(self.model, "get_model_args", lambda: {})()
        
        return generate_sampler_hash(
            model_config=model_config,
            observed_data=self.observed_data,
            epsilon=self.epsilon
        )
        
    def _init_summary_stats(self):
        """Initializes summary statistics based on the model and data."""
        self.observed_summary_stats = None
        self.config['summary_stats_enabled'] = False
        if self.model is not None and self.observed_data is not None and hasattr(self.model, 'summary_stat_fn'):
            try:
                self.observed_summary_stats = self.model.summary_stat_fn(self.observed_data)
                self.config['summary_stats_enabled'] = True
            except (AttributeError, NotImplementedError):
                # Model is expected to have the function but it's not implemented
                pass
    
    def _init_epsilon(self):
        """Initializes epsilon, computing it from a quantile if requested."""
        if self.quantile_distance is not None:
            if not (0 < self.quantile_distance <= 1):
                raise ValueError("quantile_distance must be between 0 and 1")
            
            if self.config.get('verbose', False):
                print(f"Computing epsilon for {self.quantile_distance:.1%} quantile...")
            if self.quantile_distance == 1.0:
                # If quantile_distance is 1.0, set epsilon to the maximum distance
                self.epsilon = jnp.inf
                if self.sampler:
                    self.sampler.update_epsilon(self.epsilon)
                print("Setting epsilon to infinity (maximum distance).")
            # Use a fixed key for this internal, automatic setup
            key = random.PRNGKey(0) 
            computed_epsilon, _, _ = self.get_epsilon_quantile(
                key, self.quantile_distance, self.config['quantile_n_samples']
            )
            self.epsilon = computed_epsilon
            if self.sampler:
                self.sampler.update_epsilon(self.epsilon)
            
            if self.config.get('verbose', False):
                print(f"Computed epsilon = {self.epsilon:.6f}")

    def _initialize_sampler(self):
        """Initialize the rejection sampler with the current configuration."""
        if self.model is None or self.observed_data is None:
            raise ValueError("Model and observed_data must be set before initializing sampler.")
        
        summary_stat_fn = getattr(self.model, 'summary_stat_fn', None)
        
        self.sampler = RejectionSampler(
            prior_simulator=self.model.prior_sample,
            data_simulator=self.model.simulate,
            discrepancy_fn=self.model.discrepancy_fn,
            summary_stat_fn=summary_stat_fn,
            epsilon=self.epsilon,
            observed_data=self.observed_data,
            observed_summary_stats=self.observed_summary_stats
        )

    def save(self, config_path: Union[str, Path]):
        """
        Saves the simulator's configuration and data.

        This is a convenience wrapper around the `save_simulator` utility function.
        
        Args:
            config_path: The path where the YAML configuration file will be saved.
        """
        utils.save_simulator(self, config_path)

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> 'ABCSimulator':
        """
        Loads a simulator from a configuration file.
        
        This is a convenience wrapper around the `load_simulator` utility function.

        Args:
            config_path: Path to the simulator's YAML configuration file.

        Returns:
            A new, fully instantiated ABCSimulator instance.
        """
        return utils.load_simulator(config_path)

    def generate_samples(self, key: random.PRNGKey, n_samples: int) -> ABCSampleResult:
        """
        Generates multiple ABC samples using vectorized sampling.
        
        Args:
            key: A JAX random key.
            n_samples: The number of samples to generate.
            
        Returns:
            An ABCSampleResult named tuple with all sampling results.
        """
        if self.sampler is None: self._initialize_sampler()
        return self.sampler.sample(key, n_samples)
    
    def generate_training_samples(self, key: random.PRNGKey, n_samples: int, phi_transform: Optional[Callable] = None) -> ABCTrainingResult:
        """
        Generates a training dataset for Neural Ratio Estimation.
        
        Args:
            key: A JAX random key.
            n_samples: The total number of samples for the training set.
            phi_transform: An optional function to transform theta to the parameter
                         of interest, phi. If None, the model's default is used.
            
        Returns:
            An ABCTrainingResult with features, labels, and metadata.
        """
        if self.sampler is None: self._initialize_sampler()
        
        # Use the provided transform or default to the one on the model
        transform = phi_transform if phi_transform is not None else getattr(self.model, 'transform_phi', lambda theta: theta)

        return self.sampler.generate_training_samples(key, n_samples, transform)

    def get_epsilon_quantile(self, key: random.PRNGKey, alpha: float, n_samples: int = 10000) -> Tuple[float, jnp.ndarray, random.PRNGKey]:
        """
        Gets the epsilon value for a given quantile of the distance distribution.
        
        Args:
            key: A JAX random key.
            alpha: The quantile level (e.g., 0.1 for the 10th percentile).
            n_samples: Number of simulations to estimate the distribution.
            
        Returns:
            A tuple of (epsilon_quantile, all_distances, updated_key).
        """
        if self.sampler is None: self._initialize_sampler()
        return self.sampler.get_epsilon_quantile(key, alpha, n_samples)

    def __repr__(self) -> str:
        """Provides a clean string representation of the ABCSimulator."""
        model_name = self.model.__class__.__name__ if self.model else "None"
        obs_shape = self.observed_data.shape if self.observed_data is not None else "None"
        
        return (f"ABCSimulator(model={model_name}, "
                f"epsilon={self.epsilon:.4f}, observed_data_shape={obs_shape})")
    
    def get_true_posterior_samples(self, key: 'jax.random.PRNGKey', n_samples: int) -> jnp.ndarray:
        """
        Draws samples from the true analytical posterior, if available.

        This method relies on the underlying model having a method to sample
        from its analytical posterior (e.g., for conjugate models).

        Args:
            key: A JAX random key.
            n_samples: The number of samples to draw.

        Returns:
            An array of samples from the true posterior.
        
        Raises:
            NotImplementedError: If the model does not support analytical sampling.
        """
        if not hasattr(self.model, 'get_posterior_distribution'):
            raise NotImplementedError("The current model does not have a method for analytical posterior sampling.")
        
        seed = int(random.randint(key, (), 0, jnp.iinfo(jnp.int32).max))
        return self.model.get_posterior_distribution(self.observed_data).rvs(size=n_samples, random_state=seed)

# Backward compatibility alias
ABCDataGenerator = ABCSimulator

# Export main components
__all__ = [
    "ABCSimulator",
    "ABCDataGenerator"
]