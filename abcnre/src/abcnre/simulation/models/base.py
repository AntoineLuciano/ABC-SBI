"""
Base class for statistical models in ABC simulation.

This module defines the interface that all statistical models must implement
to be used with the ABCSimulator.
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import random, vmap
from typing import Callable, Dict, Any, Optional, Tuple


# RG: It would be better to call this a "sampler."  A statistical
# model does sampling and more.  Currently the naming is the opposite.
class StatisticalModel(ABC):
    """
    Abstract base class for all statistical models used in ABC.
    
    This class defines the interface that statistical models must implement
    to be compatible with the ABCSimulator and RejectionSampler.
    
    All models must implement:
    - prior_sample(): Sample parameters from prior distribution
    - simulate(): Simulate data given parameters
    - discrepancy_fn(): Compute distance between datasets
    - get_model_args(): Return arguments for model reconstruction
    
    Optional methods:
    - summary_stat_fn(): Compute summary statistics
    - transform_phi(): Transform parameters to target parameter
    """
    
    @abstractmethod
    def get_prior_sample(self, key: random.PRNGKey) -> jnp.ndarray:
        """
        Sample parameters from prior distribution.
        
        Args:
            key: JAX random key
            
        Returns:
            Parameter sample from prior (can be scalar or vector)
        """
        pass
    
    def get_prior_samples(self, key: random.PRNGKey, n_samples: int) -> jnp.ndarray:
        """
        Draws multiple samples from the prior distribution efficiently.

        This method leverages JAX's vectorization capabilities (`vmap`) to
        draw multiple samples in a single, efficient operation.

        Args:
            key: A single JAX random key.
            n_samples: The number of samples to draw from the prior.

        Returns:
            An array of parameter sets of shape (n_samples, n_params).
        """
        keys = random.split(key, n_samples)
        
        return vmap(self.get_prior_sample)(keys)
    
    
    @abstractmethod
    def simulate_data(self, key: random.PRNGKey, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Simulate data given parameters.
        
        Args:
            key: JAX random key
            theta: Parameter values
            
        Returns:
            Simulated dataset
        """
        pass
    
    def simulate_datas(self, key: random.PRNGKey, theta: jnp.ndarray):
        """
        Simulate multiple datasets given parameters.
        
        Args:
            key: JAX random key
            theta: Parameter values.  If 1d, a single sample is drawn.
                   If 2d, the first dimension is the number of samples.
            
        Returns:
            Array of simulated datasets of shape (n_samples, data_shape)
        """
        print('theta.shape = ', theta.shape)
        if theta.ndim > 2:
            raise ValueError(f'theta must be 1d or 2d (got shape = {theta.shape})')
        n_samples = theta.shape[0] if theta.ndim > 1 else 1
        keys = random.split(key, n_samples)
        return vmap(lambda k: self.simulate_data(k, theta))(keys)
    
    def sample_theta_x(self, key: random.PRNGKey):
        """
        Sample theta and simulate data in one step.

        Args:
            key: JAX random key

        Returns:
            Simulated dataset
        """
        key_theta, key_data = random.split(key)
        theta = self.get_prior_sample(key_theta)
        return theta, self.simulate_data(key_data, theta)

    def sample_theta_x_multiple(
        self, key: random.PRNGKey,  n_samples: int
    ) -> jnp.ndarray:
        """
        Sample multiple theta and simulate data in one step.

        Args:
            key: JAX random key
            n_samples: Number of samples to draw

        Returns:
            Array of simulated datasets of shape (n_samples, data_shape)
        """
        keys = random.split(key, n_samples)
        return vmap(lambda k: self.sample_theta_x(k))(keys)
    
    @abstractmethod
    def get_model_args(self) -> Dict[str, Any]:
        """
        Get model-specific arguments for serialization.
        
        Returns:
            Dictionary of arguments needed to recreate the model
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_class': self.__class__.__name__,
            'model_module': self.__module__,
            #'has_summary_stats': self.has_summary_stats(),
            'model_args': self.get_model_args()
        }
    
    def validate_parameters(self, theta: jnp.ndarray) -> bool:
        """
        Validate parameter values (optional).
        
        Override this method to add parameter validation,
        e.g., checking positivity constraints.
        
        Args:
            theta: Parameter values to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        # Default: all parameters are valid
        return True
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}({self.get_model_args()})"





# Export main class
__all__ = ["StatisticalModel"]



