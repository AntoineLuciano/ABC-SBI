"""
Base class for statistical models in ABC simulation.

This module defines the interface that all statistical models must implement
to be used with the ABCSimulator.
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import random, vmap
from typing import Dict, Any, Optional


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
    def prior_sample(self, key: random.PRNGKey) -> jnp.ndarray:
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
        # 1. Split the main key into n_samples sub-keys
        keys = random.split(key, n_samples)
        
        # 2. Use jax.vmap to apply prior_sample across all keys in parallel
        # This is significantly faster than a Python for-loop.
        return vmap(self.prior_sample)(keys)
    
    
    @abstractmethod
    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray, n_obs: int) -> jnp.ndarray:
        """
        Simulate data given parameters.
        
        Args:
            key: JAX random key
            theta: Parameter values
            n_obs: Number of observations to simulate
            
        Returns:
            Simulated dataset
        """
        pass
        
    @abstractmethod
    def discrepancy_fn(self, data1: jnp.ndarray, data2: jnp.ndarray) -> float:
        """
        Compute distance/discrepancy between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Distance/discrepancy value (scalar)
        """
        pass
        
    @abstractmethod
    def get_model_args(self) -> Dict[str, Any]:
        """
        Get model-specific arguments for serialization.
        
        Returns:
            Dictionary of arguments needed to recreate the model
        """
        pass
    
    def summary_stat_fn(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Compute summary statistics from data (optional).
        
        Override this method if you want to use summary statistics
        instead of raw data for ABC comparison.
        
        Args:
            data: Input data
            
        Returns:
            Summary statistics
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("summary_stat_fn not implemented")
    
    def transform_phi(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Transform theta to phi (target parameter of interest).
        
        Override this method if you want to focus inference on a specific
        transformation or marginal of the parameter vector.
        
        Args:
            theta: Full parameter vector
            
        Returns:
            Transformed parameter phi (typically scalar)
        """
        # Default: return first component as scalar
        if jnp.isscalar(theta):
            return theta
        else:
            return theta[0]
    
    def has_summary_stats(self) -> bool:
        """
        Check if summary statistics function is implemented.
        
        Returns:
            True if summary_stat_fn is implemented, False otherwise
        """
        try:
            # Test with dummy data
            dummy_data = jnp.array([1.0, 2.0, 3.0])
            self.summary_stat_fn(dummy_data)
            return True
        except NotImplementedError:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_class': self.__class__.__name__,
            'model_module': self.__module__,
            'has_summary_stats': self.has_summary_stats(),
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