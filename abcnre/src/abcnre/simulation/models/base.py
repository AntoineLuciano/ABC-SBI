"""
Base class for statistical models in ABC simulation.

This module defines the interface that all statistical models must implement
to be used with the ABCSimulator.
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import random, vmap
from typing import Callable, Dict, Any, Optional, Tuple


class StatisticalModel(ABC):
    """
    Abstract base class for all statistical models used in ABC.
    
    This class defines the interface that statistical models must implement
    to be compatib7le with the ABCSimulator and RejectionSampler.
    
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
        
        return vmap(self.prior_sample)(keys)
    
    
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
    
    def simulate_datas(
        self, key: random.PRNGKey, theta: jnp.ndarray
    ) -> jnp.ndarray:
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
    
    def sample_theta_x(self, key: random.PRNGKey) -> jnp.ndarray:
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
    
    # def sample_phi_x(
    #     self, key: random.PRNGKey
    # ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #     """
    #     Sample phi and simulate data in one step.

    #     Args:
    #         key: JAX random key

    #     Returns:
    #         tuple: (phi, data)
    #         - phi: Transformed parameter (typically scalar)
    #         - data: Simulated dataset
    #     """
    #     key_theta, key_data = random.split(key)
    #     theta = self.get_prior_sample(key_theta)
    #     phi = self.transform_phi(theta)
    #     return phi, self.simulate_data(key_data, theta)

    # def sample_phi_x_multiple(
    #     self, key: random.PRNGKey, n_samples: int
    # ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #     """
    #     Sample multiple phi and simulate data in one step.

    #     Args:
    #         key: JAX random key
    #         n_samples: Number of samples to draw

    #     Returns:
    #         tuple: (phi_samples, data_samples)
    #         - phi_samples: Array of transformed parameters (shape (n_samples,))
    #         - data_samples: Array of simulated datasets (shape (n_samples, data_shape))
    #     """
  
    #     keys = random.split(key, n_samples)
    #     return vmap(lambda k: self.sample_phi_x(k))(keys)
    
    # # RG: Make a "summarized statistical model class"
    # # that just transforms theta and x.

    # # RG: Make a "rejection sampler model class"

    # @abstractmethod
    # def discrepancy_fn(self, data1: jnp.ndarray, data2: jnp.ndarray) -> float:
    #     """
    #     Compute distance/discrepancy between two datasets.
        
    #     Args:
    #         data1: First dataset/statistics
    #         data2: Second dataset/statistics
            
    #     Returns:
    #         Distance/discrepancy value (scalar)
    #     """
    #     # RG: Why is this a property of a statistical model?
    #     pass
        
    @abstractmethod
    def get_model_args(self) -> Dict[str, Any]:
        """
        Get model-specific arguments for serialization.
        
        Returns:
            Dictionary of arguments needed to recreate the model
        """
        pass
    
    # def summary_stat_fn(self, data: jnp.ndarray) -> jnp.ndarray:
    #     """
    #     Compute summary statistics from data (optional).
        
    #     Override this method if you want to use summary statistics
    #     instead of raw data for ABC comparison.
        
    #     Args:
    #         data: Input data
            
    #     Returns:
    #         Summary statistics
            
    #     Raises:
    #         NotImplementedError: If not implemented by subclass
    #     """
    #     raise NotImplementedError("summary_stat_fn not implemented")
    
    # def transform_phi(self, theta: jnp.ndarray) -> jnp.ndarray:
    #     """
    #     Transform theta to phi (target parameter of interest).
        
    #     Override this method if you want to focus inference on a specific
    #     transformation or marginal of the parameter vector.
        
    #     Args:
    #         theta: Full parameter vector
            
    #     Returns:
    #         Transformed parameter phi (typically scalar)
    #     """
    #     # Default: return first component as scalar
    #     if jnp.isscalar(theta):
    #         return theta
    #     else:
    #         return theta[0]
    
    # def has_summary_stats(self) -> bool:
    #     """
    #     Check if summary statistics function is implemented.
        
    #     Returns:
    #         True if summary_stat_fn is implemented, False otherwise
    #     """
    #     try:
    #         # Test with dummy data
    #         dummy_data = jnp.array([1.0, 2.0, 3.0])
    #         self.summary_stat_fn(dummy_data)
    #         return True
    #     except NotImplementedError:
    #         return False
    
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



class SummarizedStatisticalModel(StatisticalModel):
    """
    Draw from a statistical model but using a summary of the parameter.
    """
    def __init__(
            self,
            model: StatisticalModel,
            summary_fn: Callable[[jnp.ndarray], jnp.ndarray]):

        self.model = model
        self.summary_fn = summary_fn

    def get_prior_sample(self, key: random.PRNGKey) -> jnp.ndarray:
        return self.summary_fn(self.model.get_prior_sample(key))

    def get_prior_samples(self, key: random.PRNGKey, n_samples: int) -> jnp.ndarray:
        return super().get_prior_samples(key, n_samples)

    def sample_theta_x(self, key: random.PRNGKey) -> jnp.ndarray:
        theta, x = self.model.sample_theta_x(key)
        return self.summary_fn(theta), x

    def sample_theta_x_multiple(
            self, key: random.PRNGKey,  n_samples: int) -> jnp.ndarray:
        # keys = random.split(key, n_samples)
        # return vmap(lambda k: self.sample_theta_x(k))(keys)
        return super.sample_theta_multiple(key, n_samples)

    def simulate_data(self, key: random.PRNGKey, theta: jnp.ndarray) -> jnp.ndarray:
        self.model.simulate_data(key, theta)

    def simulate_datas(self, key: random.PRNGKey, theta: jnp.ndarray) -> jnp.ndarray:
        self.model.simulate_datas(key, theta)

    def get_model_args(self):
        # TODO: annotate the summary function, too
        return self.model.get_model_args()





# Export main class
__all__ = ["StatisticalModel", "SummarizedStatisticalModel"]



