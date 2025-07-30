"""
Gaussian-Gaussian ABC models with known variance.

This module implements statistical models for Gaussian models where the variance
is known and we want to infer the mean parameter. These models are adapted
from the original ABCDataGenerator implementations.
"""

from jax import random
import jax.numpy as jnp
from typing import Dict, Optional, Any
import scipy.stats as scstats

from .base import StatisticalModel


class GaussGaussModel(StatisticalModel):
    """
    Gaussian-Gaussian statistical model with known standard deviation.

    Model: X | theta ~ N(theta, sigma^2)
    Prior: theta ~ N(mu0, sigma0^2)

    This is a classic ABC example where we have Gaussian observations
    with known variance sigma^2, and we want to infer the mean theta
    using a Gaussian prior.

    Args:
        mu0: Prior mean (default: 0.0)
        sigma0: Prior standard deviation (default: 1.0)
        sigma: Model standard deviation (known, default: 1.0)

    Example:
        import jax.numpy as jnp
        from jax import random
        from abcnre.simulation import ABCSimulator
        from abcnre.simulation.models import GaussGaussModel

        # Generate some observed data
        key = random.PRNGKey(42)
        true_theta = 2.5
        observed_data = true_theta + 0.2 * random.normal(key, shape=(100,))

        # Create model and simulator
        model = GaussGaussModel(mu0=0.0, sigma0=2.0, sigma=0.2)
        simulator = ABCSimulator(
            model=model,
            observed_data=observed_data,
            quantile_distance=0.01
        )

        # Generate ABC samples
        result = simulator.generate_samples(key, n_samples=1000)
        print(f"Posterior mean estimate: {jnp.mean(result.theta_samples):.3f}")
    """

    def __init__(
        self,
        mu0: float = 0.0,
        sigma0: float = 1.0,
        sigma: float = 1.0,
        n_obs: int = 100,
    ):
        """
        Initialize Gaussian-Gaussian model.

        Args:
            mu0: Prior mean for theta
            sigma0: Prior standard deviation for theta
            sigma: Model standard deviation (known noise level)
        """
        # Validate inputs
        if sigma0 <= 0 or sigma <= 0:
            raise ValueError("Standard deviations must be positive")

        # Store model parameters
        self.mu0 = float(mu0)  # Prior mean
        self.sigma0 = float(sigma0)  # Prior std
        self.sigma = float(sigma)  # Model std (known)
        self.sample_is_iid = True
        self.parameter_dim = 1  # Single parameter (theta)
        self.data_shape = (n_obs,)  # Shape of simulated data
        self.n_obs = n_obs  # Number of observations per sample
    
    def get_prior_sample(self, key: random.PRNGKey) -> jnp.ndarray:
        """
        Sample from Gaussian prior: theta ~ N(mu0, sigma0^2).

        Args:
            key: JAX random key

        Returns:
            Scalar parameter sample from prior
        """
        return self.mu0 + self.sigma0 * random.normal(key)

    def get_prior_samples(self, key: random.PRNGKey, n_samples: int) -> jnp.ndarray:
        """
        Draw multiple samples from the prior distribution efficiently.

        Args:
            key: JAX random key
            n_samples: Number of samples to draw

        Returns:
            Array of parameter samples of shape (n_samples,)
        """
        return self.mu0 + self.sigma0 * random.normal(key, shape=(n_samples,))

    def get_prior_dist(self):
        """
        Return a scipy.stats distribution object for the prior.

        Returns:
            Scipy normal distribution object representing the prior
        """
        return scstats.norm(loc=self.mu0, scale=self.sigma0)

    def prior_logpdf(self, theta: jnp.ndarray) -> float:
        """
        Log-density of the prior at theta.

        Args:
            theta: Parameter value (scalar or array)

        Returns:
            Log-probability under the prior
        """
        # Use scipy for numerical stability (works with floats)
        theta_val = float(theta) if jnp.isscalar(theta) else (theta.flatten())
        return scstats.norm.logpdf(theta_val, loc=self.mu0, scale=self.sigma0)

    def prior_pdf(self, theta: jnp.ndarray) -> float:
        """
        Density of the prior at theta.

        Args:
            theta: Parameter value (scalar or array)

        Returns:
            Probability density under the prior
        """
        theta_val = float(theta) if jnp.isscalar(theta) else (theta.flatten())
        return scstats.norm.pdf(theta_val, loc=self.mu0, scale=self.sigma0)

    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Sample from Gaussian likelihood: X | theta ~ N(theta, sigma^2).

        Generates synthetic data with each observation being Gaussian
        with mean theta and known standard deviation sigma.

        Args:
            key: JAX random key
            theta: Parameter value (scalar)
            n_obs: Number of observations to generate

        Returns:
            Simulated dataset of shape (n_obs,)
        """
        # JAX-compatible way to handle scalar/array theta
        if jnp.isscalar(theta):
            theta_val = theta
        else:
            # Use indexing instead of .item() for JAX compatibility
            theta_val = theta.flatten()[0]

        return theta_val + self.sigma * random.normal(key, shape=(self.n_obs, 1))

    def discrepancy_fn(self, data1: jnp.ndarray, data2: jnp.ndarray) -> float:
        """
        Euclidean distance between datasets.

        For Gaussian models, L2 distance is a natural choice.

        Args:
            data1: First dataset
            data2: Second dataset

        Returns:
            Euclidean distance between datasets
        """
        return jnp.linalg.norm(data1 - data2)

    def summary_stat_fn(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Summary statistics: sample mean only.

        For the Gaussian location model with known variance,
        the sample mean is the sufficient statistic.

        Args:
            data: Input dataset

        Returns:
            Array containing the mean of the data
        """
        return jnp.array([jnp.mean(data)])

    def transform_phi(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Identity transformation - theta is already our parameter of interest.

        Args:
            theta: Parameter value

        Returns:
            Same parameter value as 1D array
        """
        if jnp.isscalar(theta):
            return jnp.array([theta])
        else:
            # If theta is an array, return the first element as 1D array
            return jnp.array([theta.flatten()[0]])

    def get_posterior_stats(self, observed_data: jnp.ndarray) -> Dict[str, float]:
        """
        Get posterior statistics for comparison.

        For Gaussian-Gaussian conjugate model, the posterior is:
        theta | X ~ N(mu_post, sigma_post^2)

        where:
        - tau0 = 1/sigma0^2 (prior precision)
        - tau = n/sigma^2 (likelihood precision)
        - mu_post = (tau0*mu0 + tau*x_bar) / (tau0 + tau)
        - sigma_post^2 = 1 / (tau0 + tau)

        Args:
            observed_data: Observed dataset

        Returns:
            Dictionary with analytical posterior mean and std
        """
        n = len(observed_data)
        x_bar = jnp.mean(observed_data)

        # Precisions
        tau0 = 1.0 / (self.sigma0**2)  # Prior precision
        tau = n / (self.sigma**2)  # Likelihood precision

        # Posterior parameters
        mu_post = (tau0 * self.mu0 + tau * x_bar) / (tau0 + tau)
        sigma_post = 1.0 / jnp.sqrt(tau0 + tau)

        return {
            "posterior_mean": float(mu_post),
            "posterior_std": float(sigma_post),
            "sample_mean": float(x_bar),
            "sample_size": n,
            "prior_mean": self.mu0,
            "prior_std": self.sigma0,
            "model_std": self.sigma,
        }

    def get_posterior_distribution(self, observed_data: jnp.ndarray):
        """
        Get posterior distribution.

        Args:
            observed_data: Observed dataset

        Returns:
            Scipy normal distribution object
        """
        stats = self.get_posterior_stats(observed_data)
        return scstats.norm(loc=stats["posterior_mean"], scale=stats["posterior_std"])

    def get_model_args(self) -> Dict[str, Any]:
        """Get model parameters for serialization."""
        return {
            "model_type": "GaussGaussModel",
            "model_class": self.__class__.__name__,
            "model_args": {
                "mu0": self.mu0,
                "sigma0": self.sigma0,
                "sigma": self.sigma,
                "n_obs": self.n_obs,
            },
            "metadata": {
                "parameter_space_dim": 1,
                "analytical_posterior": True,
                "module": self.__class__.__module__,
            },
        }

    def update_model_params(
        self,
        mu0: Optional[float] = None,
        sigma0: Optional[float] = None,
        sigma: Optional[float] = None,
    ):
        """
        Update model parameters.

        Args:
            mu0: New prior mean (if provided)
            sigma0: New prior std (if provided)
            sigma: New model std (if provided)
        """
        if mu0 is not None:
            self.mu0 = float(mu0)
        if sigma0 is not None:
            if sigma0 <= 0:
                raise ValueError("sigma0 must be positive")
            self.sigma0 = float(sigma0)
        if sigma is not None:
            if sigma <= 0:
                raise ValueError("sigma must be positive")
            self.sigma = float(sigma)

    def validate_parameters(self, theta: jnp.ndarray) -> bool:
        """
        Validate parameter values.

        For Gaussian model, all real values are valid.

        Args:
            theta: Parameter values to validate

        Returns:
            Always True for Gaussian model
        """
        return True

    def __repr__(self) -> str:
        """String representation with model parameters."""
        return (
            f"GaussGaussModel("
            f"mu0={self.mu0}, sigma0={self.sigma0}, sigma={self.sigma})"
        )




# Export main classes
__all__ = [
    "GaussGaussModel",
]
