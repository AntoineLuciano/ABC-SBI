"""
Gaussian-Gaussian ABC models with known variance.

This module implements statistical models for Gaussian models where the variance
is known and we want to infer the mean parameter. These models are adapted
from the original ABCDataGenerator implementations.
"""

from jax import random
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Any
import scipy.stats as scstats

from .base import StatisticalModel


class GaussGaussMultiDimModel(StatisticalModel):
    """
    Multidimensional Gaussian-Gaussian statistical model.

    Model: X | theta ~ N(theta, Sigma)
    Prior: theta ~ N(mu0, Sigma0)

    Extension to multidimensional case where theta is a vector.
    This is a conjugate model where both the prior and likelihood
    are multivariate Gaussian, allowing for analytical posterior
    computation.

    Args:
        mu0: Prior mean vector or scalar (default: zeros)
        sigma0: Prior covariance matrix or scalar (default: identity)
        sigma: Model covariance matrix or scalar (default: identity)
        dim: Dimension of parameter space (required if not inferred from mu0)
        n_obs: Number of observations per sample
        marginal_of_interest: Index of the marginal parameter of interest for phi transformation (default: 0)
    """

    def __init__(
        self,
        mu0: Optional[jnp.ndarray] = None,
        sigma0: Optional[jnp.ndarray] = None,
        sigma: Optional[jnp.ndarray] = None,
        dim: Optional[int] = None,
        n_obs: int = None,
        marginal_of_interest: int = 0,
    ):
        """Initialize multidimensional Gaussian-Gaussian model."""

        if n_obs is None:
            raise ValueError("You must specify n_obs.")

        # Determine dimension
        if mu0 is not None:
            if jnp.isscalar(mu0) or isinstance(mu0, (int, float)):
                # If mu0 is scalar and dim is provided, create constant vector
                if dim is not None:
                    mu0 = jnp.full(dim, float(mu0))
                else:
                    raise ValueError("Must provide dim when mu0 is scalar")
            else:
                mu0 = jnp.array(mu0)
                if mu0.ndim != 1:
                    raise ValueError("mu0 must be 1D array")
                dim = len(mu0)
        elif dim is not None:
            mu0 = jnp.zeros(dim)
        else:
            raise ValueError("Must provide either mu0 or dim")

        self.dim = dim

        # Validate marginal_of_interest
        if marginal_of_interest < 0 or marginal_of_interest >= dim:
            raise ValueError(f"marginal_of_interest must be between 0 and {dim-1}")
        self.marginal_of_interest = marginal_of_interest

        # Set default parameters
        if sigma0 is None:
            sigma0 = jnp.eye(dim)
        elif jnp.isscalar(sigma0) or isinstance(sigma0, (int, float)):
            sigma0 = float(sigma0) * jnp.eye(dim)
        else:
            sigma0 = jnp.array(sigma0)
            if sigma0.shape != (dim, dim):
                raise ValueError(f"sigma0 must be scalar or ({dim}, {dim}) matrix")

        if sigma is None:
            sigma = jnp.eye(dim)
        elif jnp.isscalar(sigma) or isinstance(sigma, (int, float)):
            sigma = float(sigma) * jnp.eye(dim)
        else:
            sigma = jnp.array(sigma)
            if sigma.shape != (dim, dim):
                raise ValueError(f"sigma must be scalar or ({dim}, {dim}) matrix")

        # Store parameters
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.sigma = sigma
        self.sample_is_iid = True
        self.parameter_dim = dim
        self.data_shape = (n_obs, dim)
        self.n_obs = n_obs

        # Precompute Cholesky decompositions for efficiency
        self.chol_sigma0 = jnp.linalg.cholesky(sigma0)
        self.chol_sigma = jnp.linalg.cholesky(sigma)

    def get_prior_sample(self, key: random.PRNGKey) -> jnp.ndarray:
        """Sample from multivariate Gaussian prior."""
        z = random.normal(key, shape=(self.dim,))
        return self.mu0 + self.chol_sigma0 @ z

    def get_prior_samples(self, key: random.PRNGKey, n_samples: int) -> jnp.ndarray:
        """
        Draw multiple samples from the prior distribution efficiently.

        Args:
            key: JAX random key
            n_samples: Number of samples to draw

        Returns:
            Array of parameter samples of shape (n_samples, dim)
        """
        z = random.normal(key, shape=(n_samples, self.dim))
        return self.mu0[None, :] + (self.chol_sigma0 @ z.T).T

    def get_prior_dist(self):
        """
        Return a scipy.stats distribution object for the prior.

        Returns:
            Scipy multivariate normal distribution object representing the prior
        """
        return scstats.multivariate_normal(mean=self.mu0, cov=self.sigma0)

    def prior_logpdf(self, theta: jnp.ndarray) -> float:
        """
        Log-density of the prior at theta.

        Args:
            theta: Parameter value (vector)

        Returns:
            Log-probability under the prior
        """
        return scstats.multivariate_normal.logpdf(theta, mean=self.mu0, cov=self.sigma0)

    def prior_pdf(self, theta: jnp.ndarray) -> float:
        """
        Density of the prior at theta.

        Args:
            theta: Parameter value (vector)

        Returns:
            Probability density under the prior
        """
        return scstats.multivariate_normal.pdf(theta, mean=self.mu0, cov=self.sigma0)

    def simulate_data(self, key: random.PRNGKey, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Sample from multivariate Gaussian likelihood: X | theta ~ N(theta, Sigma).

        Generates synthetic data with each observation being multivariate Gaussian
        with mean theta and known covariance matrix sigma.

        Args:
            key: JAX random key
            theta: Parameter vector

        Returns:
            Simulated dataset of shape (n_obs, dim)
        """
        z = random.normal(key, shape=(self.n_obs, self.dim))
        return theta + (self.chol_sigma @ z.T).T

    def prior_phi_logpdf(self, phi: jnp.ndarray) -> float:
        """
        Log-density of the prior at phi.

        Args:
            phi: Transformed parameter value (scalar or array)

        Returns:
            Log-probability under the prior for phi
        """
        return scstats.norm.logpdf(
            phi,
            loc=self.mu0[self.marginal_of_interest],
            scale=jnp.sqrt(
                self.sigma0[self.marginal_of_interest, self.marginal_of_interest]
            ),
        )

    def prior_phi_pdf(self, phi: jnp.ndarray) -> float:
        """
        Density of the prior at phi.

        Args:
            phi: Transformed parameter value (scalar or array)

        Returns:
            Probability density under the prior for phi
        """
        return scstats.norm.pdf(
            phi,
            loc=self.mu0[self.marginal_of_interest],
            scale=jnp.sqrt(
                self.sigma0[self.marginal_of_interest, self.marginal_of_interest]
            ),
        )

    def discrepancy_fn(self, data1: jnp.ndarray, data2: jnp.ndarray) -> float:
        """
        Compute distance between data or summary statistics.
        Uses Frobenius norm for matrices (2D) and L2 norm for vectors (1D).
        """
        diff = data1 - data2
        if diff.ndim == 1:
            # For 1D vectors (summary statistics), use L2 norm
            return jnp.linalg.norm(diff)
        else:
            # For 2D matrices (raw data), use Frobenius norm
            return jnp.linalg.norm(diff, ord="fro")

    def summary_stat_fn(self, data: jnp.ndarray) -> jnp.ndarray:
        """Summary statistics: sample mean for each dimension."""
        return jnp.mean(data, axis=0)

    def transform_phi(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Extract marginal of interest from theta vector.

        Args:
            theta: Parameter vector

        Returns:
            Scalar value of the marginal of interest
        """
        if jnp.isscalar(theta):
            return jnp.array([theta])
        else:
            return jnp.array([theta[self.marginal_of_interest]])

    def get_posterior_stats(self, observed_data: jnp.ndarray) -> Dict[str, Any]:
        """
        Get posterior statistics for comparison.

        For multivariate Gaussian-Gaussian conjugate model, the posterior is:
        theta | X ~ N(mu_post, Sigma_post)

        where:
        - Tau0 = Sigma0^{-1} (prior precision matrix)
        - Tau = n * Sigma^{-1} (likelihood precision matrix)
        - mu_post = (Tau0 + Tau)^{-1} * (Tau0 * mu0 + Tau * x_bar)
        - Sigma_post = (Tau0 + Tau)^{-1}

        Args:
            observed_data: Observed dataset of shape (n_obs, dim)

        Returns:
            Dictionary with analytical posterior mean and covariance
        """
        n, dim = observed_data.shape
        x_bar = jnp.mean(observed_data, axis=0)

        # Precision matrices
        tau0 = jnp.linalg.inv(self.sigma0)  # Prior precision
        tau = n * jnp.linalg.inv(self.sigma)  # Likelihood precision

        # Posterior parameters
        sigma_post = jnp.linalg.inv(tau0 + tau)
        mu_post = sigma_post @ (tau0 @ self.mu0 + tau @ x_bar)

        return {
            "posterior_mean": mu_post,
            "posterior_cov": sigma_post,
            "sample_mean": x_bar,
            "sample_size": n,
            "dimension": dim,
            "prior_mean": self.mu0,
            "prior_cov": self.sigma0,
            "model_cov": self.sigma,
        }

    def get_posterior_distribution(self, observed_data: jnp.ndarray):
        """
        Get posterior distribution.

        Args:
            observed_data: Observed dataset

        Returns:
            Scipy multivariate normal distribution object
        """
        stats = self.get_posterior_stats(observed_data)
        return scstats.multivariate_normal(
            mean=stats["posterior_mean"], cov=stats["posterior_cov"]
        )

    def get_analytical_posterior_stats(
        self, observed_data: jnp.ndarray
    ) -> Dict[str, Any]:
        """
        Get analytical posterior statistics for this model.

        This is an alias for get_posterior_stats() to clarify that this model
        supports analytical posteriors.

        Args:
            observed_data: Observed dataset

        Returns:
            Dictionary with analytical posterior statistics
        """
        return self.get_posterior_stats(observed_data)

    def get_pymc_posterior(self, observed_data: jnp.ndarray, n_samples: int = 1000):
        """
        Get PyMC posterior samples for the analytical posterior.

        This allows integration with PyMC for advanced sampling and diagnostics.

        Args:
            observed_data: Observed dataset
            n_samples: Number of samples to draw (default: 1000)

        Returns:
            PyMC trace with posterior samples

        Raises:
            ImportError: If PyMC is not installed
        """
        try:
            import pymc as pm
            import pytensor.tensor as pt
        except ImportError:
            raise ImportError(
                "PyMC is required for get_pymc_posterior(). "
                "Please install it with: pip install pymc"
            )

        # Get analytical posterior parameters
        stats = self.get_posterior_stats(observed_data)
        mu_post = np.array(stats["posterior_mean"])  # Convert JAX array to numpy
        sigma_post = np.array(stats["posterior_cov"])  # Convert JAX array to numpy

        # Create PyMC model and sample
        with pm.Model() as model:
            # Define the multivariate posterior distribution for theta
            theta = pm.MvNormal("theta", mu=mu_post, cov=sigma_post, shape=self.dim)

            # Store analytical parameters as model attributes for reference
            model.analytical_mu = mu_post
            model.analytical_sigma = sigma_post
            model.observed_data = observed_data
            model.prior_mu = self.mu0
            model.prior_sigma = self.sigma0
            model.likelihood_sigma = self.sigma
            model.dimension = self.dim
            model.marginal_of_interest = self.marginal_of_interest

            # Sample from the posterior
            trace = pm.sample(n_samples, tune=500, chains=2, return_inferencedata=True)

        return trace

    def has_analytical_posterior(self) -> bool:
        """
        Check if this model supports analytical posterior computation.

        Returns:
            True for multivariate Gaussian-Gaussian conjugate model
        """
        return True

    def get_posterior_phi_distribution(self, observed_data: jnp.ndarray):
        """
        Get posterior distribution for the marginal of interest.

        Args:
            observed_data: Observed dataset
        Returns:
            Scipy multivariate normal distribution object for the marginal of interest
        """
        stats = self.get_posterior_stats(observed_data)
        # Extract marginal of interest from posterior mean and covariance
        mu_post_marginal = stats["posterior_mean"][self.marginal_of_interest]
        cov_post_marginal = stats["posterior_cov"][
            self.marginal_of_interest, self.marginal_of_interest
        ]

        return scstats.norm(loc=mu_post_marginal, scale=jnp.sqrt(cov_post_marginal))

    def get_model_args(self) -> Dict[str, Any]:
        """Get model parameters for serialization."""
        return {
            "model_type": "GaussGaussMultiDimModel",
            "model_class": self.__class__.__name__,
            "model_args": {
                "mu0": self.mu0.tolist(),
                "sigma0": self.sigma0.tolist(),
                "sigma": self.sigma.tolist(),
                "dim": self.dim,
                "n_obs": self.n_obs,
                "marginal_of_interest": self.marginal_of_interest,
            },
            "metadata": {
                "parameter_space_dim": self.dim,
                "analytical_posterior": True,
                "module": self.__class__.__module__,
            },
        }

    def update_model_params(
        self,
        mu0: Optional[jnp.ndarray] = None,
        sigma0: Optional[jnp.ndarray] = None,
        sigma: Optional[jnp.ndarray] = None,
        marginal_of_interest: Optional[int] = None,
    ):
        """
        Update model parameters.

        Args:
            mu0: New prior mean vector or scalar (if provided)
            sigma0: New prior covariance matrix or scalar (if provided)
            sigma: New model covariance matrix or scalar (if provided)
            marginal_of_interest: New marginal of interest index (if provided)
        """
        if mu0 is not None:
            if jnp.isscalar(mu0) or isinstance(mu0, (int, float)):
                mu0 = jnp.full(self.dim, float(mu0))
            else:
                mu0 = jnp.array(mu0)
                if mu0.shape != (self.dim,):
                    raise ValueError(f"mu0 must have shape ({self.dim},)")
            self.mu0 = mu0

        if marginal_of_interest is not None:
            if marginal_of_interest < 0 or marginal_of_interest >= self.dim:
                raise ValueError(
                    f"marginal_of_interest must be between 0 and {self.dim-1}"
                )
            self.marginal_of_interest = marginal_of_interest

        if sigma0 is not None:
            if jnp.isscalar(sigma0) or isinstance(sigma0, (int, float)):
                sigma0 = float(sigma0) * jnp.eye(self.dim)
            else:
                sigma0 = jnp.array(sigma0)
                if sigma0.shape != (self.dim, self.dim):
                    raise ValueError(
                        f"sigma0 must be scalar or ({self.dim}, {self.dim}) matrix"
                    )
            # Check positive definiteness
            try:
                jnp.linalg.cholesky(sigma0)
            except Exception:
                raise ValueError("sigma0 must be positive definite")
            self.sigma0 = sigma0
            self.chol_sigma0 = jnp.linalg.cholesky(sigma0)

        if sigma is not None:
            if jnp.isscalar(sigma) or isinstance(sigma, (int, float)):
                sigma = float(sigma) * jnp.eye(self.dim)
            else:
                sigma = jnp.array(sigma)
                if sigma.shape != (self.dim, self.dim):
                    raise ValueError(
                        f"sigma must be scalar or ({self.dim}, {self.dim}) matrix"
                    )
            # Check positive definiteness
            try:
                jnp.linalg.cholesky(sigma)
            except Exception:
                raise ValueError("sigma must be positive definite")
            self.sigma = sigma
            self.chol_sigma = jnp.linalg.cholesky(sigma)

    def validate_parameters(self, theta: jnp.ndarray) -> bool:
        """
        Validate parameter values.

        Args:
            theta: Parameter values to validate

        Returns:
            True if theta has correct dimension and contains finite values
        """
        if theta.shape != (self.dim,):
            return False
        return jnp.all(jnp.isfinite(theta))

    def __repr__(self) -> str:
        """String representation with model parameters."""
        return (
            f"GaussGaussMultiDimModel("
            f"dim={self.dim}, n_obs={self.n_obs}, marginal_of_interest={self.marginal_of_interest})"
        )


# Export main classes
__all__ = [
    "GaussGaussMultiDimModel",
]
