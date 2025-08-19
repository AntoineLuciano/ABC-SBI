"""
G-and-K distribution ABC models.

The G-and-K distribution is a flexible four-parameter family defined by its
quantile function. This module implements the statistical model following
Fearnhead & Prangle (2011).

Example usage:
    import jax.numpy as jnp
    from jax import random
    from abcnre.simulation import ABCSimulator
    from abcnre.simulation.models import GAndKModel, create_synthetic_g_and_k_data

    # Generate observed data from known parameters
    key = random.PRNGKey(42)
    observed_data, true_params = create_synthetic_g_and_k_data(key)

    # Create model and simulator
    model = GAndKModel(prior_bounds=(0.0, 10.0))
    simulator = ABCSimulator(
        model=model,
        observed_data=observed_data,
        quantile_distance=0.01
    )

    # Run ABC inference
    result = simulator.generate_samples(key, n_samples=1000)
"""

import jax
from jax import random
import jax.numpy as jnp
from jax.scipy import stats as jstats
from typing import Dict, Optional, Tuple, Any
import scipy.stats as scstats

from .base import StatisticalModel

from jax import random
import jax.numpy as jnp

def generate_g_and_k_samples(
    key: random.PRNGKey,
    n_samples: int,
    A: float = 0.0,
    B: float = 1.0,
    g: float = 0.0,
    k: float = 0.0,
) -> jnp.ndarray:
    """
    Generate samples from G-and-K distribution.

    Args:
        key: JAX random key
        n_samples: Number of samples to generate
        A: Location parameter (real)
        B: Scale parameter (B > 0)
        g: Skewness parameter (real)
        k: Kurtosis parameter (k > -1/2)

    Returns:
        Array of G-and-K samples
    """


    z = random.normal(key, shape=(n_samples, 1))
    c = 0.8

    samples = A + B * (1 + c * jnp.tanh(g * z / 2)) * (1 + z**2) ** k * z

    # Check for numerical issues (optional)
    samples = jnp.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

    return samples



class GAndKModel(StatisticalModel):
    """
    G-and-K distribution statistical model following Fearnhead & Prangle (2011).

    The G-and-K distribution is defined by four parameters:
    - A: location parameter (real)
    - B: scale parameter (B > 0)
    - g: skewness parameter (real)
    - k: kurtosis parameter (k > -1/2)

    This implementation uses a uniform prior on [0,10]^4 as in the paper,
    with the constraint that B > 0 and k > -1/2.

    Args:
        prior_bounds: Either (low, high) for uniform prior on [low,high]^4,
                     or custom bounds dict {'A': (min, max), ...}
        n_obs: Number of observations per sample (default: 100)

    Example:
        # Following Section 4.2 of Fearnhead & Prangle (2011)
        model = GAndKModel(prior_bounds=(0.0, 10.0), n_obs=100)
    """

    def __init__(
        self,
        prior_bounds: tuple = (0.0, 10.0),
        n_obs: int = 100,
        marginal_of_interest: int = -1,
        parameter_of_interest: str = None,
        dim: int = 1,
    ):
        """
        Initialize G-and-K model.

        Args:
            prior_bounds: Tuple (low, high) for uniform prior on [low,high]^4
            n_obs: Number of observations per sample
        """
        # Set up parameter bounds - uniform on [0,10]^4 as in the paper
        if (isinstance(prior_bounds, tuple) or isinstance(prior_bounds, list)) and len(
            prior_bounds
        ) == 2:
            low, high = prior_bounds
            self.param_bounds = {
                "A": (low, high),
                "B": (max(low, 0.001), high),  # B must be positive
                "g": (low, high),
                "k": (max(low, -0.49), high),  # k > -1/2
            }
        elif isinstance(prior_bounds, dict):
            self.param_bounds = {
                "A": (
                    float(prior_bounds.get("A", (0.0, 10.0))[0]),
                    float(prior_bounds.get("A", (0.0, 10.0))[1]),
                ),
                "B": (
                    float(prior_bounds.get("B", (0.0, 10.0))[0]),
                    float(prior_bounds.get("B", (0.0, 10.0))[1]),
                ),
                "g": (
                    float(prior_bounds.get("g", (0.0, 10.0))[0]),
                    float(prior_bounds.get("g", (0.0, 10.0))[1]),
                ),
                "k": (
                    float(prior_bounds.get("k", (0.0, 10.0))[0]),
                    float(prior_bounds.get("k", (0.0, 10.0))[1]),
                ),
            }
        else:
            raise ValueError("prior_bounds must be tuple or list (low, high)")

        # Store model parameters following the pattern of other models
        self.sample_is_iid = True
        self.parameter_dim = 4  # Four parameters [A, B, g, k]
        self.dim = dim
        self.data_shape = (
            n_obs,
            self.dim,
        )  # Shape of simulated data (1D like other models)
        self.n_obs = n_obs  # Number of observations per sample
        self.marginal_of_interest = marginal_of_interest
        if self.marginal_of_interest == -1:
            self.parameter_of_interest = "all"
            self.phi_dim = 4
        elif self.marginal_of_interest == 0:
            self.parameter_of_interest = "A"
            self.phi_dim = 1
        elif self.marginal_of_interest == 1:
            self.parameter_of_interest = "B"
            self.phi_dim = 1
        elif self.marginal_of_interest == 2:
            self.parameter_of_interest = "g"
            self.phi_dim = 1
        elif self.marginal_of_interest == 3:
            self.parameter_of_interest = "k"
            self.phi_dim = 1
        else:
            raise ValueError(
                "marginal_of_interest must be one of -1, 0, 1, 2, 3"
            )

        # Pre-compute bounds arrays for efficient sampling
        self.param_names = ["A", "B", "g", "k"]
        self.lower_bounds = jnp.array(
            [self.param_bounds[name][0] for name in self.param_names]
        )
        self.upper_bounds = jnp.array(
            [self.param_bounds[name][1] for name in self.param_names]
        )
        self.param_ranges = self.upper_bounds - self.lower_bounds
        self.name = f"G_and_K_{self.dim}D_marginal_{self.parameter_of_interest}"

    def sample_theta(self, key: random.PRNGKey) -> jnp.ndarray:
        """
        Sample from uniform prior over parameter bounds.

        Args:
            key: JAX random key

        Returns:
            Single parameter vector [A, B, g, k]
        """
        u = random.uniform(key, shape=(4,))
        return self.lower_bounds + self.param_ranges * u

    def sample_thetas(self, key: random.PRNGKey, n_samples: int) -> jnp.ndarray:
        """
        Draw multiple samples from the prior distribution efficiently.

        Args:
            key: JAX random key
            n_samples: Number of prior samples to draw

        Returns:
            Array of shape (n_samples, 4) with prior parameter samples
        """
        u = random.uniform(key, shape=(n_samples, 4))
        return self.lower_bounds + self.param_ranges * u

    def get_prior_log_density(self, theta: jnp.ndarray) -> float:
        """
        Compute log density of uniform prior.

        Args:
            theta: Parameter vector [A, B, g, k]

        Returns:
            Log density of uniform prior (constant within bounds, -inf outside)
        """
        # Check if parameters are within bounds
        within_bounds = jnp.all(
            (theta >= self.lower_bounds) & (theta <= self.upper_bounds)
        )

        # For uniform prior: log density = -log(volume) if within bounds, -inf otherwise
        log_volume = jnp.sum(jnp.log(self.param_ranges))
        return jnp.where(within_bounds, -log_volume, -jnp.inf)

    def prior_sample(self, key: random.PRNGKey) -> jnp.ndarray:
        """
        Sample from uniform prior over parameter bounds (backward compatibility).

        Args:
            key: JAX random key

        Returns:
            Parameter vector [A, B, g, k]
        """
        return self.sample_theta(key)

    def sample_x(self, key: random.PRNGKey, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Simulate data from G-and-K distribution.

        Args:
            key: JAX random key
            theta: Parameter vector [A, B, g, k] (single parameter set)

        Returns:
            Simulated dataset of size n_obs
        """
        # Extract parameters from theta vector - handle both scalar and array cases
        A = theta[0]
        B = theta[1]
        g = theta[2]
        k = theta[3]
        return generate_g_and_k_samples(key, self.n_obs, A, B, g, k)

    def discrepancy_fn(self, data1: jnp.ndarray, data2: jnp.ndarray) -> float:
        """
        Compute distance between data or summary statistics.
        Uses Frobenius norm for matrices (2D) and L2 norm for vectors (1D).
        """
        
        data1, data2 = jnp.asarray(data1).flatten(), jnp.asarray(data2).flatten()
        diff = data1 - data2
        if diff.ndim == 1:
            # For 1D vectors (summary statistics), use L2 norm
            return jnp.linalg.norm(diff)
        else:
            return jnp.linalg.norm(diff, ord="fro")
        

 

    def predefined_summary_stat_fn(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Predefined summary statistics function.

        For G-and-K models, we use quantiles as predefined summary statistics.
        This provides a good balance between information preservation and
        dimensionality reduction.

        Args:
            data: Input dataset

        Returns:
            Array of quantiles at levels [0.1, 0.25, 0.5, 0.75, 0.9]
        """
        quantile_levels = jnp.array([.1,.2,.3,.4,.5,.6,.7,.8,.9])
        if data.ndim == 1:
            return jnp.quantile(data, quantile_levels)
        elif data.ndim == 2:
            # For 2D data, compute quantiles along the first axis
            return jnp.quantile(data, quantile_levels, axis=0)
        elif data.ndim == 3:
            # For 3D data, compute quantiles along the second axis
            return jnp.quantile(data, quantile_levels, axis=1)
        else:
            raise ValueError("Data must be 1D, 2D, or 3D array")

    def transform_phi(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Default transformation: return A parameter (location).

        Override this method to focus on different parameters.

        Args:
            theta: Parameter vector [A, B, g, k]

        Returns:
            Location parameter A as scalar
        """
        if self.marginal_of_interest == -1:
            return jnp.array(theta)
        else:
            return jnp.array([theta[self.marginal_of_interest]])


    def get_parameter_names(self) -> list:
        """Get parameter names in order."""
        return self.param_names.copy()

    def get_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds."""
        return self.param_bounds.copy()

    def get_evenly_spaced_order_stats(self, data: jnp.ndarray, m: int) -> jnp.ndarray:
        """
        Get m evenly spaced order statistics from data.

        This follows the approach in Section 4.2 of the paper for reducing
        the dimensionality of summary statistics.

        Args:
            data: Input data
            m: Number of order statistics to select

        Returns:
            Array of m evenly spaced order statistics
        """
        sorted_data = jnp.sort(data)
        n = len(sorted_data)

        # Select evenly spaced indices
        indices = jnp.linspace(0, n - 1, m).astype(int)
        return sorted_data[indices]

    def get_model_args(self) -> Dict[str, Any]:
        """Get model parameters for serialization."""
        return {
            "model_type": "GAndKModel",
            "model_class": self.__class__.__name__,
            "model_args": {
                "prior_bounds": {
                    "A": [float(self.lower_bounds[0]), float(self.upper_bounds[0])],
                    "B": [float(self.lower_bounds[1]), float(self.upper_bounds[1])],
                    "g": [float(self.lower_bounds[2]), float(self.upper_bounds[2])],
                    "k": [float(self.lower_bounds[3]), float(self.upper_bounds[3])],
                },
                "dim": int(self.dim),
                "n_obs": int(self.n_obs),
                "marginal_of_interest": int(self.marginal_of_interest),
                "parameter_of_interest": (self.parameter_of_interest),
                # Note: marginal_of_interest is computed automatically from parameter_of_interest
            },
        }

    def prior_phi_logpdf(self, phi: jnp.ndarray) -> float:
        """
        Compute log PDF of prior distribution for phi.

        Args:
            phi: Parameter vector [A, B, g, k]
        Returns:

            Log PDF of prior distribution
        """
        if self.marginal_of_interest == -1:
            return self.get_prior_log_density(phi)

        return jnp.where(
            jnp.logical_and(
                phi >= self.lower_bounds[self.marginal_of_interest],
                phi <= self.upper_bounds[self.marginal_of_interest],
            ),
            -jnp.log(self.param_ranges[self.marginal_of_interest]),
            -jnp.inf,
        )

    def prior_phi_pdf(self, phi: jnp.ndarray) -> float:
        """
        Compute PDF of prior distribution for phi.

        Args:
            phi: Parameter vector [A, B, g, k]
        Returns:
            PDF of prior distribution
        """
        if self.marginal_of_interest == -1:
            return jnp.exp(self.get_prior_log_density(phi))

        return jnp.where(
            jnp.logical_and(
                phi >= self.lower_bounds[self.marginal_of_interest],
                phi <= self.upper_bounds[self.marginal_of_interest],
            ),
            1.0 / self.param_ranges[self.marginal_of_interest],
            0.0,
        )
        
    def get_posterior_logpdf(self, observed_data: jnp.ndarray) -> jnp.ndarray:
        import jax.lax as lax
        # Constantes
        LOG2PI = jnp.log(2.0 * jnp.pi)

        # Fonctions auxiliaires G&K optimisées
        @jax.jit
        def _A(z, g):
            return 1.0 + 0.8 * jnp.tanh(0.5 * g * z)

        @jax.jit
        def _dA(z, g):
            u = 0.5 * g * z
            return 0.4 * g * (1.0 / jnp.cosh(u))**2

        @jax.jit
        def _H(z, k):
            return (1.0 + z**2)**k

        @jax.jit
        def _dH(z, k):
            return 2.0 * k * z * (1.0 + z**2)**(k - 1.0)

        @jax.jit
        def Q_of_z(z, theta):
            a, b, g, k = theta
            return a + b * _A(z, g) * _H(z, k) * z

        @jax.jit
        def dQdz(z, theta):
            a, b, g, k = theta
            A, H = _A(z, g), _H(z, k)
            dA, dH = _dA(z, g), _dH(z, k)
            return b * (A * H + A * z * dH + H * z * dA)

        @jax.jit
        def solve_z_for_x(x, theta, z_min=-8.0, z_max=8.0, newton_steps=25, bisect_steps=35):
            x_scalar = jnp.squeeze(x)
            
            # Bornes de l'intervalle
            q_lo = Q_of_z(z_min, theta)
            q_hi = Q_of_z(z_max, theta)
            
            # Cas limites
            def _return_low(_): return z_min
            def _return_high(_): return z_max
            
            def _solve(_):
                a, b, g, k = theta
                # Initialisation intelligente
                z0 = jnp.clip((x_scalar - a) / jnp.maximum(jnp.abs(b), 1e-8), z_min, z_max)
                z_lo, z_hi, z = z_min, z_max, z0
                
                # Phase Newton-Raphson avec fallback bissection
                def newton_body(carry, _):
                    z_lo, z_hi, z = carry
                    f = Q_of_z(z, theta) - x_scalar
                    df = dQdz(z, theta)
                    
                    # Newton step protégé
                    z_newton = z - f / jnp.where(jnp.abs(df) < 1e-12, jnp.sign(df) * 1e-12, df)
                    z_bis = 0.5 * (z_lo + z_hi)
                    
                    # Choix du meilleur step
                    in_bracket = (z_newton >= z_lo) & (z_newton <= z_hi)
                    f_newton = Q_of_z(z_newton, theta) - x_scalar
                    f_bis = Q_of_z(z_bis, theta) - x_scalar
                    
                    use_newton = in_bracket & (jnp.abs(f_newton) <= jnp.abs(f_bis))
                    z_next = jnp.where(use_newton, z_newton, z_bis)
                    f_next = jnp.where(use_newton, f_newton, f_bis)
                    
                    # Update brackets
                    z_lo = jnp.where(f_next < 0.0, z_next, z_lo)
                    z_hi = jnp.where(f_next >= 0.0, z_next, z_hi)
                    
                    return (z_lo, z_hi, z_next), None
                
                (z_lo, z_hi, z), _ = lax.scan(newton_body, (z_lo, z_hi, z), xs=None, length=newton_steps)
                
                # Raffinement final par bissection
                def bisect_body(carry, _):
                    z_lo, z_hi = carry
                    z_mid = 0.5 * (z_lo + z_hi)
                    f_mid = Q_of_z(z_mid, theta) - x_scalar
                    z_lo = jnp.where(f_mid < 0.0, z_mid, z_lo)
                    z_hi = jnp.where(f_mid >= 0.0, z_mid, z_hi)
                    return (z_lo, z_hi), None
                
                (z_lo, z_hi), _ = lax.scan(bisect_body, (z_lo, z_hi), xs=None, length=bisect_steps)
                return 0.5 * (z_lo + z_hi)
            
            return lax.cond(x_scalar <= q_lo, _return_low,
                            lambda _: lax.cond(x_scalar >= q_hi, _return_high, _solve, operand=None),
                            operand=None)

        @jax.jit
        def logpdf_gandk(x, theta, eps=1e-12):
            z_star = solve_z_for_x(x, theta)
            
            # Log-PDF = log φ(z) - log |Q'(z)|
            log_phi = -0.5 * z_star**2 - 0.5 * LOG2PI
            qprime = jnp.maximum(jnp.abs(dQdz(z_star, theta)), eps)
            
            return log_phi - jnp.log(qprime)

        logpdf_gandk_vectorized = jax.jit(jax.vmap(logpdf_gandk, in_axes=(0, None)))

        return lambda theta: self.get_prior_log_density(theta) + jnp.sum(logpdf_gandk_vectorized(observed_data, theta))






# Export main components
__all__ = [
    "GAndKModel",
    "generate_g_and_k_samples",
]
