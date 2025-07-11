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


@jax.jit
def generate_g_and_k_samples(
    key: random.PRNGKey, 
    n_samples: int, 
    A: float = 0.0, 
    B: float = 1.0, 
    g: float = 0.0, 
    k: float = 0.0
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
    u = random.uniform(key, shape=(n_samples,))
    z = jstats.norm.ppf(u)
    c = 0.8
    
    # Handle skewness term with numerical stability
    skew_term = jnp.where(
        jnp.abs(g) < 1e-10,
        c * z,
        c * (1 - jnp.exp(-g * z)) / (1 + jnp.exp(-g * z))
    )
    
    samples = A + B * (1 + skew_term) * (1 + z**2)**k * z
    
    # Check for numerical issues
    samples = jnp.where(jnp.isnan(samples), 0.0, samples)
    samples = jnp.where(jnp.isinf(samples), 0.0, samples)
    
    return samples


def compute_g_and_k_quantiles(
    A: float, 
    B: float, 
    g: float, 
    k: float, 
    quantiles: jnp.ndarray = None
) -> jnp.ndarray:
    """
    Compute specific quantiles of G-and-K distribution.
    
    Args:
        A, B, g, k: G-and-K parameters
        quantiles: Array of quantile levels (default: octiles)
        
    Returns:
        Array of quantile values
    """
    if quantiles is None:
        # Use octiles as in Fearnhead & Prangle (2011)
        quantiles = jnp.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
    
    # Convert to standard normal quantiles
    z = jstats.norm.ppf(quantiles)
    
    # G-and-K transformation
    c = 0.8
    skew_term = jnp.where(
        jnp.abs(g) < 1e-10,
        c * z,
        c * (1 - jnp.exp(-g * z)) / (1 + jnp.exp(-g * z))
    )
    
    return A + B * (1 + skew_term) * (1 + z**2)**k * z


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
        
    Example:
        # Following Section 4.2 of Fearnhead & Prangle (2011)
        model = GAndKModel(prior_bounds=(0.0, 10.0))  # Uniform prior on [0,10]^4
    """
    
    def __init__(
        self,
        prior_bounds: tuple = (0.0, 10.0)
    ):
        """
        Initialize G-and-K model.
        
        Args:
            prior_bounds: Tuple (low, high) for uniform prior on [low,high]^4
        """
        # Set up parameter bounds - uniform on [0,10]^4 as in the paper
        if isinstance(prior_bounds, tuple) and len(prior_bounds) == 2:
            low, high = prior_bounds
            self.param_bounds = {
                'A': (low, high),
                'B': (max(low, 0.001), high),  # B must be positive
                'g': (low, high),
                'k': (max(low, -0.49), high)   # k > -1/2
            }
        else:
            raise ValueError("prior_bounds must be tuple (low, high)")
        
        # Pre-compute bounds arrays for efficient sampling
        self.param_names = ['A', 'B', 'g', 'k']
        self.lower_bounds = jnp.array([self.param_bounds[name][0] for name in self.param_names])
        self.upper_bounds = jnp.array([self.param_bounds[name][1] for name in self.param_names])
        self.param_ranges = self.upper_bounds - self.lower_bounds
    
    def prior_sample(self, key: random.PRNGKey) -> jnp.ndarray:
        """
        Sample from uniform prior over parameter bounds.
        
        Args:
            key: JAX random key
            
        Returns:
            Parameter vector [A, B, g, k]
        """
        u = random.uniform(key, shape=(4,))
        return self.lower_bounds + self.param_ranges * u
    
    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray, n_obs: int) -> jnp.ndarray:
        """
        Simulate data from G-and-K distribution.
        
        Args:
            key: JAX random key
            theta: Parameter vector [A, B, g, k]
            n_obs: Number of observations to generate
            
        Returns:
            Simulated dataset of size n_obs
        """
        A, B, g, k = theta[0], theta[1], theta[2], theta[3]
        return generate_g_and_k_samples(key, n_obs, A, B, g, k)
    
    def discrepancy_fn(self, data1: jnp.ndarray, data2: jnp.ndarray) -> float:
        """
        L2 distance between datasets with numerical stability.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            L2 distance between datasets
        """
        d = jnp.sum((data1 - data2) ** 2)
        return jnp.where(jnp.isnan(d), 1e6, d)
    
    def summary_stat_fn(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Order statistics as summary statistics.
        
        Following Fearnhead & Prangle (2011), we use order statistics
        as summary statistics for the G-and-K model.
        
        Args:
            data: Input dataset
            
        Returns:
            Array of order statistics (sorted data)
        """
        # Sort the data to get order statistics
        sorted_data = jnp.sort(data)
        return sorted_data  # Return all order statistics
    
    def transform_phi(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Default transformation: return A parameter (location).
        
        Override this method to focus on different parameters.
        
        Args:
            theta: Parameter vector [A, B, g, k]
            
        Returns:
            Location parameter A as scalar
        """
        return theta[0]  # Return A parameter
    
    def get_parameter_names(self) -> list:
        """Get parameter names in order."""
        return self.param_names.copy()
    
    def get_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds."""
        return self.param_bounds.copy()
    
    def compute_theoretical_quantiles(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Compute theoretical quantiles for given parameters.
        
        Args:
            theta: Parameter vector [A, B, g, k]
            
        Returns:
            Theoretical quantile values
        """
        A, B, g, k = theta[0], theta[1], theta[2], theta[3]
        return compute_g_and_k_quantiles(A, B, g, k)
    
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
        indices = jnp.linspace(0, n-1, m).astype(int)
        return sorted_data[indices]
    
    def get_model_args(self) -> Dict[str, Any]:
        """Get model parameters for serialization."""
        return {
            'prior_bounds': (float(self.lower_bounds[0]), float(self.upper_bounds[0])) 
            if jnp.allclose(self.lower_bounds, self.lower_bounds[0]) and 
               jnp.allclose(self.upper_bounds, self.upper_bounds[0])
            else {'A': (float(self.lower_bounds[0]), float(self.upper_bounds[0])),
                  'B': (float(self.lower_bounds[1]), float(self.upper_bounds[1])),
                  'g': (float(self.lower_bounds[2]), float(self.upper_bounds[2])),
                  'k': (float(self.lower_bounds[3]), float(self.upper_bounds[3]))}
        }
    
    def validate_parameters(self, theta: jnp.ndarray) -> bool:
        """
        Validate G-and-K parameter values.
        
        Args:
            theta: Parameter values [A, B, g, k] to validate
            
        Returns:
            True if parameters satisfy constraints (B > 0, k > -0.5)
        """
        if len(theta) != 4:
            return False
        
        A, B, g, k = theta[0], theta[1], theta[2], theta[3]
        
        # Check constraints
        if B <= 0:  # B must be positive
            return False
        if k <= -0.5:  # k must be > -1/2
            return False
        
        # Check bounds
        for i, param in enumerate(theta):
            if param < self.lower_bounds[i] or param > self.upper_bounds[i]:
                return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation with parameter bounds."""
        bounds_str = ", ".join([f"{name}={self.param_bounds[name]}" 
                               for name in self.param_names])
        return f"GAndKModel(bounds=[{bounds_str}])"


def create_synthetic_g_and_k_data(
    key: random.PRNGKey,
    true_params: jnp.ndarray = None,
    n_samples: int = int(10**4),
    noise_level: float = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create synthetic G-and-K data for testing following Section 4.2.
    
    Args:
        key: JAX random key
        true_params: True parameter values [A, B, g, k] (default: [3, 1, 2, 0.5])
        n_samples: Number of samples to generate (default: 10^4 as in paper)
        noise_level: Additional Gaussian noise std (default: 0.0)
        
    Returns:
        Tuple of (synthetic_data, true_params_array)
    """
    if true_params is None:
        # Default parameters from Fearnhead & Prangle (2011) Section 4.2
        true_params = jnp.array([3.0, 1.0, 2.0, 0.5])  # A, B, g, k
    
    if len(true_params) != 4:
        raise ValueError("true_params must contain exactly 4 values [A, B, g, k]")
    
    A, B, g, k = true_params[0], true_params[1], true_params[2], true_params[3]
    
    # Generate clean G-and-K data
    key, data_key = random.split(key)
    data = generate_g_and_k_samples(data_key, n_samples, A, B, g, k)
    
    # Add noise if specified
    if noise_level > 0:
        key, noise_key = random.split(key)
        noise = noise_level * random.normal(noise_key, shape=data.shape)
        data = data + noise
    
    return data, true_params


def get_fearnhead_prangle_setup() -> Dict[str, Any]:
    """
    Get the exact experimental setup from Fearnhead & Prangle (2011) Section 4.2.
    
    Returns:
        Dictionary with experimental parameters
    """
    return {
        'true_params': jnp.array([3.0, 1.0, 2.0, 0.5]),  # A, B, g, k
        'n_samples': int(100),  # Number of independent draws
        'prior_bounds': (0.0, 10.0),  # Uniform prior on [0,10]^4
        'summary_stats': 'order_statistics',  # Full set of order statistics
        'paper_reference': 'Fearnhead & Prangle (2011) Section 4.2'
    }


def create_order_statistics_subset(
    data: jnp.ndarray, 
    m: int = 100, 
    include_powers: bool = True,
    max_power: int = 4
) -> jnp.ndarray:
    """
    Create subset of order statistics with optional powers.
    
    Following the paper's approach of using m evenly spaced order statistics
    plus powers up to the fourth power for better summary statistics.
    
    Args:
        data: Input data
        m: Number of evenly spaced order statistics
        include_powers: Whether to include powers of order statistics
        max_power: Maximum power to include (up to 4)
        
    Returns:
        Array of summary statistics
    """
    sorted_data = jnp.sort(data)
    n = len(sorted_data)
    
    # Select evenly spaced order statistics
    indices = jnp.linspace(0, n-1, m).astype(int)
    order_stats = sorted_data[indices]
    
    if not include_powers:
        return order_stats
    
    # Add powers of order statistics
    all_stats = [order_stats]
    for power in range(2, max_power + 1):
        all_stats.append(order_stats ** power)
    
    return jnp.concatenate(all_stats)


def create_g_and_k_benchmark_study(
    key: random.PRNGKey,
    n_datasets: int = 100,
    n_samples_per_dataset: int = 100
) -> Dict[str, Any]:
    """
    Create benchmark study following Fearnhead & Prangle (2011).
    
    Args:
        key: JAX random key
        n_datasets: Number of independent datasets
        n_samples_per_dataset: Samples per dataset
        
    Returns:
        Dictionary with benchmark datasets and setup
    """
    setup = get_fearnhead_prangle_setup()
    true_params = setup['true_params']
    
    datasets = []
    keys = random.split(key, n_datasets + 1)
    
    for i in range(n_datasets):
        data, _ = create_synthetic_g_and_k_data(
            keys[i+1], 
            true_params, 
            n_samples_per_dataset
        )
        datasets.append(data)
    
    return {
        'datasets': datasets,
        'true_params': true_params,
        'setup': setup,
        'n_datasets': n_datasets,
        'n_samples_per_dataset': n_samples_per_dataset
    }


# Backward compatibility alias
GAndKGenerator = GAndKModel


# Export main components
__all__ = [
    "GAndKModel",
    "generate_g_and_k_samples",
    "compute_g_and_k_quantiles", 
    "create_synthetic_g_and_k_data",
    "get_fearnhead_prangle_setup",
    "create_order_statistics_subset",
    "create_g_and_k_benchmark_study",
    # Backward compatibility
    "GAndKGenerator"
]