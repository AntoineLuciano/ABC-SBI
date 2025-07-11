"""
Base classes and result structures for ABC simulation.

This module contains the core result structures and base classes
used throughout the ABC simulation framework.
"""

import jax.numpy as jnp
from jax import random
from typing import NamedTuple, Optional
from abc import ABC, abstractmethod



class ABCSampleResult(NamedTuple):
    """Result structure for ABC sampling operations."""
    sim_data: jnp.ndarray
    theta_samples: jnp.ndarray
    distances: jnp.ndarray
    summary_stats: Optional[jnp.ndarray]
    key: random.PRNGKey
    rejection_count: Optional[jnp.ndarray] = None 

    


class ABCTrainingResult(NamedTuple):
    """Result structure for ABC training dataset generation."""
    features: jnp.ndarray
    labels: jnp.ndarray
    distances: jnp.ndarray
    summary_stats: Optional[jnp.ndarray]
    key: random.PRNGKey
    phi_samples: Optional[jnp.ndarray] = None 
    total_sim_count: int = 0


class ABCSingleResult(NamedTuple):
    """Result structure for single ABC sample."""
    sim_data: jnp.ndarray
    theta: jnp.ndarray
    distance: float
    summary_stat: Optional[jnp.ndarray]


class BaseSampler(ABC):
    """Base class for ABC samplers."""
    
    @abstractmethod
    def sample(self, key: random.PRNGKey, n_samples: int) -> ABCSampleResult:
        """Generate ABC samples."""
        pass
    
    @abstractmethod
    def sample_single(self, key: random.PRNGKey) -> ABCSingleResult:
        """Generate single ABC sample."""
        pass


# Export main components
__all__ = [
    "ABCSampleResult", 
    "ABCTrainingResult",
    "ABCSingleResult",
    "BaseSampler"
]