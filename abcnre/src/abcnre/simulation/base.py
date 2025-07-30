"""
Base classes and result structures for ABC simulation.

This module contains the core result structures and base classes
used throughout the ABC simulation framework.
"""

import jax.numpy as jnp
from jax import random
from typing import NamedTuple, Optional, List, Dict
from abc import ABC, abstractmethod


class ABCSampleResult(NamedTuple):
    """Result structure for ABC sampling operations."""

    data: jnp.ndarray
    theta: jnp.ndarray
    distances: jnp.ndarray
    summary_stats: Optional[jnp.ndarray]
    key: random.PRNGKey
    simulation_count: Optional[jnp.ndarray] = None
    phi: Optional[jnp.ndarray] = None


class ABCTrainingResult(NamedTuple):
    """
    Result structure for ABC training dataset generation.

    Supports both flattened features (backward compatibility) and structured features
    (Option A.1 implementation for advanced architectures like DeepSet).
    """

    labels: jnp.ndarray  # Binary labels (0=joint, 1=marginal)
    data :   jnp.ndarray  # Training data samples
    distances: jnp.ndarray  # ABC distances
    summary_stats: Optional[jnp.ndarray]  # Summary statistics if available
    key: random.PRNGKey
    theta: jnp.ndarray  
    phi: Optional[jnp.ndarray] = None  # Parameter of interest samples
    total_sim_count: int = 0  # Total simulation count


class ABCSingleResult(NamedTuple):
    """Result structure for single ABC sample."""

    sim_data: jnp.ndarray
    theta: jnp.ndarray
    distance: float
    summary_stat: Optional[jnp.ndarray]
    phi: Optional[jnp.ndarray] = None


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
__all__ = ["ABCSampleResult", "ABCTrainingResult", "ABCSingleResult", "BaseSampler"]
