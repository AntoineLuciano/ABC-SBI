"""
Neural network architectures for NRE using Flax.

This module provides various neural network architectures optimized
for neural ratio estimation in ABC inference.
"""

from .base import NetworkBase
from .mlp import MLPNetwork, SimpleMLP, ResidualMLP
from .deepset import DeepSetNetwork, CompactDeepSetNetwork

__all__ = [
    "NetworkBase",
    "MLPNetwork",
    "SimpleMLP", 
    "ResidualMLP",
    "DeepSetNetwork",
    "CompactDeepSetNetwork"
]