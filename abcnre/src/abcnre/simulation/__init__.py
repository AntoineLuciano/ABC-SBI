"""
ABC simulation module for ABCNRE.

This module provides classes and utilities for performing ABC simulation
and generating training data for neural ratio estimation.
"""

# Main simulator class
from .simulator import ABCSimulator

# Sampling utilities
from .sampler import RejectionSampler, BaseSampler

# Result structures
from .base import ABCSampleResult, ABCTrainingResult, ABCSingleResult

# Utility functions
from .utils import save_simulator, load_simulator
# from .utils import save_generator_config, load_generator_config

# Model imports
from .models.base import StatisticalModel
from .models.gauss_gauss import GaussGaussModel
from .models.g_and_k import GAndKModel

__all__ = [
    # Main classes
    "ABCSimulator",
    "RejectionSampler",
    "BaseSampler",
    
    # Result structures
    "ABCSampleResult",
    "ABCTrainingResult", 
    "ABCSingleResult",
    
    # Utilities
    "save_generator_config",
    "load_generator_config",
    
    # Models
    "StatisticalModel",
    "GaussGaussModel",
    "GAndKModel"
]
