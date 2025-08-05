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

# Utility functions - now only from io.py (utils.py is minimal)
# Legacy imports redirected to io.py for backward compatibility
from .io import (
    save_simulator_to_yaml as save_simulator,
    load_simulator_from_yaml as load_simulator,
)

# New dedicated I/O functions
from .io import (
    save_simulator_to_yaml,
    load_simulator_from_yaml,
    validate_simulator_config_yaml,
    validate_simulator_config_dict,
)

# Registry functions
from .registry import (
    create_simulator_from_dict,
    get_supported_model_types,
    validate_simulator_config_dict as validate_config_dict,
)

# Model imports
from .models.base import StatisticalModel, SummarizedStatisticalModel
from .models.gauss_gauss_1D import GaussGaussModel
from .models.g_and_k import GAndKModel

__all__ = [
    # Main classes
    "ABCSimulator",
    "RejectionSampler",
    "BaseSampler",
    "SummarizedStatisticalModel",
    # Result structures
    "ABCSampleResult",
    "ABCTrainingResult",
    "ABCSingleResult",
    # Legacy utilities
    "save_simulator",
    "load_simulator",
    # New I/O functions
    "save_simulator_to_yaml",
    "load_simulator_from_yaml",
    "validate_simulator_config_yaml",
    "validate_simulator_config_dict",
    # Registry functions
    "create_simulator_from_dict",
    "get_supported_model_types",
    "validate_config_dict",
    # Models
    "StatisticalModel",
    "GaussGaussModel",
    "GAndKModel",
]
