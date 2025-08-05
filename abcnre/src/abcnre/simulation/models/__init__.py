"""
Statistical models for ABC simulation.

This module provides concrete implementations of statistical models
that can be used with the ABCSimulator for various ABC applications.

Available models:
- GaussGaussModel: Gaussian-Gaussian model with known variance
- GaussGaussMultiDimModel: Multidimensional Gaussian-Gaussian model
- GAndKModel: G-and-K distribution model

Example YAML-first workflow:
    from abcnre.simulation.models import create_model_from_yaml, save_model_to_yaml

    # Load any model from YAML
    model = create_model_from_yaml("my_model.yml")

    # Save any model to YAML
    save_model_to_yaml(model, "output_model.yml")

Example direct usage:
    from abcnre.simulation.models import GaussGaussModel

    # Create model directly
    model = GaussGaussModel(mu0=0.0, sigma0=2.0, sigma=0.5)
"""

from .base import StatisticalModel
from .gauss_gauss_1D import GaussGaussModel
from .gauss_gauss_multi import GaussGaussMultiDimModel
from .g_and_k import GAndKModel, generate_g_and_k_samples

# Registry functions
from .registry import (
    register_model,
    get_available_models,
    get_example_model_configs,
    MODEL_REGISTRY,
)

# I/O functions
from .io import (
    create_model_from_yaml,
    create_model_from_dict,
    save_model_to_yaml,
    validate_model_config_yaml,
    validate_model_config_dict,
)

__all__ = [
    # I/O functions
    "create_model_from_yaml",
    "create_model_from_dict",
    "save_model_to_yaml",
    "validate_model_config_yaml",
    "validate_model_config_dict",
    # Registry functions
    "register_model",
    "get_available_models",
    "get_example_model_configs",
    "MODEL_REGISTRY",
    # Models
    "StatisticalModel",
    "GaussGaussModel",
    "GaussGaussMultiDimModel",
    "GAndKModel",
    "generate_g_and_k_samples",
    "create_synthetic_g_and_k_data",
]
