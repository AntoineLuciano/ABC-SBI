"""
Statistical models for ABC simulation.

This module provides concrete implementations of statistical models
that can be used with the ABCSimulator for various ABC applications.

Available models:
- GaussGaussModel: Gaussian-Gaussian model with known variance
- GaussGaussMultiDimModel: Multidimensional Gaussian-Gaussian model
- GAndKModel: G-and-K distribution model

Example:
    from abcnre.simulation import ABCSimulator
    from abcnre.simulation.models import GaussGaussModel
    
    # Create model
    model = GaussGaussModel(mu0=0.0, sigma0=2.0, sigma=0.5)
    
    # Create simulator
    simulator = ABCSimulator(
        model=model,
        observed_data=my_data,
        quantile_distance=0.01
    )
"""

from .base import StatisticalModel
from .gauss_gauss import (
    GaussGaussModel,
    GaussGaussMultiDimModel
)
from .g_and_k import (
    GAndKModel,
    generate_g_and_k_samples,
    create_synthetic_g_and_k_data
)

__all__ = [
    # Base class
    'StatisticalModel',
    
    # Gaussian models
    'GaussGaussModel',
    'GaussGaussMultiDimModel',
    
    # G-and-K models
    'GAndKModel',
    'generate_g_and_k_samples', 
    'create_synthetic_g_and_k_data',
]

# Future models will be added here:
# from .linear_regression import LinearRegressionModel  
# from .logistic_regression import LogisticRegressionModel
# from .potus import PotusFullModel, PotusNatModel