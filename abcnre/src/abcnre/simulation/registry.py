"""
Registry for simulation module components.

This module provides factory functions and configuration templates for:
- ABCSimulator instances
- Model configurations
- Simulator configuration templates

Functions follow the naming convention:
- create_simulator_from_dict()
- get_example_simulator_configs()
"""

from typing import Dict, Any, List, Union, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .samplers import ABCSimulator

# Configure logging
logger = logging.getLogger(__name__)


def create_simulator_from_dict(
    config: Dict[str, Any], observed_data: Optional[Any] = None
) -> "ABCSimulator":
    """
    Create an ABCSimulator instance from a configuration dictionary.

    Args:
        config: Configuration dictionary with simulator parameters
        observed_data: Optional observed data to attach to simulator

    Returns:
        Configured ABCSimulator instance

    Example:
        config = {
            "model": {"type": "GaussianModel", "mu": 0.0, "sigma": 1.0},
            "epsilon": 0.1,
            "quantile_distance": None,
            "simulator_config": {"num_samples": 1000}
        }
        simulator = create_simulator_from_dict(config, observed_data)
    """
    # RG: better not to do inline imports I think
    from .samplers import ABCSimulator

    # Validate required fields
    if "model" not in config:
        raise ValueError("Model configuration is required")
    if "epsilon" not in config:
        raise ValueError("Epsilon parameter is required")

    # Create model from configuration
    model = _create_model_from_dict(config["model"])

    # Extract simulator parameters
    epsilon = config["epsilon"]
    quantile_distance = config.get("quantile_distance")
    simulator_config = config.get("simulator_config", {})

    # Create simulator
    simulator = ABCSimulator(
        model=model,
        observed_data=observed_data,
        epsilon=epsilon,
        quantile_distance=quantile_distance,
        config=simulator_config,
    )

    logger.info(f"Created simulator: {type(simulator).__name__} with epsilon={epsilon}")
    return simulator





def get_supported_model_types() -> List[str]:
    """
    Get list of supported model types for simulator creation.

    Returns:
        List of supported model type strings
    """
    return ["GaussGaussModel", "GaussGaussMultiDimModel", "GAndKModel"]


def validate_simulator_config_dict(config: Dict[str, Any]) -> None:
    """
    Validate the structure of a simulator configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid or incomplete
    """
    # Required top-level fields
    required_fields = {"model", "epsilon"}
    missing_fields = required_fields - set(config.keys())

    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # Validate model configuration
    model_config = config["model"]
    if not isinstance(model_config, dict):
        raise ValueError("Model configuration must be a dictionary")

    if "type" not in model_config:
        raise ValueError("Model type is required in model configuration")

    model_type = model_config["type"]
    supported_types = get_supported_model_types()
    if model_type not in supported_types:
        raise ValueError(
            f"Unsupported model type '{model_type}'. Supported: {supported_types}"
        )

    # Validate epsilon
    try:
        epsilon = float(config["epsilon"])
        if epsilon < 0 and not np.isinf(epsilon):
            raise ValueError("Epsilon must be non-negative or infinity")
    except (ValueError, TypeError):
        raise ValueError("Epsilon must be a valid number")

    # Validate quantile_distance if present
    if "quantile_distance" in config and config["quantile_distance"] is not None:
        try:
            qd = float(config["quantile_distance"])
            if not 0 < qd <= 1:
                raise ValueError("Quantile distance must be between 0 and 1")
        except (ValueError, TypeError):
            raise ValueError("Quantile distance must be a valid number between 0 and 1")

    # Validate simulator_config if present
    if "simulator_config" in config:
        sim_config = config["simulator_config"]
        if not isinstance(sim_config, dict):
            raise ValueError("Simulator configuration must be a dictionary")


def _create_model_from_dict(model_config: Dict[str, Any]) -> Any:
    """
    Create a model instance from configuration dictionary.

    Args:
        model_config: Model configuration dictionary

    Returns:
        Model instance

    Raises:
        ValueError: If model type is unsupported or configuration is invalid
    """
    # Convert our simple format to the models/io.py expected format
    model_type = model_config["type"]

    # Extract all arguments except 'type'
    model_args = {k: v for k, v in model_config.items() if k != "type"}

    # Create the format expected by models/io.py
    models_io_config = {"model_type": model_type, "model_args": model_args}

    # Use the models registry
    from .models.io import create_model_from_dict

    return create_model_from_dict(models_io_config)


# Import numpy for validation
import numpy as np
