"""
I/O operations for statistical models - YAML serialization/deserialization.

This module handles saving and loading statistical models to/from YAML files.
"""

from typing import Dict, Any, Union
from pathlib import Path
import yaml
import logging

from .base import StatisticalModel
from .registry import MODEL_REGISTRY

# Configure logging
logger = logging.getLogger(__name__)


def create_model_from_yaml(yaml_path: Union[str, Path]) -> StatisticalModel:
    """
    Create a model instance directly from a YAML file.
    
    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        Instantiated model

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If model_type is not registered
        TypeError: If model arguments are invalid
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {yaml_path}: {e}")

    return create_model_from_dict(config, yaml_path)


def create_model_from_dict(
    config: Dict[str, Any], source_path: Path = None
) -> StatisticalModel:
    """
    Create a model instance from a configuration dictionary.

    Args:
        config: Model configuration dictionary
        source_path: Optional source file path for better error messages

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_type is not registered or config is invalid
        TypeError: If model arguments are invalid
    """
    # Validate required fields
    required_fields = {"model_type", "model_args"}
    missing_fields = required_fields - set(config.keys())
    if missing_fields:
        source_info = f" in {source_path}" if source_path else ""
        raise ValueError(f"Missing required fields{source_info}: {missing_fields}")

    model_type = config["model_type"]
    model_args = config["model_args"]

    # Check if model type is registered
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        source_info = f" from {source_path}" if source_path else ""
        raise ValueError(
            f"Unknown model type '{model_type}'{source_info}. "
            f"Available types: {available}"
        )

    # Get the model class and instantiate
    model_class = MODEL_REGISTRY[model_type]

    try:
        return model_class(**model_args)
    except TypeError as e:
        source_info = f" from {source_path}" if source_path else ""
        raise TypeError(f"Invalid arguments for {model_type}{source_info}: {e}")


def save_model_to_yaml(model: StatisticalModel, yaml_path: Union[str, Path]) -> None:
    """
    Save a model instance to a YAML file.

    Args:
        model: Model instance to save
        yaml_path: Output YAML file path
    """
    yaml_path = Path(yaml_path)

    # Get model configuration
    config = model.get_model_args()

    # Ensure metadata exists
    if "metadata" not in config:
        config["metadata"] = {}

    # Add timestamp info
    from datetime import datetime

    config["metadata"]["saved_at"] = datetime.now().isoformat()

    # Ensure output directory exists
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to YAML
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    logger.info(f"Model saved to: {yaml_path}")


# def save_model_to_dict(model: StatisticalModel) -> Dict[str, Any]:
#     """
#     Convert a model instance to a configuration dictionary.

#     Args:
#         model: Model instance to convert

#     Returns:
#         Configuration dictionary
#     """
#     config = model.get_model_args()

#     # Ensure metadata exists
#     if "metadata" not in config:
#         config["metadata"] = {}

#     # Add timestamp info
#     from datetime import datetime

#     config["metadata"]["created_at"] = datetime.now().isoformat()

#     return config


def validate_model_config_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a YAML model configuration without creating the model.

    Args:
        yaml_path: Path to YAML file

    Returns:
        Validated configuration dictionary

    Raises:
        Various validation errors
    """
    yaml_path = Path(yaml_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate structure
    validate_model_config_dict(config)

    return config


def validate_model_config_dict(config: Dict[str, Any]) -> None:
    """
    Validate the structure of a model configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = {"model_type", "model_args"}
    missing_fields = required_fields - set(config.keys())

    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    if config["model_type"] not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type: {config['model_type']}. " f"Available: {available}"
        )


