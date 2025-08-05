"""
Model registry for statistical models.

This module provides a centralized registry for all statistical models.
I/O operations are handled separately in the io.py module.
"""

from typing import Dict, Type
import logging
from pathlib import Path
import yaml
from .base import StatisticalModel

# Configure logging
logger = logging.getLogger(__name__)

# # Central registry of all available models
# MODEL_REGISTRY: Dict[str, Type[StatisticalModel]] = {}


from .gauss_gauss_1D import GaussGaussModel
from .gauss_gauss_multi import GaussGaussMultiDimModel
from .g_and_k import GAndKModel


MODEL_REGISTRY = {"GaussGaussModel": GaussGaussModel, 
                  "GaussGaussMultiDimModel": GaussGaussMultiDimModel, 
                    "GAndKModel": GAndKModel}



def register_model(model_type: str, model_class: Type[StatisticalModel]) -> None:
    """
    Register a new model type in the global registry.

    Args:
        model_type: Stable identifier for the model (used in YAML files)
        model_class: The actual Python class
    """
    if model_type in MODEL_REGISTRY:
        logger.warning(f"Overriding existing model type: {model_type}")

    MODEL_REGISTRY[model_type] = model_class
    logger.debug(f"Registered model: {model_type} -> {model_class.__name__}")


def get_available_models() -> Dict[str, str]:
    """
    Get all available model types and their corresponding class names.

    Returns:
        Dictionary mapping model_type -> class_name
    """
    return {model_type: cls.__name__ for model_type, cls in MODEL_REGISTRY.items()}


def get_example_model_configs(model_name = None) -> Dict[str, str]:
    """
    Get paths to example configuration files for each model type.

    Returns:
        Dictionary mapping scenario_name -> config_file_path
    """
    # Get the package root directory
    current_dir = Path(__file__).parent
    config_dir = current_dir.parent.parent.parent.parent / "examples" / "configs"/ "models"

    example_configs = {}

    if config_dir.exists():
        for config_file in config_dir.glob("*.yml"):
            scenario_name = config_file.stem
            example_configs[scenario_name] = str(config_file)
    if model_name:
        if model_name in example_configs:
            
            with open(example_configs[model_name], 'r') as f:
                config_content = yaml.safe_load(f)

            return config_content
        else:
            raise ValueError(f"No example config found for model: {model_name}\nAvailable models: {list(example_configs.keys())}")

    return example_configs


# # Auto-register models when module is imported
# def _auto_register_models():
#     """Automatically register all known models."""
#     try:
#         from .gauss_gauss_1D import GaussGaussModel
#         from .gauss_gauss_multi import GaussGaussMultiDimModel

#         register_model("GaussGaussModel", GaussGaussModel)
#         register_model("GaussGaussMultiDimModel", GaussGaussMultiDimModel)
#         logger.debug("Auto-registered Gaussian models")
#     except ImportError as e:
#         logger.warning(f"Could not auto-register models: {e}")


# Register models when module is imported
# _auto_register_models()
