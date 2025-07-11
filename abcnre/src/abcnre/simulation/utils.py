"""
Utilities for ABC simulation.

This module provides robust utility functions for saving and loading 
the state of an ABCSimulator, ensuring reproducibility.
"""

import yaml
import json
import hashlib
import importlib
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Any, Dict, Union, Optional


def generate_sampler_hash(
    model_config: Dict[str, Any],
    observed_data: jnp.ndarray,
    epsilon: float,
    length: int = 12
) -> str:
    """
    Generates a unique and deterministic hash for a sampler instance.

    Args:
        model_config: A dictionary with the statistical model's configuration.
        observed_data: The observed data array.
        epsilon: The ABC tolerance threshold.
        length: The desired length of the final hash string.

    Returns:
        A unique truncated SHA-256 hash as a hexadecimal string.
    """
    hasher = hashlib.sha256()

    # Serialize the config dictionary into a canonical JSON string
    config_str = json.dumps(model_config, sort_keys=True, ensure_ascii=False)
    hasher.update(config_str.encode('utf-8'))

    # Update with the raw bytes of the observed data array
    hasher.update(observed_data.tobytes())

    # Update with the string representation of epsilon
    hasher.update(str(epsilon).encode('utf-8'))

    return hasher.hexdigest()[:length]

# Forward declaration for type hinting
if 'ABCSimulator' not in globals():
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from .simulator import ABCSimulator


def save_simulator(
    simulator: 'ABCSimulator',
    config_path: Union[str, Path],
    observed_data_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Saves an ABCSimulator's configuration and data.

    This function serializes the simulator's state into a human-readable
    YAML configuration file and a separate NumPy file for the observed data.
    It embeds a content-based hash in the config for easy verification.

    Args:
        simulator: The ABCSimulator instance to save.
        config_path: The path where the YAML configuration file will be saved.
        observed_data_path: Optional path for the .npy file. If None, it's
            saved next to the config_path with a hash-based name.
    """
    config_path = Path(config_path)
    output_dir = config_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build a dictionary for hashing from the core components.
    model_class = simulator.model.__class__
    model_config = getattr(simulator.model, "get_model_args", lambda: {})()
    
    hash_inputs = {
        "model_class": f"{model_class.__module__}.{model_class.__name__}",
        "model_args": model_config,
        "observed_data": np.array(simulator.observed_data).tolist(), # Include data in hash
        "epsilon": float(simulator.epsilon),
        "simulator_config": simulator.config
    }
    
    # 2. Compute a short, stable hash from the JSON representation.
    config_str = json.dumps(hash_inputs, sort_keys=True)
    hash_name = hashlib.sha256(config_str.encode()).hexdigest()[:12]

    # 3. Save the observed data array.
    if observed_data_path is None:
        observed_data_path = output_dir / f"observed_data_{hash_name}.npy"
    else:
        observed_data_path = Path(observed_data_path)
    np.save(observed_data_path, simulator.observed_data)

    # 4. Create the final configuration dictionary for the YAML file.
    final_config = {
        "abcnre_version": "0.1.0",  # Example: good practice to version-stamp configs
        "config_hash": hash_name,
        "model_class": f"{model_class.__module__}.{model_class.__name__}",
        "model_args": model_config,
        "observed_data_path": str(observed_data_path.relative_to(output_dir)),
        "epsilon": float(simulator.epsilon),
        "simulator_config": simulator.config,
    }

    # 5. Save the configuration to the YAML file.
    with open(config_path, "w") as f:
        yaml.dump(final_config, f, default_flow_style=False, indent=2, sort_keys=False)
    
    if simulator.config.get('verbose', False):
        print(f"✅ Simulator saved with hash: {hash_name}")
        print(f"   - Configuration: {config_path}")
        print(f"   - Observed Data: {observed_data_path}")


def load_simulator(config_path: Union[str, Path]) -> 'ABCSimulator':
    """
    Loads an ABCSimulator from a configuration file.

    This function reconstructs a simulator by reading a YAML configuration
    file and loading the associated observed data from its NumPy file.

    Args:
        config_path: Path to the simulator's YAML configuration file.

    Returns:
        A fully instantiated ABCSimulator instance.
    """
    # Import locally to avoid circular dependencies
    from .simulator import ABCSimulator
    
    config_path = Path(config_path)
    config_dir = config_path.parent

    # 1. Load the YAML configuration.
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Recreate the statistical model.
    ModelClass = import_class_from_string(config["model_class"])
    model = ModelClass(**config.get("model_args", {}))

    # 3. Load the observed data array from the relative path.
    observed_data_path = config_dir / config["observed_data_path"]
    observed_data = np.load(observed_data_path)

    # 4. Re-instantiate the ABCSimulator.
    simulator = ABCSimulator(
        model=model,
        observed_data=jnp.array(observed_data),
        epsilon=config.get("epsilon"),
        config=config.get("simulator_config", {})
    )
    
    if simulator.config.get('verbose', False):
        print(f"✅ Simulator loaded from: {config_path}")

    return simulator


def import_class_from_string(class_path: str) -> type:
    """
    Imports a class from its string path.
    
    Args:
        class_path: Full path to the class (e.g., "abcnre.simulation.models.GaussGaussModel").
        
    Returns:
        The imported class object.
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not import class '{class_name}' from '{module_path}': {e}")


__all__ = [
    "save_simulator",
    "load_simulator",
    "import_class_from_string"
]