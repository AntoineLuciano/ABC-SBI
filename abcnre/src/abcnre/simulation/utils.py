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
import pickle
from pathlib import Path
from typing import Any, Dict, Union, Optional, Callable, Tuple
import flax.serialization


def generate_sampler_hash(
    model_config: Dict[str, Any],
    observed_data: jnp.ndarray,
    epsilon: float,
    length: int = 12,
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
    hasher.update(config_str.encode("utf-8"))

    # Update with the raw bytes of the observed data array
    hasher.update(observed_data.tobytes())

    # Update with the string representation of epsilon
    hasher.update(str(epsilon).encode("utf-8"))

    return hasher.hexdigest()[:length]


# Forward declaration for type hinting
if "ABCSimulator" not in globals():
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from .simulator import ABCSimulator


def save_simulator(
    simulator: "ABCSimulator",
    config_path: Union[str, Path],
    observed_data_path: Optional[Union[str, Path]] = None,
    save_summary_stats: bool = True,
) -> None:
    """
    Saves an ABCSimulator's configuration and data.

    This function serializes the simulator's state into a human-readable
    YAML configuration file and a separate NumPy file for the observed data.
    It embeds a content-based hash in the config for easy verification.
    Optionally saves summary statistics if the model has learned ones.

    Args:
        simulator: The ABCSimulator instance to save.
        config_path: The path where the YAML configuration file will be saved.
        observed_data_path: Optional path for the .npy file. If None, it's
            saved next to the config_path with a hash-based name.
        save_summary_stats: Whether to save learned summary statistics if available.
    """
    config_path = Path(config_path)
    output_dir = config_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build a dictionary for hashing from the core components.
    model_class = simulator.model.__class__
    model_config = getattr(simulator.model, "get_model_args", lambda: {})()

    # Extract model_type from model_config if available (new YAML system)
    model_type = None
    if isinstance(model_config, dict) and "model_type" in model_config:
        model_type = model_config["model_type"]
        # Use the model_args from the config if available
        model_args = model_config.get("model_args", {})
    else:
        # Fallback: model_config is directly the arguments
        model_args = model_config if isinstance(model_config, dict) else {}

    hash_inputs = {
        "model_class": f"{model_class.__module__}.{model_class.__name__}",
        "model_type": model_type,
        "model_args": model_args,
        "observed_data": np.array(
            simulator.observed_data
        ).tolist(),  # Include data in hash
        "epsilon": float(simulator.epsilon),
        "simulator_config": simulator.config,
    }

    # 2. Compute a short, stable hash from the JSON representation.
    config_str = json.dumps(hash_inputs, sort_keys=True)
    hash_name = hashlib.sha256(config_str.encode()).hexdigest()[:12]

    # Create epsilon string for filenames
    epsilon_str = (
        f"eps_{simulator.epsilon:.0e}"
        if simulator.epsilon != float("inf")
        else "eps_inf"
    )
    hash_suffix = f"{epsilon_str}_{hash_name}"

    # 3. Save the observed data array.
    if observed_data_path is None:
        observed_data_path = output_dir / f"observed_data_{hash_suffix}.npy"
    else:
        observed_data_path = Path(observed_data_path)
    np.save(observed_data_path, simulator.observed_data)

    # 4. Check for learned summary statistics and save them
    summary_stats_config_path = None
    if save_summary_stats and hasattr(simulator.model, "summary_stat_fn"):
        # Check if the summary stats function has learned parameters
        if hasattr(simulator.model, "_summary_params") and hasattr(
            simulator.model, "_summary_config"
        ):
            summary_stats_config_path = output_dir / f"summary_stats_{hash_suffix}.yaml"
            save_summary_statistics(
                summary_function=simulator.model.summary_stat_fn,
                params=simulator.model._summary_params,
                config=simulator.model._summary_config,
                config_path=summary_stats_config_path,
            )

    # 5. Create the final configuration dictionary for the YAML file.
    final_config = {
        "abcnre_version": "0.1.0",  # Example: good practice to version-stamp configs
        "config_hash": hash_name,
        "hash_suffix": hash_suffix,  # Include full suffix for reference
        "model_class": f"{model_class.__module__}.{model_class.__name__}",
        "model_args": model_args,
        "observed_data_path": str(observed_data_path.relative_to(output_dir)),
        "epsilon": float(simulator.epsilon),
        "simulator_config": simulator.config,
        "paths": {
            "simulator_config_path": str(config_path.resolve()),
            "observed_data_path": str(observed_data_path.resolve()),
        },
    }

    # Add model_type if available (for new YAML system compatibility)
    if model_type:
        final_config["model_type"] = model_type

    # Add summary stats reference if available
    if summary_stats_config_path:
        final_config["summary_stats_config"] = str(
            summary_stats_config_path.relative_to(output_dir)
        )
        # Also add to paths section for consistency
        final_config["paths"]["summary_stats_config_path"] = str(
            summary_stats_config_path.resolve()
        )

    # 6. Save the configuration to the YAML file.
    with open(config_path, "w") as f:
        yaml.dump(final_config, f, default_flow_style=False, indent=2, sort_keys=False)

    if simulator.config.get("verbose", False):
        print(f"✅ Simulator saved with hash: {hash_name}")
        print(f"   - Configuration: {config_path}")
        print(f"   - Observed Data: {observed_data_path}")


def load_simulator(config_path: Union[str, Path]) -> "ABCSimulator":
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

    # 2. Recreate the statistical model using the registry system.
    try:
        # Try to use the new YAML-based registry first
        from .models.registry import MODEL_REGISTRY, _create_model_from_config

        # Build a config that the registry can understand
        model_config_for_registry = {
            "model_type": config.get("model_type"),
            "model_args": config.get("model_args", {}),
        }

        # If model_type is available, use the registry
        if (
            model_config_for_registry["model_type"]
            and model_config_for_registry["model_type"] in MODEL_REGISTRY
        ):
            model = _create_model_from_config(model_config_for_registry)
        else:
            # Fallback to the old method if model_type is not in config or registry
            ModelClass = import_class_from_string(config["model_class"])
            model = ModelClass(**config.get("model_args", {}))

    except (ImportError, KeyError) as e:
        # Final fallback to the original method
        ModelClass = import_class_from_string(config["model_class"])
        model = ModelClass(**config.get("model_args", {}))

    # 3. Load the observed data array from the relative path.
    observed_data_path = config_dir / config["observed_data_path"]
    observed_data = np.load(observed_data_path)

    # 4. Load summary statistics if available
    if "summary_stats_config" in config:
        summary_stats_config_path = config_dir / config["summary_stats_config"]
        try:
            summary_fn, summary_config = load_summary_statistics(
                summary_stats_config_path
            )
            # Attach the summary function and its metadata to the model
            model.summary_stat_fn = summary_fn
            model._summary_config = summary_config
            if config.get("simulator_config", {}).get("verbose", False):
                print(f"✅ Loaded learned summary statistics")
        except Exception as e:
            print(f"⚠️  Warning: Could not load summary statistics: {e}")

    # 5. Re-instantiate the ABCSimulator.
    simulator = ABCSimulator(
        model=model,
        observed_data=jnp.array(observed_data),
        epsilon=config.get("epsilon"),
        config=config.get("simulator_config", {}),
    )

    if simulator.config.get("verbose", False):
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
        raise ImportError(
            f"Could not import class '{class_name}' from '{module_path}': {e}"
        )


def save_summary_statistics(
    summary_function: Callable,
    params: Any,
    config: Dict[str, Any],
    config_path: Union[str, Path],
    params_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Saves summary statistics configuration and trained parameters.

    This function serializes the summary statistics learner's state into a
    YAML configuration file and a separate NPZ file for the parameters.
    It follows the same pattern as the inference module's persistence utilities.

    Args:
        summary_function: The trained summary statistics function
        params: The JAX/Flax parameters from training
        config: The learner configuration dictionary containing:
            - 'learner_type': 'MLP' or 'DeepSet'
            - 'param_dim': Parameter dimension
            - 'data_shape': Input data shape
            - 'hidden_dims': Network architecture
            - Other training hyperparameters
        config_path: The path where the YAML configuration file will be saved
        params_path: Optional path for the parameters file. If None, it's
            saved next to the config_path with a hash-based name
    """
    config_path = Path(config_path)
    output_dir = config_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build configuration dictionary for hashing
    data_shape = config.get("data_shape")
    if isinstance(data_shape, tuple):
        data_shape = list(data_shape)  # Convert tuple to list for YAML compatibility

    hash_inputs = {
        "learner_type": config.get("learner_type", "MLP"),
        "param_dim": config.get("param_dim"),
        "data_shape": data_shape,
        "hidden_dims": config.get("hidden_dims", [64, 32]),
        "learning_rate": config.get("learning_rate", 3e-4),
        "n_epochs": config.get("n_epochs", 100),
        "lr_schedule": config.get("lr_schedule", "constant"),
        "optimizer_type": config.get("optimizer_type", "adam"),
    }

    # 2. Compute a short, stable hash from the JSON representation
    config_str = json.dumps(hash_inputs, sort_keys=True)
    hash_name = hashlib.sha256(config_str.encode()).hexdigest()[:12]

    # 3. Save the model parameters
    if params_path is None:
        params_path = output_dir / f"summary_params_{hash_name}.npz"
    else:
        params_path = Path(params_path)

    # Serialize JAX/Flax parameters to bytes for NPZ storage
    params_bytes = flax.serialization.to_bytes(params)
    np.savez_compressed(params_path, params_bytes=params_bytes)

    # 4. Create the final configuration dictionary for the YAML file
    final_config = {
        "abcnre_version": "0.1.0",
        "config_hash": hash_name,
        "summary_stats_type": "learned",
        "learner_type": config.get("learner_type", "MLP"),
        "params_path": str(params_path.relative_to(output_dir)),
        "model_config": {
            "param_dim": config.get("param_dim"),
            "data_shape": list(config.get("data_shape", [])),
            "hidden_dims": config.get("hidden_dims", [64, 32]),
            "output_dim": config.get("param_dim"),  # For backward compatibility
        },
        "training_config": {
            "learning_rate": config.get("learning_rate", 3e-4),
            "n_epochs": config.get("n_epochs", 100),
            "batch_size": config.get("batch_size", 256),
            "lr_schedule": config.get("lr_schedule", "constant"),
            "lr_schedule_args": config.get("lr_schedule_args", {}),
            "optimizer_type": config.get("optimizer_type", "adam"),
            "weight_decay": config.get("weight_decay", 0.0),
        },
        "metadata": {
            "loss_type": "direct",  # Indicates we learn s(x) ≈ φ directly
            "timestamp": {
                "learner_type": config.get("learner_type", "MLP"),
                "param_dim": config.get("param_dim"),
                "data_shape": (
                    list(config.get("data_shape", []))
                    if isinstance(config.get("data_shape"), tuple)
                    else config.get("data_shape", [])
                ),
                "hidden_dims": config.get("hidden_dims", [64, 32]),
                "learning_rate": config.get("learning_rate", 3e-4),
                "n_epochs": config.get("n_epochs", 100),
                "lr_schedule": config.get("lr_schedule", "constant"),
                "optimizer_type": config.get("optimizer_type", "adam"),
            },  # For reproducibility tracking without Python-specific types
        },
    }

    # 5. Save the configuration to the YAML file
    with open(config_path, "w") as f:
        yaml.dump(final_config, f, default_flow_style=False, indent=2, sort_keys=False)

    if config.get("verbose", False):
        print(f"✅ Summary stats saved with hash: {hash_name}")
        print(f"   - Configuration: {config_path}")
        print(f"   - Parameters: {params_path}")


def load_summary_statistics(
    config_path: Union[str, Path],
) -> Tuple[Callable, Dict[str, Any]]:
    """
    Loads summary statistics function from a configuration file.

    This function reconstructs a summary statistics function by reading a YAML
    configuration file and loading the associated parameters from the NPZ file.

    Args:
        config_path: Path to the summary stats YAML configuration file

    Returns:
        A tuple containing:
        - The reconstructed summary statistics function
        - The configuration dictionary for reference
    """
    config_path = Path(config_path)
    config_dir = config_path.parent

    # 1. Load the YAML configuration
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Validate configuration
    if config.get("summary_stats_type") != "learned":
        raise ValueError(
            f"Invalid summary stats type: {config.get('summary_stats_type')}"
        )

    learner_type = config.get("learner_type")
    if learner_type not in ["MLP", "DeepSet"]:
        raise ValueError(f"Unsupported learner type: {learner_type}")

    # 3. Load the model parameters
    params_path = config_dir / config["params_path"]
    if not params_path.is_file():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")

    # 3. Load the model parameters
    params_path = config_dir / config["params_path"]
    if not params_path.is_file():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")

    # 4. Recreate the summary statistics model first
    model_config = config["model_config"]

    if learner_type == "DeepSet":
        from ..preprocessing.summary_networks import SummaryDeepSet

        model = SummaryDeepSet(
            summary_dim=model_config["output_dim"],
            phi_hidden_dims=model_config["hidden_dims"],
            rho_hidden_dims=model_config["hidden_dims"],
        )
    else:  # MLP
        from ..preprocessing.summary_networks import SummaryMLP

        model = SummaryMLP(
            summary_dim=model_config["output_dim"],
            hidden_dims=model_config["hidden_dims"],
        )

    # Load and deserialize JAX/Flax parameters from NPZ
    params_data = np.load(params_path)
    params_bytes = params_data["params_bytes"]

    # Initialize model parameters with dummy data to get the template
    try:
        import jax
        import jax.numpy as jnp

        dummy_key = jax.random.PRNGKey(0)
        if learner_type == "DeepSet":
            # DeepSet expects (batch, n_samples, features)
            dummy_data = jnp.ones((1, 10, 1))  # Adjust based on data_shape
        else:  # MLP
            dummy_data = jnp.ones((1, 10))  # Flatten for MLP

        # Initialize to get parameter template
        param_template = model.init(dummy_key, dummy_data, training=False)
        # Now deserialize with the correct template
        params = flax.serialization.from_bytes(param_template, params_bytes)
    except Exception as e:
        # Fallback to the original method if initialization fails
        params = flax.serialization.from_bytes(None, params_bytes)

    # 5. Create the summary statistics function
    def summary_fn(x):
        """Reconstructed summary statistics function."""
        # Prepare data format based on model type
        if learner_type == "DeepSet":
            # DeepSet expects (batch, n_samples, features)
            if x.ndim == 2:
                x_formatted = x[None, :, :]  # Add batch dimension
            else:
                x_formatted = x
        else:  # MLP
            # MLP expects (batch, features) - use first sample if multiple
            if x.ndim == 2:
                x_formatted = x[:1]  # Take first sample
            else:
                x_formatted = x

        return model.apply(params, x_formatted, training=False)

    if config.get("verbose", False):
        print(f"✅ Summary stats loaded from: {config_path}")
        print(f"   - Learner type: {learner_type}")
        print(f"   - Output dimension: {model_config['output_dim']}")

    return summary_fn, config


__all__ = [
    "save_simulator",
    "load_simulator",
    "import_class_from_string",
    "save_summary_statistics",
    "load_summary_statistics",
]
