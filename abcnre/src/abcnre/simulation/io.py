"""
I/O operations for simulation module - ABCSimulator configurations and trained components.

This module handles saving and loading of:
- Complete ABCSimulator instances (model + summary stats + configuration)
- Individual model configurations
- Summary statistics networks (via preprocessing module)
- Observed data and simulator state

Functions follow the naming convention:
- save_simulator_to_yaml() / load_simulator_from_yaml()
- validate_simulator_config_yaml() / validate_simulator_config_dict()
"""

from typing import Dict, Any, Union, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import yaml
import logging
import numpy as np
import jax.numpy as jnp
from datetime import datetime
import json
import hashlib

if TYPE_CHECKING:
    from .samplers import ABCSimulator
    from ..training.config import NNConfig

# Configure logging
logger = logging.getLogger(__name__)


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


def save_simulator_to_yaml(
    simulator: "ABCSimulator",
    output_path: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """
    Save an ABCSimulator configuration to YAML format.

    Creates multiple files:
        - simulator.yaml: Main configuration with references to other files
        - simulator_model.yml: Model configuration
        - simulator_observed_data.npy: Observed data (if present)
        - simulator_summary_network_config.yml: Summary network config (if trained)
        - simulator_summary_network_weights.npz: Network weights (if trained)
    """
    output_path = Path(output_path)

    # Check if we should skip saving
    if _should_skip_save(output_path, overwrite):
        return

    # Create output directory
    _ensure_output_directory(output_path)

    # Extract base simulator configuration
    simulator_config = _extract_simulator_config(simulator)

    # Save model configuration
    _save_model_configuration(simulator, simulator_config, output_path, overwrite)

    # Save observed data if present
    _save_observed_data(simulator, simulator_config, output_path, overwrite)

    # Save summary statistics if trained
    _save_summary_statistics(simulator, simulator_config, output_path, overwrite)

    # Save main simulator configuration file
    _save_main_config(simulator_config, output_path)


def _should_skip_save(output_path: Path, overwrite: bool) -> bool:
    """Check if saving should be skipped due to existing file."""
    if output_path.exists() and not overwrite:
        logger.info(f"Output path already exists and overwrite is False: {output_path}")
        return True
    return False


def _ensure_output_directory(output_path: Path) -> None:
    """Create output directory if it doesn't exist."""
    output_path.parent.mkdir(parents=True, exist_ok=True)


def _save_model_configuration(
    simulator: "ABCSimulator",
    simulator_config: Dict[str, Any],
    output_path: Path,
    overwrite: bool,
) -> None:
    """Save model configuration to separate file."""
    model_filename = f"{output_path.stem}_model.yml"
    model_path = output_path.parent / model_filename
    _save_model_config(simulator.model, model_path, overwrite=overwrite)
    simulator_config["paths"]["model"] = model_filename


def _save_observed_data(
    simulator: "ABCSimulator",
    simulator_config: Dict[str, Any],
    output_path: Path,
    overwrite: bool,
) -> None:
    """Save observed data if present."""
    if simulator.observed_data is None:
        return

    observed_data_filename = f"{output_path.stem}_observed_data.npy"
    observed_data_path = output_path.parent / observed_data_filename

    if _should_save_file(observed_data_path, overwrite):
        np.save(observed_data_path, simulator.observed_data)
        logger.info(f"Saved observed data to: {observed_data_path}")
        simulator_config["paths"]["observed_data"] = observed_data_filename
    else:
        logger.info(
            f"Observed data file already exists and overwrite is False: {observed_data_path}"
        )


def _save_summary_statistics(
    simulator: "ABCSimulator",
    simulator_config: Dict[str, Any],
    output_path: Path,
    overwrite: bool,
) -> None:
    """Save summary statistics configuration and weights if trained."""
    if not simulator.trained_summary_stats:
        simulator_config["paths"]["summary_network_config"] = None
        simulator_config["paths"]["summary_network_weights"] = None
        return

    # Save summary network config
    summary_config_filename = f"{output_path.stem}_summary_network_config.yml"
    summary_config_path = output_path.parent / summary_config_filename
    simulator._summary_config.save(summary_config_path, overwrite=overwrite)
    simulator_config["paths"]["summary_network_config"] = summary_config_filename

    # Save summary network weights
    _save_summary_weights(simulator, simulator_config, output_path, overwrite)


def _save_summary_weights(
    simulator: "ABCSimulator",
    simulator_config: Dict[str, Any],
    output_path: Path,
    overwrite: bool,
) -> None:
    """Save summary network weights using Flax serialization."""
    params = simulator._summary_params
    summary_params_filename = f"{output_path.stem}_summary_network_weights.npz"
    summary_params_path = output_path.parent / summary_params_filename

    if _should_save_file(summary_params_path, overwrite):
        try:
            import flax.serialization

            params_bytes = flax.serialization.to_bytes(params)
            # Save bytes as a numpy array using frombuffer to preserve them correctly
            params_array = np.frombuffer(params_bytes, dtype=np.uint8)
            np.savez_compressed(summary_params_path, params=params_array)
            simulator_config["paths"][
                "summary_network_weights"
            ] = summary_params_filename
            logger.info(f"Saved summary network weights to: {summary_params_path}")
        except ImportError:
            raise ImportError(
                "Flax serialization is required to save summary network weights. "
                "Please install Flax."
            )
    else:
        logger.info(
            f"Summary network weights file already exists and overwrite is False: {summary_params_path}"
        )


def _should_save_file(file_path: Path, overwrite: bool) -> bool:
    """Check if a file should be saved based on existence and overwrite flag."""
    return not file_path.exists() or overwrite


def _save_main_config(simulator_config: Dict[str, Any], output_path: Path) -> None:
    """Save the main simulator configuration to YAML file."""
    try:
        with open(output_path, "w") as f:
            yaml.dump(simulator_config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"ABCSimulator configuration saved to: {output_path}")
    except Exception as e:
        raise ValueError(
            f"Failed to save simulator configuration to {output_path}: {e}"
        )


def load_simulator_from_yaml(yaml_path: Union[str, Path]) -> "ABCSimulator":
    """
    Load a complete ABCSimulator from YAML configuration.

    Args:
        yaml_path: Path to the simulator YAML configuration file

    Returns:
        Reconstructed ABCSimulator instance

    Raises:
        FileNotFoundError: If YAML file or associated files don't exist
        ValueError: If configuration is invalid or loading fails
    """
    yaml_path = Path(yaml_path)

    # Load and validate configuration
    config = _load_simulator_config(yaml_path)

    # Load model
    model = _load_model_from_config(config, yaml_path.parent)

    # Load observed data if present
    observed_data = _load_observed_data_from_config(config, yaml_path.parent)

    # Create simulator instance
    simulator = _create_simulator_instance(config, model, observed_data)

    # Load summary statistics if available
    _load_summary_statistics(config, yaml_path.parent, simulator)

    return simulator


def _load_simulator_config(yaml_path: Path) -> Dict[str, Any]:
    """Load and validate simulator configuration from YAML file."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Simulator YAML file not found: {yaml_path}")

    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {yaml_path}: {e}")

    _validate_simulator_config_dict(config, yaml_path)
    return config


def _load_model_from_config(config: Dict[str, Any], yaml_dir: Path):
    """Load model from configuration."""
    if "model" not in config["paths"]:
        raise ValueError("Missing model path in simulator configuration")

    model_path = yaml_dir / config["paths"]["model"]
    model = _load_model_config(model_path)
    logger.info(f"Loaded model: {type(model).__name__}")
    return model


def _load_observed_data_from_config(config: Dict[str, Any], yaml_dir: Path):
    """Load observed data from configuration if present."""
    if "observed_data" not in config["paths"]:
        return None

    observed_data_path = yaml_dir / config["paths"]["observed_data"]
    if observed_data_path.exists():
        observed_data = np.load(observed_data_path)
        logger.info(f"Loaded observed data from: {observed_data_path}")
        return observed_data
    else:
        logger.warning(f"Observed data file not found: {observed_data_path}")
        return None


def _create_simulator_instance(config: Dict[str, Any], model, observed_data):
    """Create ABCSimulator instance from configuration."""
    from .samplers import ABCSimulator

    return ABCSimulator(
        model=model,
        observed_data=observed_data,
        epsilon=config["epsilon"],
        quantile_distance=config.get("quantile_distance"),
        config=config.get("simulator_config", {}),
    )


def _load_summary_statistics(config: Dict[str, Any], yaml_dir: Path, simulator) -> None:
    """Load summary statistics configuration and weights if available."""
    summary_config_path_str = config["paths"].get("summary_network_config")
    summary_weights_path_str = config["paths"].get("summary_network_weights")

    if not summary_config_path_str or not summary_weights_path_str:
        return

    summary_config_path = yaml_dir / summary_config_path_str
    summary_params_path = yaml_dir / summary_weights_path_str

    if not (summary_config_path.exists() and summary_params_path.exists()):
        logger.warning("Summary statistics files not found, skipping...")
        return

    _apply_summary_statistics(simulator, summary_config_path, summary_params_path)


def _apply_summary_statistics(
    simulator, summary_config_path: Path, summary_params_path: Path
) -> None:
    """Apply summary statistics configuration and weights to simulator."""
    try:
        from ..training.config import NNConfig
        from ..training.registry import create_network_from_nn_config
        import flax.serialization

        # Load summary network configuration and weights
        summary_config = NNConfig.load(summary_config_path)

        # Load the bytes array and convert back to bytes
        params_array = np.load(summary_params_path)["params"]
        params_bytes = params_array.tobytes()

        summary_params = flax.serialization.from_bytes(None, params_bytes)

        logger.info(f"Loaded summary network config from: {summary_config_path}")
        logger.info(f"Loaded summary network weights from: {summary_params_path}")

        # Apply to simulator
        from .utils import create_summary_stats_fn

        simulator._summary_config = summary_config
        simulator._summary_params = summary_params

        summary_network = create_network_from_nn_config(summary_config)
        summary_fn = create_summary_stats_fn(
            network=summary_network,
            params=summary_params)

        simulator.model.summary_stat_fn = summary_fn
        simulator.summary_stat_fn = summary_fn
        simulator.config["summary_stats_enabled"] = True
        simulator._initialize_sampler()

        logger.info("Summary network loaded and integrated into simulator")

    except ImportError:
        raise ImportError(
            "Flax serialization is required to load summary network weights. "
            "Please install Flax."
        )


def validate_simulator_config_dict(
    config: Dict[str, Any], source_path: Optional[Path] = None
) -> None:
    """
    Validate the structure of a simulator configuration dictionary.

    Args:
        config: Configuration dictionary to validate
        source_path: Optional source file path for better error messages

    Raises:
        ValueError: If configuration is invalid
    """
    _validate_simulator_config_dict(config, source_path)


def validate_simulator_config_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a simulator YAML configuration file.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        Validated configuration dictionary

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If YAML configuration is invalid
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {yaml_path}: {e}")

    _validate_simulator_config_dict(config, yaml_path)
    return config


def _extract_simulator_config(simulator: "ABCSimulator") -> Dict[str, Any]:
    """
    Extract configuration from a simulator instance.

    Args:
        simulator: ABCSimulator instance to extract config from

    Returns:
        Configuration dictionary
    """
    config = {
        "epsilon": float(simulator.epsilon),
        "quantile_distance": simulator.quantile_distance,
        "simulator_config": simulator.config,
        "paths": {},  # Will be filled by save function
        "metadata": {
            "created_date": datetime.now().isoformat(),
            "abcnre_version": "0.1.0",
            "simulator_type": type(simulator).__name__,
            "sampler_id": getattr(simulator, "sampler_id", None),
        },
    }

    # Add observed data info if present
    if simulator.observed_data is not None:
        config["observed_data_shape"] = list(simulator.observed_data.shape)

    return config


def _validate_simulator_config_dict(
    config: Dict[str, Any], source_path: Optional[Path] = None
) -> None:
    """Internal validation function for simulator configuration."""
    source_info = f" in {source_path}" if source_path else ""

    # Required top-level fields
    required_fields = {"epsilon", "paths", "metadata"}
    missing_fields = required_fields - set(config.keys())

    if missing_fields:
        raise ValueError(f"Missing required fields{source_info}: {missing_fields}")

    # Validate paths section
    if not isinstance(config["paths"], dict):
        raise ValueError(f"'paths' must be a dictionary{source_info}")

    if "model" not in config["paths"]:
        raise ValueError(f"Missing required path 'model'{source_info}")

    # Validate epsilon
    try:
        epsilon = float(config["epsilon"])
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative{source_info}")
    except (ValueError, TypeError):
        raise ValueError(f"epsilon must be a valid number{source_info}")


def _save_model_config(model: Any, output_path: Path, overwrite: bool) -> None:
    """Save model configuration to YAML."""
    from .models.io import save_model_to_yaml

    if output_path.exists() and not overwrite:
        logger.info(f"Output path already exists and overwrite is False: {output_path}")
    else:
        # Save model configuration to YAML
        save_model_to_yaml(model, output_path)
        logger.info(f"Saved model configuration to: {output_path}")


def _load_model_config(yaml_path: Path) -> Any:
    """Load model from YAML configuration."""
    from .models.io import create_model_from_yaml

    return create_model_from_yaml(yaml_path)


def _extract_network_config(network: Any) -> Dict[str, Any]:
    """Extract configuration from a network instance."""
    network_type = type(network).__name__

    config = {
        "type": network_type,
        "summary_dim": getattr(network, "summary_dim", None),
    }

    # Extract type-specific configuration
    if hasattr(network, "hidden_dims"):
        config["hidden_dims"] = network.hidden_dims

    # DeepSet specific
    if network_type == "SummaryDeepSet":
        config.update(
            {
                "phi_hidden_dims": getattr(network, "phi_hidden_dims", None),
                "rho_hidden_dims": getattr(network, "rho_hidden_dims", None),
                "pooling_type": getattr(network, "pooling_type", "mean"),
            }
        )

    # Common network properties
    config.update(
        {
            "use_layer_norm": getattr(network, "use_layer_norm", True),
            "dropout_rate": getattr(network, "dropout_rate", 0.0),
            "activation": getattr(network, "activation", "relu"),
        }
    )

    return config
