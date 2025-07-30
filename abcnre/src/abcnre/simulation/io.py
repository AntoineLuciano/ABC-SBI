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

if TYPE_CHECKING:
    from .simulator import ABCSimulator
    from ..training.config import NNConfig

# Configure logging
logger = logging.getLogger(__name__)



def save_simulator_to_yaml(
    simulator: "ABCSimulator",
    output_path: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """
    Save a complete ABCSimulator to YAML + associated files.

    Args:
        simulator: The ABCSimulator instance to save
        output_path: Path to save the main YAML config file
        save_summary_stats: Whether to save trained summary statistics
        metadata: Optional additional metadata

    Creates:
        - simulator.yml: Main configuration file
        - simulator_model.yml: Model configuration
        - simulator_observed_data.npy: Observed data (if present)
        - simulator_summary_network.yml: Summary network (if trained)
        - simulator_summary_network_weights.npz: Network weights (if trained)
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not overwrite:
        logger.info(f"Output path already exists and overwrite is False: {output_path}")
        return
    else:
        simulator_config = _extract_simulator_config(simulator)
        # Save model configuration
        model_filename = f"{output_path.stem}_model.yml"
        model_path = output_dir / model_filename
        _save_model_config(simulator.model, model_path, overwrite=overwrite)
        simulator_config["paths"]["model"] = model_filename
        # Save observed data if present
        if simulator.observed_data is not None:
            observed_data_filename = f"{output_path.stem}_observed_data.npy"
            observed_data_path = output_dir / observed_data_filename
            if not observed_data_path.exists() or overwrite:
                np.save(observed_data_path, simulator.observed_data)
                logger.info(f"Saved observed data to: {observed_data_path}")
            else:
                logger.info(f"Observed data file already exists and overwrite is False: {observed_data_path}")
            simulator_config["paths"]["observed_data"] = observed_data_filename

        # Save summary statistics if trained and requested
        if simulator.trained_summary_stats:
        
            summary_config_filename = f"{output_path.stem}_summary_network_config.yml"
            summary_config_path = output_dir / summary_config_filename
            simulator._summary_config.save(summary_config_path, overwrite=overwrite)
            simulator_config["paths"]["summary_network_config"] = summary_config_filename
            
            
            params = simulator._summary_params
            summary_params_filename = f"{output_path.stem}_summary_network_weights.npz"
            summary_params_path = output_dir / summary_params_filename
            if not summary_params_path.exists() or overwrite:
                try:
                    import flax.serialization
                    params_bytes = flax.serialization.to_bytes(params)
                    np.savez_compressed(summary_params_path, params=params_bytes)
                    simulator_config["paths"]["summary_network_weights"] = summary_params_filename
                    logger.info(f"Saved summary network weights to: {summary_params_path}")
                except ImportError:
                    raise ImportError(
                        "Flax serialization is required to save summary network weights. "
                        "Please install Flax."
                    )

            else:
                logger.info(f"Summary network weights file already exists and overwrite is False: {summary_params_path}")
            
        
            # Use local save_summary_network_to_yaml function
        
        else:
            simulator_config["paths"]["summary_network_config"] = None
            simulator_config["paths"]["summary_network_weights"] = None

        # Save main simulator configuration
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
    yaml_dir = yaml_path.parent

    if not yaml_path.exists():
        raise FileNotFoundError(f"Simulator YAML file not found: {yaml_path}")

    # Load main configuration
    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {yaml_path}: {e}")

    # Validate configuration structure
    _validate_simulator_config_dict(config, yaml_path)

    # Load model
    if "model" not in config["paths"]:
        raise ValueError("Missing model path in simulator configuration")

    model_path = yaml_dir / config["paths"]["model"]
    model = _load_model_config(model_path)
    logger.info(f"Loaded model: {type(model).__name__}")

    # Load observed data if present
    observed_data = None
    if "observed_data" in config["paths"]:
        observed_data_path = yaml_dir / config["paths"]["observed_data"]
        if observed_data_path.exists():
            observed_data = np.load(observed_data_path)
            logger.info(f"Loaded observed data from: {observed_data_path}")
        else:
            logger.warning(f"Observed data file not found: {observed_data_path}")

    # Recreate simulator
    from .simulator import ABCSimulator

    simulator = ABCSimulator(
        model=model,
        observed_data=observed_data,
        epsilon=config["epsilon"],
        quantile_distance=config.get("quantile_distance"),
        config=config.get("simulator_config", {}),
    )


    if config["paths"].get("summary_network_config") and config["paths"].get("summary_network_weights"):
        summary_config_path = yaml_dir / config["paths"]["summary_network_config"]
        summary_params_path = yaml_dir / config["paths"]["summary_network_weights"]
        
        if summary_config_path.exists() and summary_params_path.exists():
            try: 
                from ..training.config import NNConfig
                from ..training.registry import create_network_from_nn_config
                import flax.serialization
                summary_config = NNConfig.load(summary_config_path)
                params_bytes = np.load(summary_params_path)["params"].item()
                summary_params = flax.serialization.from_bytes(None, params_bytes)
                logger.info(f"Loaded summary network config from: {summary_config_path}")
                logger.info(f"Loaded summary network weights from: {summary_params_path}")
            except ImportError:
                raise ImportError(
                    "Flax serialization is required to load summary network weights. "
                    "Please install Flax."
                )
                
            from .simulator import create_summary_stats_fn
            simulator._summary_config = summary_config
            simulator._summary_params = summary_params
            summary_network = create_network_from_nn_config(summary_config)
            summary_fn = create_summary_stats_fn(
                network=summary_network,
                params=summary_params,
                network_type=summary_config.network.network_type,
            )
            simulator.model.summary_stat_fn = summary_fn
            simulator.summary_stat_fn = summary_fn
            simulator.config["summary_stats_enabled"] = True
            simulator._initialize_sampler()
            logger.info("Summary network loaded and integrated into simulator")

   
    return simulator


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
