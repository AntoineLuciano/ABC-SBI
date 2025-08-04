from typing import Dict, Any, Union, Optional, TYPE_CHECKING
from pathlib import Path
import yaml
import logging
import numpy as np
from datetime import datetime
from ..training import NNConfig

from .estimator import NeuralRatioEstimator

# Configure logging
logger = logging.getLogger(__name__)


def save_estimator_to_yaml(
    estimator: Union[NNConfig, Dict[str, Any]],
    output_path: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """
    Save an estimator configuration to a YAML file.

    Args:
        estimator: Estimator configuration or NNConfig instance
        output_path: Path to save the YAML file
        overwrite: Whether to overwrite existing file

    Raises:
        ValueError: If the estimator is not valid
    """
    output_path = Path(output_path)

    # Check if we should skip saving
    if _should_skip_estimator_save(output_path, overwrite):
        return

    # Create output directory
    _ensure_estimator_output_directory(output_path)

    # Save classifier configuration
    _save_classifier_config(estimator, output_path, overwrite)

    # Save simulator configuration
    _save_simulator_config(estimator, output_path, overwrite)

    # Create and save main estimator configuration
    estimator_config = _create_estimator_config(estimator, output_path)

    # Save trained parameters if available
    if estimator.is_trained:
        _save_trained_parameters(estimator, estimator_config, output_path, overwrite)
        _save_stored_phis(estimator, estimator_config, output_path, overwrite)

    # Save main estimator configuration file
    _save_main_estimator_config(estimator_config, output_path)


def _should_skip_estimator_save(output_path: Path, overwrite: bool) -> bool:
    """Check if saving should be skipped due to existing file."""
    if output_path.exists() and not overwrite:
        logger.info(f"Output path already exists and overwrite is False: {output_path}")
        return True
    return False


def _ensure_estimator_output_directory(output_path: Path) -> None:
    """Create output directory if it doesn't exist."""
    output_path.parent.mkdir(parents=True, exist_ok=True)


def _save_classifier_config(estimator, output_path: Path, overwrite: bool) -> None:
    """Save classifier configuration to separate file."""
    classifier_config = estimator.nn_config
    classif_config_filename = f"{output_path.stem}_classif_config.yaml"
    classif_config_path = output_path.parent / classif_config_filename
    classifier_config.save(classif_config_path, overwrite=overwrite)
    logger.info(f"Saved classifier config to: {classif_config_path}")


def _save_simulator_config(estimator, output_path: Path, overwrite: bool) -> None:
    """Save simulator configuration to separate file."""
    simulator = estimator.simulator
    simulator_filename = f"{output_path.stem}_simulator.yaml"
    simulator_path = output_path.parent / simulator_filename

    from ..simulation.io import save_simulator_to_yaml

    save_simulator_to_yaml(simulator, simulator_path, overwrite=overwrite)
    logger.info(f"Saved simulator config to: {simulator_path}")


def _create_estimator_config(estimator, output_path: Path) -> Dict[str, Any]:
    """Create the main estimator configuration dictionary."""
    classif_config_filename = f"{output_path.stem}_classif_config.yaml"
    simulator_filename = f"{output_path.stem}_simulator.yaml"

    config = {
        "summary_as_input": estimator.summary_as_input,
        "is_trained": estimator.is_trained,
        "paths": {
            "classifier_config": classif_config_filename,
            "simulator_config": simulator_filename,
        },
        "metadata": {
            "created_date": datetime.now().isoformat(),
            "abcnre_version": "0.1.0",
            "estimator_type": type(estimator).__name__,
            "sampler_id": getattr(estimator.simulator, "sampler_id", None),
        },
    }

    return config


def _save_trained_parameters(
    estimator, estimator_config: Dict[str, Any], output_path: Path, overwrite: bool
) -> None:
    """Save trained parameters using Flax serialization."""
    params = estimator.trained_params
    params_filename = f"{output_path.stem}_classif_params.npz"
    params_path = output_path.parent / params_filename

    if not params_path.exists() or overwrite:
        try:
            import flax.serialization

            params_bytes = flax.serialization.to_bytes(params)
            np.savez_compressed(params_path, params=params_bytes)
            estimator_config["paths"]["classifier_params"] = params_filename
            logger.info(f"Saved classifier params to: {params_path}")
        except ImportError:
            logger.error("Flax serialization not available. Cannot save params.")
    else:
        logger.info(
            f"Classifier params file already exists and overwrite is False: {params_path}"
        )


def _save_stored_phis(
    estimator, estimator_config: Dict[str, Any], output_path: Path, overwrite: bool
) -> None:
    """Save stored phi samples if available."""
    if hasattr(estimator, "stored_phis") and estimator.stored_phis is not None:
        stored_phis_filename = f"{output_path.stem}_stored_phis.npy"
        stored_phis_path = output_path.parent / stored_phis_filename

        if not stored_phis_path.exists() or overwrite:
            np.save(stored_phis_path, estimator.stored_phis)
            estimator_config["paths"]["stored_phis"] = stored_phis_filename
            logger.info(f"Saved stored phis to: {stored_phis_path}")
        else:
            logger.info(
                f"Stored phis file already exists and overwrite is False: {stored_phis_path}"
            )


def _save_main_estimator_config(
    estimator_config: Dict[str, Any], output_path: Path
) -> None:
    """Save the main estimator configuration to YAML file."""
    estimator_config_filename = f"{output_path.stem}_estimator_config.yaml"
    estimator_config_path = output_path.parent / estimator_config_filename

    with open(output_path, "w") as f:
        yaml.dump(estimator_config, f, default_flow_style=False)
    logger.info(f"Saved estimator to: {output_path}")


def load_estimator_from_yaml(input_path: Union[str, Path]) -> NNConfig:
    """
    Load an estimator configuration from a YAML file.

    Args:
        input_path: Path to the YAML file

    Returns:
        NNConfig instance with loaded configuration

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the configuration is invalid
    """
    input_path = Path(input_path)

    # Load and validate configuration
    config = _load_estimator_config(input_path)

    # Load neural network configuration
    nn_config = _load_nn_config(config, input_path.parent)

    # Load simulator
    simulator = _load_simulator_for_estimator(config, input_path.parent)

    # Create estimator instance
    estimator = _create_estimator_instance(config, nn_config, simulator)

    # Load trained parameters if available
    if config.get("is_trained", False):
        _load_trained_components(config, input_path.parent, estimator)

    return estimator


def _load_estimator_config(input_path: Path) -> Dict[str, Any]:
    """Load estimator configuration from YAML file."""
    if not input_path.exists():
        raise FileNotFoundError(f"File {input_path} does not exist")

    with open(input_path, "r") as f:
        return yaml.safe_load(f)


def _load_nn_config(config: Dict[str, Any], input_dir: Path):
    """Load neural network configuration."""
    nn_config_path = input_dir / config["paths"]["classifier_config"]
    if not nn_config_path.exists():
        raise FileNotFoundError(f"NNConfig file {nn_config_path} does not exist")

    return NNConfig.load(nn_config_path)


def _load_simulator_for_estimator(config: Dict[str, Any], input_dir: Path):
    """Load simulator configuration for estimator."""
    from ..simulation.io import load_simulator_from_yaml

    simulator_path = input_dir / config["paths"]["simulator_config"]
    if not simulator_path.exists():
        raise FileNotFoundError(
            f"Simulator config file {simulator_path} does not exist"
        )

    return load_simulator_from_yaml(simulator_path)


def _create_estimator_instance(config: Dict[str, Any], nn_config, simulator):
    """Create NeuralRatioEstimator instance."""
    return NeuralRatioEstimator(
        nn_config=nn_config,
        simulator=simulator,
        summary_as_input=config.get("summary_as_input", False),
    )


def _load_trained_components(
    config: Dict[str, Any], input_dir: Path, estimator
) -> None:
    """Load trained parameters and stored phis if available."""
    estimator.is_trained = True

    # Load classifier parameters
    if "classifier_params" in config["paths"]:
        _load_classifier_params(config, input_dir, estimator)

    # Load stored phis
    if "stored_phis" in config["paths"]:
        _load_stored_phis(config, input_dir, estimator)

    # Set defaults if not trained
    if not hasattr(estimator, "trained_params") or estimator.trained_params is None:
        estimator.is_trained = False
        estimator.trained_params = None
        estimator.log_ratio_fn = None
        estimator.stored_phis = None


def _load_classifier_params(config: Dict[str, Any], input_dir: Path, estimator) -> None:
    """Load classifier parameters and create log ratio function."""
    params_path = input_dir / config["paths"]["classifier_params"]
    if not params_path.exists():
        logger.warning(f"Classifier params file {params_path} does not exist")
        return

    try:
        import flax.serialization
        from .estimator import create_log_ratio_function
        from ..training import create_network_from_nn_config

        # Load parameters
        params_bytes = np.load(params_path)["params"].item()
        params = flax.serialization.from_bytes(None, params_bytes)
        estimator.trained_params = params

        # Create log ratio function
        network = create_network_from_nn_config(estimator.nn_config)
        estimator.log_ratio_fn = create_log_ratio_function(
            network=network,
            params=estimator.trained_params,
            network_type=estimator.nn_config.network.network_type,
            summary_as_input=estimator.summary_as_input,
        )

        logger.info(f"Loaded trained params and log ratio function from: {params_path}")

    except (ImportError, KeyError) as e:
        logger.error(f"Error loading params from {params_path}: {e}")
        raise ValueError(f"Error loading classifier parameters: {e}")


def _load_stored_phis(config: Dict[str, Any], input_dir: Path, estimator) -> None:
    """Load stored phi samples."""
    stored_phis_path = input_dir / config["paths"]["stored_phis"]
    if stored_phis_path.exists():
        estimator.stored_phis = np.load(stored_phis_path)
        logger.info(f"Loaded stored phis from: {stored_phis_path}")
    else:
        logger.warning(f"Stored phis file {stored_phis_path} does not exist")
