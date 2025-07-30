from typing import Dict, Any, Union, Optional, TYPE_CHECKING
from pathlib import Path
import yaml
import logging
import numpy as np
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
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not overwrite:
        logger.info(f"Output path already exists and overwrite is False: {output_path}")
        
    else:
    
        classifier_config = estimator.nn_config
        classif_config_filename = f"{output_path.stem}_classif_config.yaml"
        classif_config_path = output_dir / classif_config_filename
        classifier_config.save(classif_config_path, overwrite=overwrite)
        logger.info(f"Saved classifier config to: {classif_config_path}")
        
        simulator = estimator.simulator
        simulator_filename = f"{output_path.stem}_simulator.yaml"
        simulator_path = output_dir / simulator_filename
        
        from ..simulation.io import save_simulator_to_yaml
        save_simulator_to_yaml(
            simulator,
            simulator_path,
            overwrite=overwrite,
        )
        
        logger.info(f"Saved simulator config to: {simulator_path}")

        estimator_config = {"summary_as_input": estimator.summary_as_input,
                            "is_trained": estimator.is_trained,
                            "paths": {
                                "classifier_config": str(classif_config_filename),
                                "simulator_config": str(simulator_filename),
                            }
        }


        if estimator.is_trained:
            params = estimator.trained_params
            params_filename = f"{output_path.stem}_classif_params.npz"
            params_path = output_dir / params_filename
            if not params_path.exists() or overwrite:
                try: 
                    import flax.serialization
                    params_bytes = flax.serialization.to_bytes(params)
                    np.savez_compressed(params_path, params=params_bytes)
                    estimator_config["paths"]["classifier_params"] = str(params_filename)
                    logger.info(f"Saved classifier params to: {params_path}")
                except ImportError:
                    logger.error("Flax serialization not available. Cannot save params.")
                    
            else:
                logger.info(f"Classifier params file already exists and overwrite is False: {params_path}")
                
            print( hasattr(estimator, 'stored_phis'), "stored_phis attribute not found in estimator")
            
            if hasattr(estimator, 'stored_phis') and estimator.stored_phis is not None:
                stored_phis_filename = f"{output_path.stem}_stored_phis.npy"
                stored_phis_path = output_dir / stored_phis_filename
                if not stored_phis_path.exists() or overwrite:
                    np.save(stored_phis_path, estimator.stored_phis)
                    estimator_config["paths"]["stored_phis"] = str(stored_phis_filename)
                    logger.info(f"Saved stored phis to: {stored_phis_path}")
                else:
                    logger.info(f"Stored phis file already exists and overwrite is False: {stored_phis_path}")  
        
        
        estimator_config_filename = f"{output_path.stem}_estimator_config.yaml"
        estimator_config_path = output_dir / estimator_config_filename
        if overwrite or not output_path.exists():
            with open(output_path, 'w') as f:
                yaml.dump(estimator_config, f, default_flow_style=False)
            logger.info(f"Saved estimator to: {output_path}")
        else:
            raise FileExistsError(f"File {output_path} already exists. Use overwrite=True to replace it.")
    
    
def load_estimator_from_yaml(
    input_path: Union[str, Path],
) -> NNConfig:
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
    if not input_path.exists():
        raise FileNotFoundError(f"File {input_path} does not exist")

    with open(input_path, 'r') as f:
        config = yaml.safe_load(f)
    nn_config_path = input_path.parent / config['paths']['classifier_config']
    if not nn_config_path.exists():
        raise FileNotFoundError(f"NNConfig file {nn_config_path} does not exist")
    
    nn_config = NNConfig.load(nn_config_path)

    from ..simulation.io import load_simulator_from_yaml
    simulator_path = input_path.parent / config['paths']['simulator_config']
    if not simulator_path.exists():
        raise FileNotFoundError(f"Simulator config file {simulator_path} does not exist")
    simulator = load_simulator_from_yaml(simulator_path)

    estimator =  NeuralRatioEstimator(
        nn_config=nn_config,
        simulator=simulator,
        summary_as_input=config.get("summary_as_input", False),
    )
    if config.get("is_trained", False):
        estimator.is_trained = True
        
        if "classifier_params" in config["paths"]:
            params_path = input_path.parent / config["paths"]["classifier_params"]
            if params_path.exists():
                try:
                    import flax.serialization
                except ImportError:
                    logger.error("Flax serialization not available. Cannot load params.")
                try:
                    from .estimator import create_log_ratio_function
                except ImportError:
                    logger.error("create_log_ratio_function not available. Cannot load params.")
                try:
                    from ..training import create_network_from_nn_config
                except ImportError:
                    logger.error("create_network_from_nn_config not available. Cannot load params.")
                try:
                    params_bytes = np.load(params_path)["params"].item()
                    params = flax.serialization.from_bytes(None, params_bytes)
                    estimator.trained_params = params
                    logger.info(f"Loaded trained params from: {params_path}")
                except KeyError:
                    logger.error(f"Invalid params file format: {params_path}")
                    raise ValueError(f"Invalid params file format: {params_path}")
                except Exception as e:
                    logger.error(f"Error loading params from {params_path}: {e}")
                try:
                    
                    network = create_network_from_nn_config(nn_config)
                    estimator.log_ratio_fn = create_log_ratio_function(
                        network=network,
                        params=estimator.trained_params,
                        network_type=nn_config.network.network_type,
                        summary_as_input=estimator.summary_as_input,
                    )
                    logger.info(f"Loaded trained params from: {params_path}")
                    logger.info(f"Loaded log ratio function from: {params_path}")
                    logger.info(f"Loaded trained config from: {params_path}")

                except Exception as e:
                    logger.error(f"Error creating log ratio function: {e}")
                    raise ValueError(f"Error creating log ratio function: {e}")
                
            else:
                logger.warning(f"Classifier params file {params_path} does not exist")
        if "stored_phis" in config["paths"]:
            stored_phis_path = input_path.parent / config["paths"]["stored_phis"]
            if stored_phis_path.exists():
                estimator.stored_phis = np.load(stored_phis_path)
                logger.info(f"Loaded stored phis from: {stored_phis_path}")
            else:
                logger.warning(f"Stored phis file {stored_phis_path} does not exist")
    else:
        estimator.is_trained = False
        estimator.trained_params = None
        estimator.log_ratio_fn = None
        estimator.stored_phis = None
    return estimator
        