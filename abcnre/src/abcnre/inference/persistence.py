"""
Persistence utilities for neural ratio estimation.

This module provides utilities for saving and loading neural ratio estimators,
including model weights, configurations, and training history.
"""

import os
import json
import yaml
import pickle
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime
import numpy as np
import jax.numpy as jnp
from jax import random
import flax.serialization

from .estimator import NeuralRatioEstimator
from .networks.base import NetworkBase
from .networks.mlp import MLPNetwork, SimpleMLP, ResidualMLP
from .networks.deepset import DeepSetNetwork, CompactDeepSetNetwork
from .config import ExperimentConfig


class ModelRegistry:
    """Registry for network architectures to enable dynamic loading."""
    
    _networks = {
        'MLPNetwork': MLPNetwork,
        'SimpleMLP': SimpleMLP,
        'ResidualMLP': ResidualMLP,
        'DeepSetNetwork': DeepSetNetwork,
        'CompactDeepSetNetwork': CompactDeepSetNetwork
    }
    
    @classmethod
    def register_network(cls, name: str, network_class: type) -> None:
        """Register a new network architecture."""
        cls._networks[name] = network_class
    
    @classmethod
    def get_network(cls, name: str) -> type:
        """Get network class by name."""
        if name not in cls._networks:
            raise ValueError(f"Unknown network type: {name}. Available: {list(cls._networks.keys())}")
        return cls._networks[name]
    
    @classmethod
    def list_networks(cls) -> List[str]:
        """List all registered network types."""
        return list(cls._networks.keys())


class EstimatorCheckpoint:
    """
    Checkpoint manager for neural ratio estimators.
    
    Handles saving and loading of complete estimator state including
    model weights, configuration, and training history.
    """
    
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        estimator: NeuralRatioEstimator,
        checkpoint_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        include_optimizer_state: bool = True
    ) -> str:
        """
        Save complete estimator checkpoint.
        
        Args:
            estimator: Trained neural ratio estimator
            checkpoint_name: Name for the checkpoint
            metadata: Additional metadata to save
            include_optimizer_state: Whether to include optimizer state
            
        Returns:
            Path to saved checkpoint
        """
        if not estimator.is_trained:
            raise ValueError("Cannot save checkpoint for untrained estimator")
        
        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights and state
        weights_path = checkpoint_path / 'model_weights.pkl'
        with open(weights_path, 'wb') as f:
            if include_optimizer_state:
                state_dict = {
                    'params': estimator.state.params,
                    'opt_state': estimator.state.opt_state,
                    'step': estimator.state.step
                }
            else:
                state_dict = {
                    'params': estimator.state.params,
                    'step': estimator.state.step
                }
            pickle.dump(state_dict, f)
        
        # Save configuration
        config_path = checkpoint_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(estimator.config, f, default_flow_style=False)
        
        # Save training history
        history_path = checkpoint_path / 'training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, value in estimator.training_history.items():
                if isinstance(value, (list, np.ndarray)):
                    serializable_history[key] = [float(x) for x in value]
                else:
                    serializable_history[key] = value
            json.dump(serializable_history, f, indent=2)
        
        # Save metadata
        metadata_dict = {
            'timestamp': datetime.now().isoformat(),
            'estimator_class': type(estimator).__name__,
            'network_class': type(estimator.network).__name__,
            'checkpoint_name': checkpoint_name,
            'custom_metadata': metadata or {}
        }
        
        metadata_path = checkpoint_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        # Create checkpoint info file
        info_path = checkpoint_path / 'checkpoint_info.txt'
        with open(info_path, 'w') as f:
            f.write(f"Checkpoint: {checkpoint_name}\n")
            f.write(f"Created: {metadata_dict['timestamp']}\n")
            f.write(f"Estimator: {metadata_dict['estimator_class']}\n")
            f.write(f"Network: {metadata_dict['network_class']}\n")
            f.write(f"Training epochs: {len(estimator.training_history['train_loss'])}\n")
            f.write(f"Final train loss: {estimator.training_history['train_loss'][-1]:.4f}\n")
            f.write(f"Final val loss: {estimator.training_history['val_loss'][-1]:.4f}\n")
        
        print(f"Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_name: str,
        restore_optimizer_state: bool = True
    ) -> NeuralRatioEstimator:
        """
        Load estimator from checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint to load
            restore_optimizer_state: Whether to restore optimizer state
            
        Returns:
            Loaded neural ratio estimator
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load configuration
        config_path = checkpoint_path / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create network
        network_class = ModelRegistry.get_network(config['network_class'])
        network = network_class(**config['network_config'])
        
        # Create estimator
        estimator = NeuralRatioEstimator(
            network=network,
            learning_rate=config['learning_rate'],
            random_seed=config['random_seed']
        )
        
        # Load weights
        weights_path = checkpoint_path / 'model_weights.pkl'
        with open(weights_path, 'rb') as f:
            state_dict = pickle.load(f)
        
        # Initialize training state
        # We need to create a dummy input to initialize the state
        dummy_input = jnp.ones((1, 10))  # This would need to be adjusted based on actual input size
        estimator.initialize_training(dummy_input.shape)
        
        # Restore parameters
        estimator.state = estimator.state.replace(params=state_dict['params'])
        
        if restore_optimizer_state and 'opt_state' in state_dict:
            estimator.state = estimator.state.replace(opt_state=state_dict['opt_state'])
        
        if 'step' in state_dict:
            estimator.state = estimator.state.replace(step=state_dict['step'])
        
        # Load training history
        history_path = checkpoint_path / 'training_history.json'
        with open(history_path, 'r') as f:
            estimator.training_history = json.load(f)
        
        # Mark as trained
        estimator.is_trained = True
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        return estimator
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []
        
        for checkpoint_dir in self.checkpoint_dir.iterdir():
            if checkpoint_dir.is_dir():
                metadata_path = checkpoint_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    checkpoints.append({
                        'name': checkpoint_dir.name,
                        'path': str(checkpoint_dir),
                        **metadata
                    })
        
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_checkpoint(self, checkpoint_name: str) -> None:
        """Delete a checkpoint."""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if checkpoint_path.exists():
            import shutil
            shutil.rmtree(checkpoint_path)
            print(f"Checkpoint '{checkpoint_name}' deleted")
        else:
            print(f"Checkpoint '{checkpoint_name}' not found")


class ExperimentManager:
    """
    Manager for complete NRE experiments.
    
    Handles saving and loading of complete experiments including
    configurations, results, and trained models.
    """
    
    def __init__(self, experiments_dir: str = './experiments'):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = EstimatorCheckpoint()
    
    def save_experiment(
        self,
        experiment_name: str,
        estimator: NeuralRatioEstimator,
        config: ExperimentConfig,
        results: Dict[str, Any],
        abc_simulator_config_path: Optional[str] = None
    ) -> str:
        """
        Save complete experiment.
        
        Args:
            experiment_name: Name of the experiment
            estimator: Trained estimator
            config: Experiment configuration
            results: Training and validation results
            abc_simulator_config_path: Path to ABC simulator configuration
            
        Returns:
            Path to saved experiment
        """
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.experiments_dir / f"{experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = exp_dir / 'experiment_config.yaml'
        config.save(config_path)
        
        # Save estimator checkpoint
        checkpoint_name = f"{experiment_name}_{timestamp}_checkpoint"
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            estimator, checkpoint_name
        )
        
        # Save results
        results_path = exp_dir / 'results.json'
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Save ABC simulator configuration if provided
        if abc_simulator_config_path:
            abc_config_dest = exp_dir / 'abc_simulator_config.yaml'
            import shutil
            shutil.copy2(abc_simulator_config_path, abc_config_dest)
        
        # Create experiment summary
        summary = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'config_path': str(config_path),
            'checkpoint_path': checkpoint_path,
            'results_path': str(results_path),
            'abc_config_path': str(abc_config_dest) if abc_simulator_config_path else None,
            'network_type': type(estimator.network).__name__,
            'num_parameters': estimator.network.count_parameters(estimator.state.params),
            'final_performance': {
                'train_loss': results.get('final_train_loss', 0),
                'val_loss': results.get('final_val_loss', 0),
                'train_accuracy': results.get('final_train_accuracy', 0),
                'val_accuracy': results.get('final_val_accuracy', 0)
            }
        }
        
        summary_path = exp_dir / 'experiment_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Experiment saved to {exp_dir}")
        return str(exp_dir)
    
    def load_experiment(self, experiment_path: str) -> Dict[str, Any]:
        """
        Load complete experiment.
        
        Args:
            experiment_path: Path to experiment directory
            
        Returns:
            Dictionary containing loaded experiment components
        """
        exp_dir = Path(experiment_path)
        
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment not found: {exp_dir}")
        
        # Load experiment summary
        summary_path = exp_dir / 'experiment_summary.json'
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Load configuration
        config_path = exp_dir / 'experiment_config.yaml'
        config = ExperimentConfig.load(config_path)
        
        # Load estimator
        checkpoint_name = Path(summary['checkpoint_path']).name
        estimator = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        
        # Load results
        results_path = exp_dir / 'results.json'
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        return {
            'estimator': estimator,
            'config': config,
            'results': results,
            'summary': summary,
            'experiment_path': str(exp_dir)
        }
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all available experiments."""
        experiments = []
        
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                summary_path = exp_dir / 'experiment_summary.json'
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                    experiments.append({
                        'path': str(exp_dir),
                        **summary
                    })
        
        return sorted(experiments, key=lambda x: x['timestamp'], reverse=True)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return obj


def export_model_for_deployment(
    estimator: NeuralRatioEstimator,
    export_path: str,
    model_name: str = 'nre_model',
    include_metadata: bool = True
) -> None:
    """
    Export trained model for deployment.
    
    Args:
        estimator: Trained neural ratio estimator
        export_path: Path to export the model
        model_name: Name of the exported model
        include_metadata: Whether to include metadata
    """
    export_path = Path(export_path)
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Export model weights
    weights_path = export_path / f'{model_name}_weights.pkl'
    with open(weights_path, 'wb') as f:
        pickle.dump(estimator.state.params, f)
    
    # Export network configuration
    config_path = export_path / f'{model_name}_config.yaml'
    network_config = {
        'network_class': type(estimator.network).__name__,
        'network_config': estimator.network.get_config()
    }
    with open(config_path, 'w') as f:
        yaml.dump(network_config, f, default_flow_style=False)
    
    # Export deployment metadata
    if include_metadata:
        metadata = {
            'model_name': model_name,
            'export_timestamp': datetime.now().isoformat(),
            'network_type': type(estimator.network).__name__,
            'num_parameters': estimator.network.count_parameters(estimator.state.params),
            'training_epochs': len(estimator.training_history['train_loss']),
            'final_performance': {
                'train_loss': estimator.training_history['train_loss'][-1],
                'val_loss': estimator.training_history['val_loss'][-1],
                'train_accuracy': estimator.training_history['train_accuracy'][-1],
                'val_accuracy': estimator.training_history['val_accuracy'][-1]
            }
        }
        
        metadata_path = export_path / f'{model_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Model exported to {export_path}")


def load_model_for_deployment(model_path: str) -> NeuralRatioEstimator:
    """
    Load model for deployment.
    
    Args:
        model_path: Path to exported model
        
    Returns:
        Loaded neural ratio estimator
    """
    model_path = Path(model_path)
    
    # Find model files
    config_files = list(model_path.glob('*_config.yaml'))
    weight_files = list(model_path.glob('*_weights.pkl'))
    
    if not config_files or not weight_files:
        raise FileNotFoundError("Model configuration or weights not found")
    
    # Load configuration
    config_path = config_files[0]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create network
    network_class = ModelRegistry.get_network(config['network_class'])
    network = network_class(**config['network_config'])
    
    # Create estimator
    estimator = NeuralRatioEstimator(network=network)
    
    # Load weights
    weights_path = weight_files[0]
    with open(weights_path, 'rb') as f:
        params = pickle.load(f)
    
    # Initialize training state and load parameters
    dummy_input = jnp.ones((1, 10))  # Adjust based on actual input size
    estimator.initialize_training(dummy_input.shape)
    estimator.state = estimator.state.replace(params=params)
    estimator.is_trained = True
    
    return estimator

import yaml
import numpy as np
import flax.serialization
from pathlib import Path
from typing import Dict, Tuple

# --- Imports for type hinting ---
from ..simulation.simulator import ABCSimulator
from .config import TrainingConfig, NetworkConfig
from .estimator import NeuralRatioEstimator
from .networks.base import create_network_from_config


def save_classifier(
    estimator: 'NeuralRatioEstimator',
    simulator: 'ABCSimulator',
    output_dir: Path,
    filename_base: str
) -> Path:
    """Saves the classifier's artifacts into three separate files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / f"{filename_base}_classifier.yml"
    weights_path = output_dir / f"{filename_base}_weights.npy"
    phis_path = output_dir / f"{filename_base}_phis.npy"

    # Save weights
    params_bytes = flax.serialization.to_bytes(estimator.state.params)
    with open(weights_path, 'wb') as f:
        f.write(params_bytes)
    
    # Save phi samples
    saved_phis_path_str = None
    if estimator.accumulated_phi_samples:
        phi_array = np.array(estimator.accumulated_phi_samples)
        np.save(phis_path, phi_array)
        saved_phis_path_str = str(phis_path.resolve())

    # Gather metrics
    history = estimator.training_history
    final_metrics = {
        'final_val_loss': float(history['val_loss'][-1]) if history.get('val_loss') else None,
        'final_val_accuracy': float(history['val_accuracy'][-1]) if history.get('val_accuracy') else None,
        'epochs_trained': len(history.get('train_loss', [])),
        'total_sim_count': estimator.total_simulation_count
    }
    

    # Get the input dimensions from the simulator for robust loading
    phi_dim = 1  # Assuming phi is a scalar.
    summary_stat_dim = simulator.observed_summary_stats.shape[0]

    network_config_to_save = {
        'network_type': estimator.network.__class__.__name__,
        'network_args': estimator.network.get_config(),
        'input_dims': {
            'phi_dim': phi_dim,
            'summary_stat_dim': summary_stat_dim
        }
    }

    master_config = {
        'network_config': network_config_to_save,
        'training_config': estimator.training_config.to_dict(),
        'metrics': final_metrics,
        'paths': {
            'weights_path': str(weights_path.resolve()),
            'phis_path': saved_phis_path_str
        }
    }

    with open(config_path, 'w') as f:
        yaml.dump(master_config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Classifier saved. Master config: {config_path}")
    return config_path


def load_classifier(config_path: Path) -> Tuple[NeuralRatioEstimator, np.ndarray, Dict]:
    """Loads a classifier and its associated artifacts from a master YAML file."""
    with open(config_path, 'r') as f:
        master_config = yaml.safe_load(f)

    # Recreate estimator
    network_config = NetworkConfig.from_dict(master_config['network_config'])
    training_config = TrainingConfig.from_dict(master_config['training_config'])
    network = create_network_from_config(network_config.to_dict())
    estimator = NeuralRatioEstimator(network=network, training_config=training_config)

    # <<< FIX 2: Remove redundant hardcoded values >>>
    # Use the dimensions saved in the config file for robust initialization.
    input_dims = master_config['network_config']['input_dims']
    input_dim = input_dims['phi_dim'] + input_dims['summary_stat_dim']
    dummy_input_shape = (1, input_dim)
    estimator.initialize_training(dummy_input_shape)

    # Load weights
    weights_path = Path(master_config['paths']['weights_path'])
    with open(weights_path, 'rb') as f:
        params_bytes = f.read()
    
    loaded_params = flax.serialization.from_bytes(estimator.state.params, params_bytes)
    estimator.state = estimator.state.replace(params=loaded_params)
    estimator.is_trained = True
    
    # Load phi samples
    phis_path_str = master_config['paths']['phis_path']
    phi_samples = np.load(phis_path_str) if phis_path_str else None
    
    metrics = master_config.get('metrics', {})

    print(f"✅ Classifier loaded from {config_path}")
    return estimator, phi_samples, metrics