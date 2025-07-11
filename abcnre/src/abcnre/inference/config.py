"""
Configuration management for neural ratio estimation.

This module provides robust and flexible configuration classes for managing
NRE training and inference experiments.
"""

from typing import Dict, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml

@dataclass
class NetworkConfig:
    """Flexible configuration for any neural network architecture."""
    network_type: str
    network_args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'NetworkConfig':
        if 'network_type' not in config:
            raise ValueError("NetworkConfig must contain a 'network_type' key.")
        return cls(
            network_type=config['network_type'],
            network_args=config.get('network_args', {})
        )

@dataclass
class LRSchedulerConfig:
    """Configuration for the learning rate scheduler."""
    schedule_name: str = 'cosine'
    schedule_args: Dict[str, Any] = field(default_factory=dict) # Use a dict for flexibility

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'LRSchedulerConfig':
        return cls(**config)

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    learning_rate: float = 1e-3
    n_samples_per_epoch: int = 10240
    batch_size: int = 256
    num_epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    optimizer: str = 'adam'
    weight_decay: float = 0.0  
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    store_thetas: bool = True  
    num_thetas_to_store: int = 10000
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['lr_scheduler'] = self.lr_scheduler.to_dict()
        return d

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TrainingConfig':
        lr_scheduler_config = LRSchedulerConfig.from_dict(config.pop('lr_scheduler', {}))
        return cls(lr_scheduler=lr_scheduler_config, **config)

@dataclass
class ExperimentConfig:
    """Complete experiment configuration, linking network and training settings."""
    experiment_name: str
    network: NetworkConfig
    training: TrainingConfig
    random_seed: int = 42
    output_dir: str = './experiments'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_name': self.experiment_name,
            'random_seed': self.random_seed,
            'output_dir': self.output_dir,
            'network': self.network.to_dict(),
            'training': self.training.to_dict(),
        }
    
    # ... (les méthodes from_dict, save, load restent les mêmes) ...
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ExperimentConfig':
        """Create an ExperimentConfig instance from a dictionary."""
        return cls(
            experiment_name=config.get('experiment_name', 'nre_experiment'),
            random_seed=config.get('random_seed', 42),
            output_dir=config.get('output_dir', './experiments'),
            network=NetworkConfig.from_dict(config.get('network', {})),
            training=TrainingConfig.from_dict(config.get('training', {}))
        )
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save the experiment configuration to a YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load an experiment configuration from a YAML file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)

def get_experiment_config(name: str) -> ExperimentConfig:
    """
    Factory function to retrieve pre-defined experiment configurations.

    This provides a single entry point for creating standard, named
    experiment setups.

    Args:
        name: The name of the pre-defined configuration recipe.
              Examples: 'default_mlp_cosine', 'default_mlp_plateau'.

    Returns:
        A fully configured ExperimentConfig object.
    """
    if name == 'default_mlp_cosine':
        network_conf = NetworkConfig(
            network_type='MLPNetwork',
            network_args={'hidden_dims': [128, 64, 32], 'activation': 'relu'}
        )
        training_conf = TrainingConfig(
            learning_rate=1e-3,
            lr_scheduler=LRSchedulerConfig(schedule_name='cosine')
        )
        return ExperimentConfig(
            experiment_name='MLP with Cosine Decay',
            network=network_conf,
            training=training_conf
        )
        
    elif name == 'default_mlp_plateau':
        network_conf = NetworkConfig(
            network_type='MLPNetwork',
            network_args={'hidden_dims': [128, 64, 32], 'activation': 'relu'}
        )
        training_conf = TrainingConfig(
            learning_rate=1e-3,
            lr_scheduler=LRSchedulerConfig(
                schedule_name='reduce_on_plateau',
                schedule_args={'patience': 10, 'factor': 0.5}
            )
        )
        return ExperimentConfig(
            experiment_name='MLP with ReduceLROnPlateau',
            network=network_conf,
            training=training_conf
        )

    else:
        raise ValueError(f"Unknown configuration recipe name: '{name}'")