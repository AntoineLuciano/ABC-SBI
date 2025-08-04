"""
Configuration management for neural ratio estimation.

This module provides robust and flexible configuration classes for managing
NRE training and inference experiments with modular stopping rules and
lr scheduler configurations.
"""

from typing import Dict, Any, Union, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


def _deep_merge_configs(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged configuration with override values taking precedence
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = _deep_merge_configs(result[key], value)
        else:
            # Override the value
            result[key] = value

    return result


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping based on validation metrics."""

    enabled: bool = True
    monitor: str = "validation_loss"
    patience: int = 10
    min_delta: float = 1e-4
    restore_best_weights: bool = True
    mode: str = "min"  # "min" for loss, "max" for accuracy


@dataclass
class LRStoppingConfig:
    """Configuration for stopping when learning rate becomes too small."""

    enabled: bool = False
    min_lr: float = 1e-8


@dataclass
class ConvergenceStoppingConfig:
    """Configuration for stopping when loss converges."""

    enabled: bool = False
    tolerance: float = 1e-6
    patience: int = 5


@dataclass
class PlateauStoppingConfig:
    """Configuration for stopping when loss plateaus."""

    enabled: bool = False
    patience: int = 20
    threshold: float = 1e-5


@dataclass
class TimeStoppingConfig:
    """Configuration for time-based stopping."""

    enabled: bool = False
    max_time_hours: Optional[float] = None


@dataclass
class SampleStoppingConfig:
    """Configuration for stopping when maximum number of samples is reached."""

    enabled: bool = False
    max_samples: Optional[int] = None


@dataclass
class SimulationStoppingConfig:
    """Configuration for stopping when maximum number of simulations is reached."""

    enabled: bool = False
    max_simulations: Optional[int] = None


@dataclass
class StoppingRulesConfig:
    """Complete stopping rules configuration."""

    max_epochs: int = 1000
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    lr_stopping: LRStoppingConfig = field(default_factory=LRStoppingConfig)
    convergence_stopping: ConvergenceStoppingConfig = field(
        default_factory=ConvergenceStoppingConfig
    )
    plateau_stopping: PlateauStoppingConfig = field(
        default_factory=PlateauStoppingConfig
    )
    time_stopping: TimeStoppingConfig = field(default_factory=TimeStoppingConfig)
    sample_stopping: SampleStoppingConfig = field(default_factory=SampleStoppingConfig)
    simulation_stopping: SimulationStoppingConfig = field(
        default_factory=SimulationStoppingConfig
    )

    def __post_init__(self):
        """Validate stopping rules configuration."""
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")

        if self.early_stopping.enabled and self.early_stopping.patience <= 0:
            raise ValueError("early_stopping patience must be positive")

        if self.lr_stopping.enabled and self.lr_stopping.min_lr <= 0:
            raise ValueError("lr_stopping min_lr must be positive")

        if (
            self.sample_stopping.enabled
            and self.sample_stopping.max_samples is not None
            and self.sample_stopping.max_samples <= 0
        ):
            raise ValueError("sample_stopping max_samples must be positive")

        if (
            self.simulation_stopping.enabled
            and self.simulation_stopping.max_simulations is not None
            and self.simulation_stopping.max_simulations <= 0
        ):
            raise ValueError("simulation_stopping max_simulations must be positive")

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "StoppingRulesConfig":
        """Create StoppingRulesConfig from dictionary."""

        # Extract and convert nested configs
        early_stopping_dict = config.get("early_stopping", {})
        early_stopping = EarlyStoppingConfig(**early_stopping_dict)

        lr_stopping_dict = config.get("lr_stopping", {})
        lr_stopping = LRStoppingConfig(**lr_stopping_dict)

        convergence_stopping_dict = config.get("convergence_stopping", {})
        convergence_stopping = ConvergenceStoppingConfig(**convergence_stopping_dict)

        plateau_stopping_dict = config.get("plateau_stopping", {})
        plateau_stopping = PlateauStoppingConfig(**plateau_stopping_dict)

        time_stopping_dict = config.get("time_stopping", {})
        time_stopping = TimeStoppingConfig(**time_stopping_dict)

        sample_stopping_dict = config.get("sample_stopping", {})
        sample_stopping = SampleStoppingConfig(**sample_stopping_dict)

        simulation_stopping_dict = config.get("simulation_stopping", {})
        simulation_stopping = SimulationStoppingConfig(**simulation_stopping_dict)

        return cls(
            max_epochs=config.get("max_epochs", 1000),
            early_stopping=early_stopping,
            lr_stopping=lr_stopping,
            convergence_stopping=convergence_stopping,
            plateau_stopping=plateau_stopping,
            time_stopping=time_stopping,
            sample_stopping=sample_stopping,
            simulation_stopping=simulation_stopping,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def get_active_stopping_criteria(self) -> list[str]:
        """Get list of active stopping criteria."""
        active = ["max_epochs"]  # Always active

        if self.early_stopping.enabled:
            active.append("early_stopping")
        if self.lr_stopping.enabled:
            active.append("lr_stopping")
        if self.convergence_stopping.enabled:
            active.append("convergence_stopping")
        if self.plateau_stopping.enabled:
            active.append("plateau_stopping")
        if self.time_stopping.enabled:
            active.append("time_stopping")
        if self.sample_stopping.enabled:
            active.append("sample_stopping")
        if self.simulation_stopping.enabled:
            active.append("simulation_stopping")

        return active


@dataclass
class NetworkConfig:
    """Flexible configuration for any neural network architecture."""

    network_type: str
    network_args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "NetworkConfig":
        if "network_type" not in config:
            raise ValueError("NetworkConfig must contain a 'network_type' key.")
        return cls(
            network_type=config["network_type"],
            network_args=config.get("network_args", {}),
        )


@dataclass
class LRSchedulerConfig:
    """Configuration for the learning rate scheduler."""

    schedule_name: str = "cosine"
    schedule_args: Dict[str, Any] = field(
        default_factory=dict
    )  # Use a dict for flexibility

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LRSchedulerConfig":
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
    optimizer: str = "adam"
    weight_decay: float = 0.0
    loss_function: str = (
        "default"  # "default", "bce", "focal", "label_smoothing", "mse", "huber", "mae"
    )
    loss_args: Dict[str, Any] = field(
        default_factory=dict
    )  # Arguments spécifiques à la loss
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    store_thetas: bool = True
    num_thetas_to_store: int = 10000
    stopping_rules: Optional[Dict[str, Any]] = None
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["lr_scheduler"] = self.lr_scheduler.to_dict()
        return d

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrainingConfig":
        lr_scheduler_config = LRSchedulerConfig.from_dict(
            config.pop("lr_scheduler", {})
        )
        stopping_rules = config.pop("stopping_rules", None)
        return cls(
            lr_scheduler=lr_scheduler_config, stopping_rules=stopping_rules, **config
        )


def get_predefined_network_config(
    network_name: str, size: str = "default"
) -> Dict[str, Any]:
    """Load a predefined network configuration with size variants."""
    config_dir = (
        Path(__file__).parent.parent.parent.parent / "examples" / "configs" / "networks"
    )
    config_file = config_dir / f"{network_name}.yaml"

    if not config_file.exists():
        available = [f.stem for f in config_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"Network config file '{network_name}.yaml' not found. Available: {available}"
        )

    with open(config_file, "r") as f:
        all_configs = yaml.safe_load(f)

    # Check if the file contains size variants
    if isinstance(all_configs, dict) and "variants" in all_configs:
        # New format with variants
        if size not in all_configs["variants"]:
            available_sizes = list(all_configs["variants"].keys())
            raise ValueError(
                f"Size variant '{size}' not found for network '{network_name}'. "
                f"Available sizes: {available_sizes}"
            )

        # Merge base config with size-specific config
        base_config = all_configs.get("base", {})
        size_config = all_configs["variants"][size]

        # Deep merge the configurations
        merged_config = _deep_merge_configs(base_config, size_config)
        return merged_config

    else:
        # Legacy format - single configuration or size-specific keys
        if size in all_configs:
            return all_configs[size]
        elif size == "default" and len(all_configs) == 1:
            # If only one config and asking for default, return it
            return list(all_configs.values())[0]
        else:
            available_sizes = list(all_configs.keys())
            raise ValueError(
                f"Size variant '{size}' not found for network '{network_name}'. "
                f"Available sizes: {available_sizes}"
            )


def get_predefined_training_config(size: str = "default") -> Dict[str, Any]:
    """Load a predefined training configuration with size variants."""
    config_dir = (
        Path(__file__).parent.parent.parent.parent / "examples" / "configs" / "training"
    )
    config_file = config_dir / "training.yaml"

    if not config_file.exists():
        available = [f.stem for f in config_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"Training config file 'training.yaml' not found. Available: {available}"
        )

    with open(config_file, "r") as f:
        all_configs = yaml.safe_load(f)

    # Check if the file contains size variants
    if isinstance(all_configs, dict) and "variants" in all_configs:
        # New format with variants
        if size not in all_configs["variants"]:
            available_sizes = list(all_configs["variants"].keys())
            raise ValueError(
                f"Size variant '{size}' not found for training config. "
                f"Available sizes: {available_sizes}"
            )

        # Merge base config with size-specific config
        base_config = all_configs.get("base", {})
        size_config = all_configs["variants"][size]

        # Deep merge the configurations
        merged_config = _deep_merge_configs(base_config, size_config)

        # Load lr_scheduler separately if specified
        if "lr_scheduler_name" in merged_config:
            scheduler_name = merged_config.pop("lr_scheduler_name")
            scheduler_variant = merged_config.pop("lr_scheduler_variant", "default")

            try:
                scheduler_config = get_predefined_lr_scheduler_config(
                    scheduler_name, scheduler_variant
                )
                merged_config["lr_scheduler"] = scheduler_config
            except Exception as e:
                logger.warning(
                    f"Failed to load lr_scheduler '{scheduler_name}_{scheduler_variant}': {e}"
                )
                # Fallback to default
                merged_config["lr_scheduler"] = {
                    "schedule_name": "constant",
                    "schedule_args": {},
                }

        return merged_config

    else:
        # Legacy format - size-specific keys
        if size in all_configs:
            return all_configs[size]
        elif size == "default" and len(all_configs) == 1:
            # If only one config and asking for default, return it
            return list(all_configs.values())[0]
        else:
            available_sizes = list(all_configs.keys())
            raise ValueError(
                f"Size variant '{size}' not found for training config. "
                f"Available sizes: {available_sizes}"
            )


def get_predefined_loss_config(
    loss_variant: str = "default_regressor",
) -> Dict[str, Any]:
    """Load a predefined loss function configuration with variants."""
    config_dir = (
        Path(__file__).parent.parent.parent.parent / "examples" / "configs" / "training"
    )
    config_file = config_dir / "loss.yaml"

    if not config_file.exists():
        available = [f.stem for f in config_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"Loss config file 'loss.yaml' not found. Available: {available}"
        )

    with open(config_file, "r") as f:
        all_configs = yaml.safe_load(f)

    # Check if the file contains variants
    if isinstance(all_configs, dict) and "variants" in all_configs:
        # New format with variants
        if loss_variant not in all_configs["variants"]:
            available_variants = list(all_configs["variants"].keys())
            raise ValueError(
                f"Loss variant '{loss_variant}' not found. "
                f"Available variants: {available_variants}"
            )

        # Merge base config with variant-specific config
        base_config = all_configs.get("base", {})
        variant_config = all_configs["variants"][loss_variant]

        # Deep merge the configurations
        merged_config = _deep_merge_configs(base_config, variant_config)
        return merged_config

    else:
        # Legacy format - direct variant keys
        if loss_variant in all_configs:
            return all_configs[loss_variant]
        else:
            available_variants = list(all_configs.keys())
            raise ValueError(
                f"Loss variant '{loss_variant}' not found. "
                f"Available variants: {available_variants}"
            )


def get_predefined_lr_scheduler_config(
    scheduler_name: str, variant: str = "default"
) -> Dict[str, Any]:
    """Load a predefined learning rate scheduler configuration."""
    config_dir = (
        Path(__file__).parent.parent.parent.parent
        / "examples"
        / "configs"
        / "lr_schedulers"
    )
    config_file = config_dir / f"{scheduler_name}.yaml"

    if not config_file.exists():
        available = [f.stem for f in config_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"LR scheduler config file '{scheduler_name}.yaml' not found. Available: {available}"
        )

    with open(config_file, "r") as f:
        all_configs = yaml.safe_load(f)

    # Check if the file contains variants
    if isinstance(all_configs, dict) and "variants" in all_configs:
        # New format with variants
        if variant not in all_configs["variants"]:
            available_variants = list(all_configs["variants"].keys())
            raise ValueError(
                f"Variant '{variant}' not found for scheduler '{scheduler_name}'. "
                f"Available variants: {available_variants}"
            )

        # Merge base config with variant-specific config
        base_config = all_configs.get("base", {})
        variant_config = all_configs["variants"][variant]

        # Deep merge the configurations
        merged_config = _deep_merge_configs(base_config, variant_config)
        return merged_config

    else:
        # Legacy format - variant-specific keys
        if variant in all_configs:
            return all_configs[variant]
        elif variant == "default" and len(all_configs) == 1:
            # If only one config and asking for default, return it
            return list(all_configs.values())[0]
        else:
            available_variants = list(all_configs.keys())
            raise ValueError(
                f"Variant '{variant}' not found for scheduler '{scheduler_name}'. "
                f"Available variants: {available_variants}"
            )


def get_predefined_stopping_rules_config(variant: str = "balanced") -> Dict[str, Any]:
    """Load a predefined stopping rules configuration."""
    config_dir = (
        Path(__file__).parent.parent.parent.parent
        / "examples"
        / "configs"
        / "stopping_rules"
    )
    config_file = config_dir / "stopping_rules.yaml"

    if not config_file.exists():
        logger.warning(f"Stopping rules config file not found: {config_file}")
        # Return default config
        return StoppingRulesConfig().to_dict()

    with open(config_file, "r") as f:
        all_configs = yaml.safe_load(f)

    # Check if the file contains variants
    if isinstance(all_configs, dict) and "variants" in all_configs:
        if variant not in all_configs["variants"]:
            available_variants = list(all_configs["variants"].keys())
            logger.warning(
                f"Variant '{variant}' not found for stopping rules. "
                f"Available variants: {available_variants}. Using 'balanced'."
            )
            variant = (
                "balanced"
                if "balanced" in available_variants
                else list(available_variants)[0]
            )

        # Merge base config with variant-specific config
        base_config = all_configs.get("base", {})
        variant_config = all_configs["variants"][variant]

        merged_config = _deep_merge_configs(base_config, variant_config)
        return merged_config

    else:
        # Legacy format
        if variant in all_configs:
            return all_configs[variant]
        else:
            logger.warning(
                f"Variant '{variant}' not found. Using default stopping rules."
            )
            return StoppingRulesConfig().to_dict()


@dataclass
class NNConfig:
    """Complete neural network configuration, linking network and training settings."""

    experiment_name: str = "nn_experiment"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "./experiments"
    task_type: str = "classifier"  # "classifier" or "regressor"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
            "task_type": self.task_type,
            "network": self.network.to_dict(),
            "training": self.training.to_dict(),
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "NNConfig":
        """Create an NNConfig instance from a dictionary."""
        return cls(
            experiment_name=config.get("experiment_name", "nn_experiment"),
            output_dir=config.get("output_dir", "./experiments"),
            task_type=config.get("task_type", "classifier"),
            network=NetworkConfig.from_dict(config.get("network", {})),
            training=TrainingConfig.from_dict(config.get("training", {})),
        )

    def save(self, filepath: Union[str, Path], overwrite: bool = False) -> None:
        """Save the experiment configuration to a YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.exists() and not overwrite:
            logger.info(
                f"Configuration file already exists and overwrite is False: {filepath}"
            )
        else:
            # Save configuration to YAML
            with open(filepath, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
            logger.info(f"Configuration saved to {filepath}")

    def to_str(self) -> str:
        """Get a string representation of the configuration."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "NNConfig":
        """Load an NN configuration from a YAML file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {filepath}")
        return cls.from_dict(config_dict)

    def is_classifier(self) -> bool:
        """Check if this is a classifier task."""
        return self.task_type == "classifier"

    def is_regressor(self) -> bool:
        """Check if this is a regressor task."""
        return self.task_type == "regressor"

    def get_default_loss_function(self) -> str:
        """Get the default loss function for this task type."""
        if self.is_classifier():
            return "binary_crossentropy"
        elif self.is_regressor():
            return "mse"
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")


def get_nn_config(
    network_name: str = "mlp",
    network_size: str = "default",
    training_size: str = "default",
    task_type: str = "classifier",
    lr_scheduler_name: Optional[str] = None,
    lr_scheduler_variant: str = "default",
    stopping_rules_variant: str = "balanced",
    experiment_name: Optional[str] = None,
    loss_function: Optional[str] = None,
    loss_args: Optional[Dict[str, Any]] = None,
    loss_variant: Optional[str] = None,
) -> NNConfig:
    """Get a complete NN configuration based on predefined templates.

    Args:
        network_name: Name of the network type (e.g., "mlp", "deepset")
        network_size: Size variant of the network (e.g., "small", "large", "default")
        training_size: Size variant of training config (e.g., "fast", "extended", "default")
        task_type: Type of task ("classifier" or "regressor")
        lr_scheduler_name: Optional override for lr scheduler name
        lr_scheduler_variant: Variant of the lr scheduler (default: "default")
        stopping_rules_variant: Variant of stopping rules (default: "balanced")
        experiment_name: Optional name for the experiment
        loss_function: Optional override for loss function (e.g., "focal", "pinball", "huber")
        loss_args: Optional arguments for the loss function (e.g., {"tau": 0.5} for pinball)
        loss_variant: Optional predefined loss variant (e.g., "pinball_median", "focal", "huber")

    Returns:
        A fully configured NNConfig object.

    Note:
        - If loss_variant is provided, it takes precedence over loss_function/loss_args
        - loss_variant loads complete configurations from loss.yaml
        - loss_function/loss_args provide direct parameter override
    """
    try:
        # Load network configuration
        network_config_dict = get_predefined_network_config(network_name, network_size)
        network_config = NetworkConfig.from_dict(network_config_dict)

        # Load training configuration
        training_config_dict = get_predefined_training_config(training_size)

        # Override lr_scheduler if specified
        if lr_scheduler_name:
            # Special handling for constant scheduler - ignore variant and use only learning_rate
            if lr_scheduler_name == "constant":
                scheduler_config = {"schedule_name": "constant", "schedule_args": {}}
                training_config_dict["lr_scheduler"] = scheduler_config
                logger.info(
                    f"Using constant lr_scheduler with base learning_rate from training config"
                )
            else:
                try:
                    scheduler_config = get_predefined_lr_scheduler_config(
                        lr_scheduler_name, lr_scheduler_variant
                    )
                    training_config_dict["lr_scheduler"] = scheduler_config
                    logger.info(
                        f"Using custom lr_scheduler: {lr_scheduler_name}_{lr_scheduler_variant}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load custom lr_scheduler '{lr_scheduler_name}_{lr_scheduler_variant}': {e}"
                    )
                    logger.info("Using lr_scheduler from training config")

        # Load stopping rules
        try:
            stopping_rules_dict = get_predefined_stopping_rules_config(
                stopping_rules_variant
            )
            # Add stopping rules to training config for now (could be separate in future)
            training_config_dict["stopping_rules"] = stopping_rules_dict
            logger.info(f"Using stopping rules: {stopping_rules_variant}")
        except Exception as e:
            logger.warning(
                f"Failed to load stopping rules '{stopping_rules_variant}': {e}"
            )
            logger.info("Using default stopping rules")

        training_config = TrainingConfig.from_dict(training_config_dict)

        # Apply loss configuration (loss_variant takes precedence over direct parameters)
        if loss_variant is not None:
            try:
                loss_config = get_predefined_loss_config(loss_variant)
                # Apply loss-specific parameters to training config
                if "loss_function" in loss_config:
                    training_config.loss_function = loss_config["loss_function"]
                if "loss_args" in loss_config:
                    training_config.loss_args = loss_config["loss_args"]

                # Also apply any other training parameters from loss config
                for key, value in loss_config.items():
                    if key not in ["loss_function", "loss_args"] and hasattr(
                        training_config, key
                    ):
                        setattr(training_config, key, value)

                logger.info(
                    f"Using predefined loss variant: {loss_variant} with function: {training_config.loss_function}"
                )
            except Exception as e:
                logger.warning(f"Failed to load loss variant '{loss_variant}': {e}")
                logger.info("Falling back to direct loss parameters")

        # Override loss function if specified directly (takes precedence over loss_variant)
        if loss_function is not None:
            training_config.loss_function = loss_function
            if loss_args is not None:
                training_config.loss_args = loss_args
            else:
                training_config.loss_args = {}
            logger.info(
                f"Using direct loss function: {loss_function} with args: {training_config.loss_args}"
            )

        # Create complete configuration
        nn_config = NNConfig(
            network=network_config,
            training=training_config,
            task_type=task_type,
            experiment_name=experiment_name
            or f"{task_type}_{network_name}_{network_size}_{training_size}",
        )

        logger.info(
            f"Created NN config: {nn_config.experiment_name} (task: {task_type})"
        )
        return nn_config

    except FileNotFoundError as e:
        logger.error(f"Failed to load configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating NN config: {e}")
        raise


# Configurations par défaut pour démarrage rapide
def get_quick_nn_config(
    network_type: str = "mlp",
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    batch_size: int = 256,
) -> NNConfig:
    """Create a quick configuration without loading from files."""

    network_config = NetworkConfig(
        network_type=network_type,
        network_args={"hidden_dims": [64, 64, 64]} if network_type == "mlp" else {},
    )

    training_config = TrainingConfig(
        learning_rate=learning_rate, num_epochs=num_epochs, batch_size=batch_size
    )

    return NNConfig(
        network=network_config,
        training=training_config,
        experiment_name=f"quick_{network_type}_config",
    )
