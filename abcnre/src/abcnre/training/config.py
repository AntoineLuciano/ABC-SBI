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
    loss_args: Dict[str, Any] = field(default_factory=dict)  # Loss specific arguments
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    store_thetas: bool = True
    num_thetas_to_store: int = 10000
    stopping_rules: Optional[Dict[str, Any]] = None
    verbose: bool = True

    # Pre-simulated data configuration
    use_presimulated_data: bool = False
    training_set_size: int = 50000
    validation_set_size: int = 10000

    # Phi storage configuration
    n_phi_to_store: int = 0  # Number of phi values to store during training

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


def _validate_and_convert_numeric_overrides(
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate and convert string representations of numbers to proper numeric types.

    Args:
        overrides: Dictionary containing override values

    Returns:
        Dictionary with numeric values properly converted
    """
    # Numeric fields that should be converted from strings if needed
    numeric_fields = {
        "learning_rate": float,
        "weight_decay": float,
        "num_epochs": int,
        "batch_size": int,
        "patience": int,
        "min_lr": float,
        "dropout_rate": float,
        "momentum": float,
        "beta1": float,
        "beta2": float,
        "eps": float,
    }

    def convert_numeric_values(data: Any) -> Any:
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key in numeric_fields and isinstance(value, str):
                    try:
                        # Try to convert string numbers (including scientific notation)
                        converted_value = numeric_fields[key](float(value))
                        logger.info(
                            f"Converted override '{key}': '{value}' -> {converted_value} ({type(converted_value)})"
                        )
                        result[key] = converted_value
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Could not convert '{key}': '{value}' to {numeric_fields[key].__name__}"
                        )
                        result[key] = value
                else:
                    result[key] = convert_numeric_values(value)
            return result
        elif isinstance(data, list):
            return [convert_numeric_values(item) for item in data]
        else:
            return data

    return convert_numeric_values(overrides)


def _apply_overrides_to_nn_config(
    nn_config: NNConfig, overrides: Dict[str, Any]
) -> NNConfig:
    """
    Apply configuration overrides to an NNConfig object.

    Args:
        nn_config: The base NNConfig object
        overrides: Dictionary containing override values

    Returns:
        Updated NNConfig object with overrides applied
    """
    # Validate and convert numeric values
    validated_overrides = _validate_and_convert_numeric_overrides(overrides)

    # Convert to dict for easier manipulation
    config_dict = asdict(nn_config)

    # Apply overrides using deep merge
    updated_dict = _deep_merge_configs(config_dict, validated_overrides)

    # Reconstruct the NNConfig object
    # Handle nested configs
    if "network" in updated_dict:
        network_config = NetworkConfig.from_dict(updated_dict["network"])
    else:
        network_config = nn_config.network

    if "training" in updated_dict:
        training_config = TrainingConfig.from_dict(updated_dict["training"])
    else:
        training_config = nn_config.training

    # Create new NNConfig with updated components
    return NNConfig(
        network=network_config,
        training=training_config,
        task_type=updated_dict.get("task_type", nn_config.task_type),
        experiment_name=updated_dict.get("experiment_name", nn_config.experiment_name),
    )


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
    training_set_size: Optional[int] = None,
    validation_set_size: Optional[int] = None,
    output_dim: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
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
        training_set_size: Optional size for pre-simulated training set. If > 0, enables pre-simulated data
        validation_set_size: Optional size for pre-simulated validation set
        output_dim: Optional output dimension. If not specified, defaults to 1 for classifiers and is required for regressors
        overrides: Optional dictionary of configuration overrides

    Returns:
        A fully configured NNConfig object.

    Note:
        - If loss_variant is provided, it takes precedence over loss_function/loss_args
        - loss_variant loads complete configurations from loss.yaml
        - loss_function/loss_args provide direct parameter override
        - If training_set_size > 0, automatically enables use_presimulated_data = True
        - For classifiers: output_dim defaults to 1 if not specified
        - For regressors: output_dim must be specified or will default to 1
    """
    try:
        # Load network configuration
        network_config_dict = get_predefined_network_config(network_name, network_size)
        network_config = NetworkConfig.from_dict(network_config_dict)

        # Set output_dim based on task_type and provided parameter
        if task_type == "classifier":
            # For classifiers, output_dim should always be 1
            network_config.network_args["output_dim"] = 1
            if output_dim is not None and output_dim != 1:
                logger.warning(
                    f"For classifiers, output_dim must be 1. Ignoring provided value: {output_dim}"
                )
        elif task_type == "regressor":
            # For regressors, use provided output_dim or default to 1
            if output_dim is not None:
                network_config.network_args["output_dim"] = output_dim
                logger.info(f"Set output_dim={output_dim} for regressor")
            elif "output_dim" not in network_config.network_args:
                network_config.network_args["output_dim"] = 1
                logger.info("No output_dim specified for regressor, defaulting to 1")
        else:
            raise ValueError(
                f"Unknown task_type: {task_type}. Must be 'classifier' or 'regressor'"
            )

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

        # Handle pre-simulated data configuration
        if training_set_size is not None and training_set_size > 0:
            training_config.use_presimulated_data = True
            training_config.training_set_size = training_set_size
            logger.info(
                f"Enabled pre-simulated data with training_set_size: {training_set_size}"
            )

            # Set validation_set_size if provided
            if validation_set_size is not None and validation_set_size > 0:
                training_config.validation_set_size = validation_set_size
                logger.info(f"Set validation_set_size: {validation_set_size}")
            else:
                # Default to 20% of training set size if not specified
                training_config.validation_set_size = max(1000, training_set_size // 5)
                logger.info(
                    f"Auto-set validation_set_size to {training_config.validation_set_size} (20% of training set)"
                )
        elif validation_set_size is not None and validation_set_size > 0:
            # Only validation_set_size specified without training_set_size
            training_config.validation_set_size = validation_set_size
            logger.info(
                f"Set validation_set_size: {validation_set_size} (pre-simulated data not enabled)"
            )

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

        # Apply overrides if provided
        if overrides:
            nn_config = _apply_overrides_to_nn_config(nn_config, overrides)
            logger.info(f"Applied overrides to configuration: {list(overrides.keys())}")

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


@dataclass
class TemplateConfig:
    """Configuration template with support for overrides."""

    network_name: str
    network_size: str
    training_size: str
    task_type: str = "classifier"  # Default to classifier if not specified
    experiment_name: str = ""
    lr_scheduler_name: Optional[str] = None
    lr_scheduler_variant: str = "default"
    stopping_rules_variant: str = "balanced"
    overrides: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, file_path: str) -> "TemplateConfig":
        """Load template configuration from YAML file."""
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)

        # Handle task_type inference from file path if not specified
        if "task_type" not in config:
            if "classifier" in str(file_path).lower():
                config["task_type"] = "classifier"
            elif "regressor" in str(file_path).lower():
                config["task_type"] = "regressor"
            else:
                config["task_type"] = "classifier"  # Default

        return cls.from_dict(config)

    def save(self, file_path: str) -> None:
        """Save template configuration to YAML file."""
        with open(file_path, "w") as f:
            yaml.safe_dump(self.to_dict(), f)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        result = {
            "network_name": self.network_name,
            "network_size": self.network_size,
            "training_size": self.training_size,
            "task_type": self.task_type,
            "experiment_name": self.experiment_name,
        }

        # Add optional fields if they exist
        if self.lr_scheduler_name:
            result["lr_scheduler_name"] = self.lr_scheduler_name
        if self.lr_scheduler_variant != "default":
            result["lr_scheduler_variant"] = self.lr_scheduler_variant
        if self.stopping_rules_variant != "balanced":
            result["stopping_rules_variant"] = self.stopping_rules_variant
        if self.overrides:
            result["overrides"] = self.overrides

        return result

    @classmethod
    def from_dict(cls, config: dict) -> "TemplateConfig":
        """Create from dictionary representation."""
        return cls(
            network_name=config["network_name"],
            network_size=config["network_size"],
            training_size=config["training_size"],
            task_type=config.get("task_type", "classifier"),
            experiment_name=config.get("experiment_name", ""),
            lr_scheduler_name=config.get("lr_scheduler_name"),
            lr_scheduler_variant=config.get("lr_scheduler_variant", "default"),
            stopping_rules_variant=config.get("stopping_rules_variant", "balanced"),
            overrides=config.get("overrides", {}),
        )


def get_nn_config_from_template(template: TemplateConfig) -> NNConfig:
    """Get a complete NN configuration from a template configuration.

    Args:
        template: TemplateConfig object containing base configuration and overrides

    Returns:
        A fully configured NNConfig object with template overrides applied
    """
    return get_nn_config(
        network_name=template.network_name,
        network_size=template.network_size,
        training_size=template.training_size,
        task_type=template.task_type,
        lr_scheduler_name=template.lr_scheduler_name,
        lr_scheduler_variant=template.lr_scheduler_variant,
        stopping_rules_variant=template.stopping_rules_variant,
        experiment_name=template.experiment_name,
        overrides=template.overrides,
    )


# # Default configurations for quick start
# def get_quick_nn_config(
#     network_type: str = "mlp",
#     learning_rate: float = 1e-3,
#     num_epochs: int = 100,
#     batch_size: int = 256,
# ) -> NNConfig:
#     """Create a quick configuration without loading from files."""

#     network_config = NetworkConfig(
#         network_type=network_type,
#         network_args={"hidden_dims": [64, 64, 64]} if network_type == "mlp" else {},
#     )

#     training_config = TrainingConfig(
#         learning_rate=learning_rate, num_epochs=num_epochs, batch_size=batch_size
#     )

#     return NNConfig(
#         network=network_config,
#         training=training_config,
#         experiment_name=f"quick_{network_type}_config",
#     )
