"""
Abcnre training module.

This module provides both the legacy training interface (train.py) and the new
modular training interface (train_v2.py) for improved testability and maintainability.
"""

# New modular interface (recommended)
from .train import train_nn, train_classifier, train_regressor, TrainingResult
from .optimization import (
    create_loss_function,
    create_learning_rate_schedule,
    create_optimizer,
)

# Config and registry (unchanged)
from .config import (
    NNConfig,
    NetworkConfig,
    TrainingConfig,
    LRSchedulerConfig,
    StoppingRulesConfig,
    get_nn_config,
    get_quick_nn_config,
    get_predefined_network_config,
    get_predefined_training_config,
    get_predefined_lr_scheduler_config,
    get_predefined_stopping_rules_config,
)

from .registry import (
    create_network_from_config,
    create_network_from_nn_config,
    create_summary_network,
    create_classifier_network,
)

# Modular components (for advanced users)
from .components import (
    # Setup
    setup_training_components,
    validate_training_config,
    # Metrics
    TrainingMetrics,
    TrainingLogger,
    evaluate_batch_metrics,
    # Schedulers
    ReduceOnPlateauManager,
    compute_effective_learning_rate,
    handle_lr_scheduling,
    # Stopping rules
    StoppingRulesManager,
    # Loop
    run_training_epoch,
    initialize_phi_storage,
    finalize_phi_storage,
)

# Main interfaces - recommended usage
__all__ = [
    "train_nn",
    "train_classifier",
    "train_regressor",
    "TrainingResult",
    "NNConfig",
    "NetworkConfig",
    "TrainingConfig",
    "LRSchedulerConfig",
    "StoppingRulesConfig",
    "get_nn_config",
    "get_quick_nn_config",
    "get_predefined_network_config",
    "get_predefined_training_config",
    "get_predefined_lr_scheduler_config",
    "get_predefined_stopping_rules_config",
    "create_network_from_config",
    "create_network_from_nn_config",
    "create_summary_network",
    "create_classifier_network",
    "get_phi_from_batch",
    "setup_training_components",
    "validate_training_config",
    "TrainingMetrics",
    "TrainingLogger",
    "evaluate_batch_metrics",
    "ReduceOnPlateauManager",
    "compute_effective_learning_rate",
    "handle_lr_scheduling",
    "StoppingRulesManager",
    "run_training_epoch",
    "initialize_phi_storage",
    "finalize_phi_storage",
    "create_loss_function",
    "create_learning_rate_schedule",
    "create_optimizer",
]

# Version info
__version__ = "2.0.0"  # Major version bump for modular architecture
