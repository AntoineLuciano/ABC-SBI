"""
Modular training components for improved testability and maintainability.

This package provides modular, testable components that replace the monolithic
train.py function with clean separation of concerns.
"""

# Setup components
from .setup import (
    setup_training_components,
    validate_training_config,
    validate_io_generator_compatibility,
    get_parameter_count,
    log_setup_summary,
    create_validation_function,
    create_training_step_function,
)

# Metrics and logging components
from .metrics import (
    TrainingMetrics,
    TrainingLogger,
    evaluate_batch_metrics,
    log_config_summary,
    compute_metrics_delta,
    is_training_stagnant,
)

# Learning rate scheduling components
from .schedulers import (
    ReduceOnPlateauManager,
    compute_effective_learning_rate,
    handle_lr_scheduling,
    initialize_lr_scheduling,
    get_scheduler_info,
    validate_scheduler_config,
    create_lr_schedule_from_config,
    is_reduce_on_plateau,
    get_lr_reduction_history,
    should_continue_training,
)

# Stopping rules components
from .stopping import (
    StoppingRulesManager,
    get_stopping_summary,
    validate_stopping_rules_consistency,
)

# Training loop components
from .loop import (
    run_training_epoch,
    initialize_phi_storage,
    finalize_phi_storage,
    compute_epoch_statistics,
    validate_training_components,
    estimate_training_time,
    check_training_health,
    log_epoch_summary,
)

# Main public API - most commonly used functions
__all__ = [
    # === CORE TRAINING FUNCTIONS ===
    "setup_training_components",
    "run_training_epoch",
    # === MANAGERS (main classes) ===
    "TrainingMetrics",
    "TrainingLogger",
    "StoppingRulesManager",
    "ReduceOnPlateauManager",
    # === VALIDATION ===
    "validate_training_config",
    "validate_scheduler_config",
    "validate_io_generator_compatibility",
    "validate_training_components",
    # === METRICS & LOGGING ===
    "evaluate_batch_metrics",
    "log_config_summary",
    "compute_metrics_delta",
    "is_training_stagnant",
    # === SCHEDULING ===
    "compute_effective_learning_rate",
    "handle_lr_scheduling",
    "initialize_lr_scheduling",
    "get_scheduler_info",
    "is_reduce_on_plateau",
    # === PHI STORAGE ===
    "initialize_phi_storage",
    "finalize_phi_storage",
    # === UTILITIES ===
    "compute_epoch_statistics",
    "estimate_training_time",
    "check_training_health",
    "get_parameter_count",
    "get_stopping_summary",
    # === ADVANCED/INTERNAL (for debugging) ===
    "create_validation_function",
    "create_training_step_function",
    "create_lr_schedule_from_config",
    "get_lr_reduction_history",
    "should_continue_training",
    "validate_stopping_rules_consistency",
    "log_setup_summary",
    "log_epoch_summary",
]

# Package metadata
__version__ = "2.0.0"
__author__ = "ABC-NRE Training Team"
__description__ = "Modular training components for neural ratio estimation"

# Usage examples in docstring
__doc__ += """

BASIC USAGE:
===========

# Simple training setup
components = setup_training_components(key, config, io_generator)
metrics = TrainingMetrics("classifier")
logger = TrainingLogger(verbose=True, task_type="classifier")

# Training loop
for epoch in range(num_epochs):
    losses, _, sims, phi = run_training_epoch(epoch, components, io_gen, config)
    # ... handle metrics and logging

ADVANCED USAGE:
===============

# Stopping rules
stopping_manager = StoppingRulesManager(config.training)
should_stop, reason = stopping_manager.check_after_epoch(...)

# Learning rate scheduling  
plateau_manager = ReduceOnPlateauManager(lr_config, base_lr)
early_stop, components = handle_lr_scheduling(...)

# Health monitoring
health = check_training_health(losses, val_loss, lr)
if health["status"] == "error":
    # Handle training issues

See train_v2.py for complete integration example.
"""

# Backward compatibility aliases (deprecated, but maintained)
_TrainingMetrics = TrainingMetrics
_TrainingLogger = TrainingLogger
_StoppingRulesManager = StoppingRulesManager


# Internal version for debugging
def _get_component_versions():
    """Get version info for all components (debugging)."""
    return {
        "package_version": __version__,
        "components": {
            "setup": "2.0.0",
            "metrics": "2.0.0",
            "schedulers": "2.0.0",
            "stopping": "2.0.0",
            "loop": "2.0.0",
        },
    }


# Validation function for complete package
def validate_components_installation():
    """
    Validate that all components can be imported and initialized.

    Returns:
        bool: True if all components work, False otherwise
    """
    try:
        # Test core imports
        from .setup import setup_training_components
        from .metrics import TrainingMetrics, TrainingLogger
        from .schedulers import ReduceOnPlateauManager
        from .stopping import StoppingRulesManager
        from .loop import run_training_epoch

        # Test basic instantiation
        metrics = TrainingMetrics("classifier")
        logger = TrainingLogger(False, "classifier")

        return True

    except Exception as e:
        print(f"Component validation failed: {e}")
        return False


# Quick health check
if __name__ == "__main__":
    print(f"Training Components v{__version__}")
    print("Running installation validation...")

    if validate_components_installation():
        print("All components installed correctly")
    else:
        print("Component installation issues detected")
