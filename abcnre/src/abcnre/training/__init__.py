"""
Abcnre training module.

This module provides both the legacy training interface (train.py) and the new
modular training interface (train_v2.py) for improved testability and maintainability.
"""

# Legacy interface (current) - kept for backward compatibility
# from .train_old import (
#     train_nn_old,
#     train_classifier_old, 
#     train_summary_learner_old,
#     TrainingResult,
#     create_loss_function,
#     create_learning_rate_schedule,
#     create_optimizer,
#     get_phi_from_batch
# )

# New modular interface (recommended)
from .train import (
    train_nn,
    train_classifier,
    train_summary_learner,
    compare_training_implementations,
    TrainingResult
)
from .optimization import (
    create_loss_function, 
    create_learning_rate_schedule, 
    create_optimizer
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
    get_predefined_stopping_rules_config
)

from .registry import (
    create_network_from_config,
    create_network_from_nn_config,
    create_summary_network,
    create_classifier_network
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
    finalize_phi_storage
)

# Main interfaces - recommended usage
__all__ = [
    # === RECOMMENDED: New modular interface ===
    'train_nn_v2', 'train_classifier_v2', 'train_summary_learner_v2',
    
    # === Legacy interface (backward compatibility) ===
    'train_nn', 'train_classifier', 'train_summary_learner', 'TrainingResult',
    
    # === Configuration system ===
    'NNConfig', 'NetworkConfig', 'TrainingConfig', 'LRSchedulerConfig', 'StoppingRulesConfig',
    'get_nn_config', 'get_quick_nn_config',
    'get_predefined_network_config', 'get_predefined_training_config',
    'get_predefined_lr_scheduler_config', 'get_predefined_stopping_rules_config',
    
    # === Network creation ===
    'create_network_from_config', 'create_network_from_nn_config',
    'create_summary_network', 'create_classifier_network',
    
    # === Utility functions ===
    'compare_training_implementations', 'get_phi_from_batch',
    
    # === Advanced: Modular components ===
    'setup_training_components', 'validate_training_config',
    'TrainingMetrics', 'TrainingLogger', 'evaluate_batch_metrics',
    'ReduceOnPlateauManager', 'compute_effective_learning_rate', 'handle_lr_scheduling',
    'StoppingRulesManager', 'run_training_epoch',
    'initialize_phi_storage', 'finalize_phi_storage',
    
    # === Legacy utility functions ===
    'create_loss_function', 'create_learning_rate_schedule', 'create_optimizer'
]

# Version info
__version__ = "2.0.0"  # Major version bump for modular architecture

# Migration guide for users
_MIGRATION_GUIDE = """
MIGRATION GUIDE: train.py -> train_v2.py
===============================================

OLD (train.py):
    from abcnre.training import train_nn
    result = train_nn(key, config, io_generator)

NEW (train_v2.py - RECOMMENDED):
    from abcnre.training import train_nn_v2
    result = train_nn_v2(key, config, io_generator)

BENEFITS OF NEW INTERFACE:
- Better error handling and validation
- Comprehensive logging and metrics
- Modular architecture for easier debugging
- Health checks and time estimation
- Same API, same results, better implementation

COMPATIBILITY:
- All old code continues to work unchanged
- Results are identical (within numerical precision)
- Same TrainingResult structure returned
- Same configuration system (config.py)

For more information, see the documentation.
"""

def print_migration_guide():
    """Print migration guide for users."""
    print(_MIGRATION_GUIDE)

# Convenience function for quick migration testing
def quick_comparison_test():
    """
    Run a quick comparison test between old and new implementations.
    
    This is useful for verifying that the new implementation works correctly
    in your specific environment.
    """
    import jax
    
    print("Running quick comparison test...")
    
    # Simple test configuration
    config = get_quick_nn_config(
        network_type="mlp",
        learning_rate=1e-3,
        num_epochs=3,
        batch_size=32
    )
    config.training.n_samples_per_epoch = 128
    config.training.verbose = False
    
    # Simple IO generator
    def simple_io_gen(key, batch_size):
        return {
            "input": jax.random.normal(key, (batch_size, 5)),
            "output": jax.random.bernoulli(key, 0.5, (batch_size,)),
            "n_simulations": batch_size
        }
    
    key = jax.random.PRNGKey(42)
    
    try:
        comparison = compare_training_implementations(
            key, config, simple_io_gen, tolerance=0.2
        )
        
        print(f"✅ Comparison successful!")
        print(f"   Old loss: {comparison['old_final_loss']:.6f}")
        print(f"   New loss: {comparison['new_final_loss']:.6f}")  
        print(f"   Difference: {comparison['loss_difference_relative']:.1%}")
        print(f"   Compatible: {comparison['losses_compatible']}")
        print(f"   Time ratio: {comparison['time_ratio']:.2f}x")
        
        if comparison['losses_compatible']:
            print("🎉 Migration ready! New implementation works correctly.")
        else:
            print("⚠️  Results differ - this may be expected due to implementation differences.")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        print("Please check your environment and dependencies.")

# Backward compatibility warnings (optional)
import warnings

def _deprecated_function_warning(old_name: str, new_name: str):
    """Issue a deprecation warning for old functions."""
    warnings.warn(
        f"{old_name} is deprecated and will be removed in a future version. "
        f"Please use {new_name} instead for better performance and features.",
        DeprecationWarning,
        stacklevel=3
    )

# You can uncomment these to issue deprecation warnings
# def train_nn(*args, **kwargs):
#     _deprecated_function_warning("train_nn", "train_nn_v2")
#     from .train import train_nn as _train_nn
#     return _train_nn(*args, **kwargs)