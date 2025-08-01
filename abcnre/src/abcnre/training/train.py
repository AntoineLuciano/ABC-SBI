"""
Refactored training interface using modular components.

This module provides a cleaner, more testable version of the training system
while maintaining backward compatibility with the existing train.py.
"""

import jax
import jax.numpy as jnp
import time
import logging
from typing import Optional, Any, Callable, Dict, NamedTuple, Union

# Import from existing modules
from .config import NNConfig

# from .train_old import TrainingResult

# Import new modular components
from .components import (
    setup_training_components,
    validate_training_config,
    log_config_summary,
    StoppingRulesManager,
    TrainingMetrics,
    TrainingLogger,
    evaluate_batch_metrics,
    ReduceOnPlateauManager,
    compute_effective_learning_rate,
    handle_lr_scheduling,
    initialize_lr_scheduling,
    validate_scheduler_config,
    run_training_epoch,
    initialize_phi_storage,
    finalize_phi_storage,
    compute_epoch_statistics,
    check_training_health,
    estimate_training_time
)

logger = logging.getLogger(__name__)

class TrainingResult(NamedTuple):
    """Base result class for all training tasks."""

    params: Any
    config: NNConfig
    training_history: Dict[str, Any]
    network: Any
    stored_phi: Optional[jnp.ndarray] = None

def train_nn(
    key: jax.random.PRNGKey,
    config: NNConfig,
    io_generator: Callable,
    network: Optional[Any] = None,
    val_io_generator: Optional[Callable] = None,
) -> TrainingResult:
    """
    Refactored unified training function using modular components.
    
    This is the new implementation that replaces the monolithic train_nn.
    It uses the same interface and returns the same results for compatibility,
    but with improved modularity, testability, and maintainability.
    
    Args:
        key: JAX random key for initialization
        config: NNConfig containing all training parameters
        io_generator: Function that generates training data
        network: Optional pre-initialized network
        val_io_generator: Optional validation data generator

    Returns:
        TrainingResult with trained parameters and inference function
    """
    training_config = config.training
    task_type = config.task_type
    
    # Comprehensive validation using proper config structure
    validate_training_config(config)
    validate_scheduler_config(training_config.lr_scheduler, training_config.learning_rate)
    
    # Log configuration summary
    log_config_summary(config, training_config.verbose)
    
    # 1. Setup all training components
    training_components = setup_training_components(
        key, config, io_generator, network, val_io_generator
    )
    
    # 2. Initialize managers using proper config structures
    stopping_manager = StoppingRulesManager(training_config)
    metrics_manager = TrainingMetrics(task_type)
    training_logger = TrainingLogger(training_config.verbose, task_type)
    
    # 3. Initialize LR scheduling using proper LRSchedulerConfig
    plateau_manager, scheduling_state = initialize_lr_scheduling(
        training_config.lr_scheduler,
        training_config.learning_rate
    )
    training_components.update(scheduling_state)
    
    # 4. Initialize phi storage
    stored_phi = initialize_phi_storage(training_config)
    
    # 5. Initialize counters and state
    total_simulations = training_components["total_simulations"]
    total_samples = training_components["total_samples"]
    step_counter = 0
    epoch_times = []
    
    # 6. Log training start with stopping rules info
    training_logger.log_training_start(
        task_type,
        training_config.num_epochs,
        training_config.batch_size,
        training_components["n_batch"]
    )
    
    if training_config.verbose:
        active_criteria = stopping_manager.get_active_criteria()
        logger.info(f"Active stopping criteria: {active_criteria}")
    
    # 7. Main training loop
    start_time = time.time()
    
    for epoch in range(training_config.num_epochs):
        epoch_start_time = time.time()
        
        # Check stopping rules before epoch using proper structure
        should_stop, reason = stopping_manager.check_before_epoch(
            epoch, training_config.n_samples_per_epoch
        )
        if should_stop:
            training_logger.log_early_stopping(epoch, reason)
            break
        
        # Run training epoch
        try:
            epoch_losses, batch_count, simulations_this_epoch, stored_phi = run_training_epoch(
                epoch, training_components, io_generator, training_config, stored_phi
            )
        except Exception as e:
            logger.error(f"Training epoch {epoch} failed: {e}")
            raise
        
        # Update counters
        total_samples += batch_count * training_config.batch_size
        total_simulations += simulations_this_epoch
        step_counter += batch_count
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Calculate epoch metrics
        if epoch_losses:
            epoch_stats = compute_epoch_statistics(epoch_losses)
            mean_loss = epoch_stats["mean"]
            
            # Evaluate validation metrics
            val_loss, val_metrics = training_components["evaluate_val"](
                training_components["params"], 
                training_components["val_data"]
            )
            
            # Handle LR scheduling using proper config structure
            if plateau_manager:
                early_stop, training_components = handle_lr_scheduling(
                    training_config.lr_scheduler,
                    training_config.optimizer,
                    training_config.weight_decay,
                    mean_loss, 
                    plateau_manager, 
                    training_components, 
                    training_logger
                )
                if early_stop:
                    break
            
            # Compute effective learning rate using proper config
            effective_lr = compute_effective_learning_rate(
                training_components["lr_schedule"],
                training_config.lr_scheduler,
                step_counter,
                training_components.get("current_lr", training_config.learning_rate)
            )
            
            # Prepare epoch metrics
            epoch_data = {
                "train_loss": mean_loss,
                "val_loss": float(val_loss),
                "learning_rate": effective_lr,
                "total_simulations": total_simulations
            }
            
            # Add accuracy for classifier
            if task_type == "classifier":
                epoch_data["val_accuracy"] = float(val_metrics.get("accuracy", 0))
                
                # Compute train accuracy on last batch
                last_batch_key, _ = jax.random.split(training_components["key"])
                last_batch = io_generator(last_batch_key, training_config.batch_size)
                train_metrics = evaluate_batch_metrics(
                    task_type, training_components["params"], last_batch,
                    training_components["network"], training_components["loss_fn"]
                )
                epoch_data["train_accuracy"] = train_metrics.get("accuracy", 0)
            
            # Store metrics
            metrics_manager.add_epoch_metrics(epoch_data)
            
            # Health check
            health_check = check_training_health(epoch_losses, float(val_loss), effective_lr)
            
            if health_check["status"] == "error":
                error_msg = f"Training health check failed at epoch {epoch}: {health_check['warnings']}"
                logger.error(error_msg)
                training_logger.log_early_stopping(epoch, "Training health check failed")
                break
            
            # Check stopping rules after epoch using proper structure with all parameters
            elapsed_time = time.time() - start_time
            should_stop, reason = stopping_manager.check_after_epoch(
                epoch, 
                total_samples, 
                total_simulations, 
                training_config.n_samples_per_epoch,
                mean_loss,
                float(val_loss),
                effective_lr,
                elapsed_time
            )
            if should_stop:
                training_logger.log_early_stopping(epoch, reason)
                break
            
            # Log progress with time estimation
            if training_config.verbose and epoch % 10 == 0:
                time_estimate = estimate_training_time(epoch_times, epoch, training_config.num_epochs)
                training_logger.log_epoch_progress(
                    epoch, training_config.num_epochs, metrics_manager, 
                    elapsed_time, total_samples
                )
                if time_estimate["eta_minutes"] > 1:
                    logger.info(f"ETA: {time_estimate['eta_minutes']:.1f} minutes")
        
        else:
            logger.warning(f"No losses computed for epoch {epoch}")
    
    # 8. Finalize training
    total_time = time.time() - start_time
    training_logger.log_training_completion(metrics_manager, total_time)
    
    # Finalize phi storage
    stored_phi = finalize_phi_storage(stored_phi, training_config, training_config.verbose)
    
    # Log stopping rules summary and final health check
    if training_config.verbose:
        stopping_info = stopping_manager.get_stopping_info()
        logger.info(f"Final stopping rules status: {stopping_info}")
        
        # Final metrics summary
        training_logger.log_metrics_summary(metrics_manager)
    
    # 9. Prepare results
    training_history = metrics_manager.get_final_summary(total_time)
    
    # Add additional training statistics
    if epoch_times:
        training_history["avg_epoch_time"] = sum(epoch_times) / len(epoch_times)
        training_history["total_epochs_run"] = len(epoch_times)
    
    result_kwargs = {
        "params": training_components["params"],
        "network": training_components["network"],
        "training_history": training_history,
        "config": config,
    }
    
    if stored_phi is not None:
        result_kwargs["stored_phi"] = stored_phi
        if training_config.verbose:
            logger.info(f"Stored phi values: {len(stored_phi)}")
    
    return TrainingResult(**result_kwargs)


# Convenience wrappers for backward compatibility
def train_classifier(
    key: jax.random.PRNGKey,
    config: NNConfig,
    io_generator: Callable,
    network: Optional[Any] = None,
    val_io_generator: Optional[Callable] = None,
) -> TrainingResult:
    """
    Train a classifier network using the new modular architecture.
    
    Args:
        key: JAX random key for initialization
        config: NNConfig with task_type="classifier"
        io_generator: Function that generates training data
        network: Optional pre-initialized network
        val_io_generator: Optional validation data generator

    Returns:
        TrainingResult with trained parameters and classifier function
    """
    if config.task_type != "classifier":
        raise ValueError("config.task_type must be 'classifier' for train_classifier_v2")

    return train_nn(key, config, io_generator, network, val_io_generator)


def train_summary_learner(
    key: jax.random.PRNGKey,
    config: NNConfig,
    io_generator: Callable,
    network: Optional[Any] = None,
    val_io_generator: Optional[Callable] = None,
) -> TrainingResult:
    """
    Train a summary learner network using the new modular architecture.
    
    Args:
        key: JAX random key for initialization
        config: NNConfig with task_type="summary_learner"
        io_generator: Function that generates training data
        network: Optional pre-initialized network
        val_io_generator: Optional validation data generator

    Returns:
        TrainingResult with trained parameters and summary function
    """
    if config.task_type != "summary_learner":
        raise ValueError("config.task_type must be 'summary_learner' for train_summary_learner_v2")

    return train_nn(key, config, io_generator, network, val_io_generator)


# Utility function for comparing old vs new training
def compare_training_implementations(
    key: jax.random.PRNGKey,
    config: NNConfig,
    io_generator: Callable,
    network: Optional[Any] = None,
    val_io_generator: Optional[Callable] = None,
    tolerance: float = 0.1
) -> Dict[str, Any]:
    """
    Compare results between old and new training implementations.
    
    Args:
        key: JAX random key for initialization
        config: NNConfig containing all training parameters
        io_generator: Function that generates training data
        network: Optional pre-initialized network
        val_io_generator: Optional validation data generator
        tolerance: Relative tolerance for comparing final losses
        
    Returns:
        Dictionary with comparison results
    """
    # Import here to avoid circular imports
    from .train_old import train_nn_old
    
    # Use same random seeds for fair comparison
    key1, key2 = jax.random.split(key)
    
    # Disable verbose logging for comparison
    original_verbose = config.training.verbose
    config.training.verbose = False
    
    try:
        # Run old implementation
        start_time = time.time()
        result_old = train_nn_old(key1, config, io_generator, network, val_io_generator)
        old_time = time.time() - start_time
        
        # Run new implementation
        start_time = time.time()
        result_new = train_nn(key2, config, io_generator, network, val_io_generator)
        new_time = time.time() - start_time
        
        # Compare results
        old_loss = result_old.training_history["final_loss"]
        new_loss = result_new.training_history["final_loss"]
        
        loss_diff_relative = abs(old_loss - new_loss) / max(old_loss, new_loss) if max(old_loss, new_loss) > 0 else 0
        
        comparison = {
            "old_final_loss": old_loss,
            "new_final_loss": new_loss,
            "loss_difference_relative": loss_diff_relative,
            "losses_compatible": loss_diff_relative < tolerance,
            "old_time": old_time,
            "new_time": new_time,
            "time_ratio": new_time / old_time if old_time > 0 else float("inf"),
            "old_simulations": result_old.training_history["total_simulations"],
            "new_simulations": result_new.training_history["total_simulations"],
            "simulations_match": result_old.training_history["total_simulations"] == result_new.training_history["total_simulations"]
        }
        
        # Compare stored phi if present
        if hasattr(result_old, 'stored_phi') and hasattr(result_new, 'stored_phi'):
            old_phi_count = len(result_old.stored_phi) if result_old.stored_phi is not None else 0
            new_phi_count = len(result_new.stored_phi) if result_new.stored_phi is not None else 0
            comparison["old_phi_count"] = old_phi_count
            comparison["new_phi_count"] = new_phi_count
            comparison["phi_counts_match"] = old_phi_count == new_phi_count
        
        return comparison
        
    finally:
        # Restore original verbose setting
        config.training.verbose = original_verbose