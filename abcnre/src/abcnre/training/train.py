import jax
import jax.numpy as jnp
import time
import logging
from typing import Optional, Any, Callable, Dict, NamedTuple, Union

# Import from existing modules
from .config import NNConfig

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
    estimate_training_time,
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
    # logger.info("DEBUG: TRAINING THE NN...")
    training_config = config.training
    task_type = config.task_type
    # Comprehensive validation using proper config structure
    validate_training_config(config)

    validate_scheduler_config(
        training_config.lr_scheduler, training_config.learning_rate
    )
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
        training_config.lr_scheduler, training_config.learning_rate
    )
    training_components.update(scheduling_state)

    # 4. Initialize phi storage
    stored_phi = initialize_phi_storage(training_config)

    # 5. Initialize counters and state
    total_simulations = training_components["total_simulations"]
    total_samples = training_components["total_samples"]
    step_counter = 0
    epoch_times = []
    epoch_timing_data = []  # Store detailed timing for each epoch

    # Record global start time for comprehensive timing analysis
    global_start_time = time.time()

    # Initialize comprehensive training history storage
    training_history_detailed = {
        "global_start_time": global_start_time,
        "epochs": [],
        "train_losses": [],
        "val_losses": [],
        "learning_rates": [],
        "total_simulations": [],
        "total_samples": [],
        "epoch_times": [],
        "simulation_times": [],
        "training_times": [],
        "overhead_times": [],
        "sim_to_train_ratios": [],
        "train_accuracies": [],  # For classifiers
        "val_accuracies": [],  # For classifiers
        "elapsed_times": [],
        "health_checks": [],
        "metrics_times": [],  # Time spent on metrics calculation
    }

    # 6. Log training start with stopping rules info
    training_logger.log_training_start(
        task_type,
        training_config.num_epochs,
        training_config.batch_size,
        training_components["n_batch"],
    )

    if training_config.verbose:
        active_criteria = stopping_manager.get_active_criteria()
        logger.info(f"Active stopping criteria: {active_criteria}")

    # 7. Main training loop
    start_time = time.time()

    # Calculate initialization time (time spent before first epoch)
    initialization_time = start_time - training_history_detailed["global_start_time"]

    # Initialize pre-simulated training data if enabled
    if training_config.use_presimulated_data:
        training_set_size = training_config.training_set_size
        key, key_io = jax.random.split(key)
        training_set = io_generator(key_io, training_set_size)
        total_samples += training_set_size
        total_simulations += training_set.num_simulations
        sim_time = time.time() - start_time
        if training_config.verbose:
            logger.info(
                f"Pre-simulated training set created: {training_set_size} samples in {sim_time:.2f} seconds"
            )
    else:
        training_set = None
    # logger.info(f"DEBUG: STARTING EPOCH LOOP...")
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
            # logger.info(f"DEBUG: STARTING EPOCH {epoch}...")
            (
                epoch_losses,
                batch_count,
                simulations_this_epoch,
                stored_phi,
                timing_data,
            ) = run_training_epoch(
                epoch,
                training_components,
                io_generator,
                training_config,
                stored_phi,
                training_set,
            )
            # logger.info(f"DEBUG: EPOCH {epoch} DONE.")
        except Exception as e:
            logger.error(f"Training epoch {epoch} failed: {e}")
            raise

        # Update counters
        if not training_config.use_presimulated_data:
            total_samples += batch_count * training_config.batch_size
            total_simulations += simulations_this_epoch

        step_counter += batch_count

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        # Store detailed timing data
        epoch_timing_data.append(
            {
                "epoch": epoch,
                "total_time": timing_data["total_time"],
                "simulation_time": timing_data["simulation_time"],
                "training_time": timing_data["training_time"],
                "overhead_time": timing_data["overhead_time"],
                "sim_to_train_ratio": timing_data["sim_to_train_ratio"],
            }
        )

        # Calculate epoch metrics - START TIMING METRICS
        metrics_start_time = time.time()
        if epoch_losses:
            epoch_stats = compute_epoch_statistics(epoch_losses)
            mean_loss = epoch_stats["mean"]

            # Evaluate validation metrics
            val_loss, val_metrics = training_components["evaluate_val"](
                training_components["params"], training_components["val_data"]
            )  # Handle LR scheduling using proper config structure
            if plateau_manager:
                early_stop, training_components = handle_lr_scheduling(
                    training_config.lr_scheduler,
                    training_config.optimizer,
                    training_config.weight_decay,
                    mean_loss,
                    plateau_manager,
                    training_components,
                    training_logger,
                )
                if early_stop:
                    break

            # Compute effective learning rate using proper config
            effective_lr = compute_effective_learning_rate(
                training_components["lr_schedule"],
                training_config.lr_scheduler,
                step_counter,
                training_components.get("current_lr", training_config.learning_rate),
            )

            # Prepare epoch metrics
            epoch_data = {
                "train_loss": mean_loss,
                "val_loss": float(val_loss),
                "learning_rate": effective_lr,
                "total_simulations": total_simulations,
            }

            # Add accuracy for classifier
            if task_type == "classifier":
                epoch_data["val_accuracy"] = float(val_metrics.get("accuracy", 0))

                # Compute train accuracy on last batch
                last_batch_key, _ = jax.random.split(training_components["key"])
                last_batch = io_generator(last_batch_key, training_config.batch_size)
                train_metrics = evaluate_batch_metrics(
                    task_type,
                    training_components["params"],
                    last_batch,
                    training_components["network"],
                    training_components["loss_fn"],
                )
                epoch_data["train_accuracy"] = train_metrics.get("accuracy", 0)

            # Store metrics
            metrics_manager.add_epoch_metrics(epoch_data)

            # Calculate metrics time - END TIMING METRICS
            metrics_time = time.time() - metrics_start_time

            # Store detailed epoch data in comprehensive history
            elapsed_time = time.time() - start_time
            latest_timing = epoch_timing_data[-1]

            training_history_detailed["epochs"].append(epoch)
            training_history_detailed["train_losses"].append(float(mean_loss))
            training_history_detailed["val_losses"].append(float(val_loss))
            training_history_detailed["learning_rates"].append(effective_lr)
            training_history_detailed["total_simulations"].append(total_simulations)
            training_history_detailed["total_samples"].append(total_samples)
            training_history_detailed["epoch_times"].append(epoch_time)
            training_history_detailed["simulation_times"].append(
                latest_timing["simulation_time"]
            )
            training_history_detailed["training_times"].append(
                latest_timing["training_time"]
            )
            training_history_detailed["overhead_times"].append(
                latest_timing["overhead_time"]
            )
            training_history_detailed["sim_to_train_ratios"].append(
                latest_timing["sim_to_train_ratio"]
            )
            training_history_detailed["elapsed_times"].append(elapsed_time)
            training_history_detailed["metrics_times"].append(metrics_time)

            # Add accuracies for classifier
            if task_type == "classifier":
                training_history_detailed["train_accuracies"].append(
                    float(epoch_data.get("train_accuracy", 0))
                )
                training_history_detailed["val_accuracies"].append(
                    float(epoch_data.get("val_accuracy", 0))
                )
            else:
                training_history_detailed["train_accuracies"].append(None)
                training_history_detailed["val_accuracies"].append(None)

            # Health check
            health_check = check_training_health(
                epoch_losses, float(val_loss), effective_lr
            )

            # Store health check in detailed history
            training_history_detailed["health_checks"].append(health_check)

            if health_check["status"] == "error":
                error_msg = f"Training health check failed at epoch {epoch}: {health_check['warnings']}"
                logger.error(error_msg)
                training_logger.log_early_stopping(
                    epoch, "Training health check failed"
                )
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
                elapsed_time,
            )
            if should_stop:
                training_logger.log_early_stopping(epoch, reason)
                break

            # Log progress with time estimation
            if training_config.verbose and epoch % 10 == 0:
                time_estimate = estimate_training_time(
                    epoch_times, epoch, training_config.num_epochs
                )
                training_logger.log_epoch_progress(
                    epoch,
                    training_config.num_epochs,
                    metrics_manager,
                    elapsed_time,
                    total_samples,
                )

                # Enhanced timing breakdown
                latest_timing = epoch_timing_data[-1]

                data_mode = (
                    "Pre-sim" if training_config.use_presimulated_data else "On-the-fly"
                )

                sim_times = [t["simulation_time"] for t in epoch_timing_data]
                train_times = [t["training_time"] for t in epoch_timing_data]
                total_sim_time = sum(sim_times)
                total_train_time = sum(train_times)
                overhead_times = [t["overhead_time"] for t in epoch_timing_data]
                total_overhead = sum(overhead_times)

                logger.info(
                    f"Timing [{data_mode}]: Sim={total_sim_time:.2f}s "
                    f"Train={total_train_time:.2f}s "
                    f"Overhead={total_overhead:.2f}s "
                    f"(Sim/Train: {total_sim_time/total_train_time:.1f}x)"
                )

                if time_estimate["eta_minutes"] > 1 and epoch >= 10:
                    logger.info(
                        f"ETA (for {training_config.num_epochs} epochs): {time_estimate['eta_minutes']:.1f} minutes"
                    )

        else:
            logger.warning(f"No losses computed for epoch {epoch}")

    # 8. Finalize training
    total_time = time.time() - start_time
    training_logger.log_training_completion(metrics_manager, total_time)

    # Finalize phi storage
    stored_phi = finalize_phi_storage(
        stored_phi, training_config, training_config.verbose
    )

    # Log stopping rules summary and final health check
    if training_config.verbose:
        stopping_info = stopping_manager.get_stopping_info()
        logger.info(f"Final stopping rules status: {stopping_info}")

        # Enhanced timing summary with simulation/training breakdown
        if epoch_timing_data:
            import numpy as np

            sim_times = [t["simulation_time"] for t in epoch_timing_data]
            train_times = [t["training_time"] for t in epoch_timing_data]
            total_sim_time = sum(sim_times)
            total_train_time = sum(train_times)
            overhead_times = [t["overhead_time"] for t in epoch_timing_data]
            total_overhead = sum(overhead_times)

            logger.info("=" * 60)
            logger.info("DETAILED TIMING BREAKDOWN")
            logger.info("=" * 60)
            logger.info(
                f"Total training time: {total_time:.2f}s ({total_time/60:.1f}m)"
            )
            logger.info(
                f"  - Pure simulation time: {total_sim_time:.2f}s ({total_sim_time/total_time*100:.1f}%)"
            )
            logger.info(
                f"  - Pure training time: {total_train_time:.2f}s ({total_train_time/total_time*100:.1f}%)"
            )
            logger.info(
                f"  - Overhead time: {total_overhead:.2f}s ({total_overhead/total_time*100:.1f}%)"
            )

            # Performance metrics
            avg_sim_ratio = np.mean(
                [t["sim_to_train_ratio"] for t in epoch_timing_data]
            )
            logger.info(f"Average sim/train ratio: {avg_sim_ratio:.2f}x")

            if training_config.use_presimulated_data:
                logger.info(
                    "Note: Using pre-simulated data - simulation time includes data sampling"
                )
            else:
                logger.info(
                    "Note: Using on-the-fly simulation - simulation time includes full model execution"
                )

            logger.info("=" * 60)

        # Final metrics summary
        training_logger.log_metrics_summary(metrics_manager)

    # 9. Prepare results
    training_history = metrics_manager.get_final_summary(total_time)

    # Add our comprehensive detailed history
    training_history["detailed_history"] = training_history_detailed

    # Add additional training statistics
    if epoch_times:
        training_history["avg_epoch_time"] = sum(epoch_times) / len(epoch_times)
        training_history["total_epochs_run"] = len(epoch_times)

    # Add detailed timing statistics
    if epoch_timing_data:
        import numpy as np

        sim_times = [t["simulation_time"] for t in epoch_timing_data]
        train_times = [t["training_time"] for t in epoch_timing_data]
        ratios = [t["sim_to_train_ratio"] for t in epoch_timing_data]

        # Calculate total metrics time
        metrics_times = [t.get("metrics_time", 0) for t in epoch_timing_data]
        total_metrics_time = sum(metrics_times)

        training_history.update(
            {
                "timing_per_epoch": epoch_timing_data,
                "initialization_time": initialization_time,
                "total_simulation_time": sum(sim_times),
                "total_training_time": sum(train_times),
                "total_metrics_time": total_metrics_time,
                "avg_simulation_time": np.mean(sim_times),
                "avg_training_time": np.mean(train_times),
                "avg_metrics_time": np.mean(metrics_times) if metrics_times else 0,
                "avg_sim_to_train_ratio": np.mean(ratios),
                "simulation_time_percentage": (
                    sum(sim_times) / sum([t["total_time"] for t in epoch_timing_data])
                )
                * 100,
                "training_time_percentage": (
                    sum(train_times) / sum([t["total_time"] for t in epoch_timing_data])
                )
                * 100,
                "metrics_time_percentage": (
                    (
                        total_metrics_time
                        / sum([t["total_time"] for t in epoch_timing_data])
                    )
                    * 100
                    if total_metrics_time > 0
                    else 0
                ),
            }
        )

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
        raise ValueError(
            "config.task_type must be 'classifier' for train_classifier_v2"
        )
    return train_nn(key, config, io_generator, network, val_io_generator)


def train_regressor(
    key: jax.random.PRNGKey,
    config: NNConfig,
    io_generator: Callable,
    network: Optional[Any] = None,
    val_io_generator: Optional[Callable] = None,
) -> TrainingResult:
    """
    Train a regressor network using the new modular architecture.

    Args:
        key: JAX random key for initialization
        config: NNConfig with task_type="regressor"
        io_generator: Function that generates training data
        network: Optional pre-initialized network
        val_io_generator: Optional validation data generator

    Returns:
        TrainingResult with trained parameters and summary function
    """
    if config.task_type != "regressor":
        raise ValueError("config.task_type must be 'regressor' for train_regressor")

    return train_nn(key, config, io_generator, network, val_io_generator)
