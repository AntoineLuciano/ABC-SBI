"""
Training loop logic - extracted from train.py

This module handles the core training loop functionality including
epoch execution and phi storage management.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import List, Tuple, Optional, Callable, Dict, Any
import logging


logger = logging.getLogger(__name__)
from functools import partial
from typing import Dict, Any
import jax
import jax.numpy as jnp
import jax.tree_util as jtu



@partial(jax.jit, static_argnames=("batch_size",))
def get_train_batch_fast(
    key: jax.random.PRNGKey, 
    training_set: Dict[str, Any], 
    batch_size: int,
    epoch_idx: int = 0
) -> Dict[str, jnp.ndarray]:


    size = training_set["output"].shape[0]
    
    perm_key = jax.random.fold_in(key, epoch_idx)
    perm = jax.random.permutation(perm_key, size)
    
    if batch_size > size:
        n_repeats = (batch_size // size) + 1
        perm_extended = jnp.tile(perm, n_repeats)
        idx = perm_extended[:batch_size]
    else:
        idx = perm[:batch_size]
    
    batch_input = jtu.tree_map(lambda x: jnp.take(x, idx, axis=0), training_set["input"])
    batch_output = jnp.take(training_set["output"], idx, axis=0)
    
    return {
        "input": batch_input,
        "output": batch_output,
        "n_simulations": jnp.asarray(0, dtype=jnp.int32),
    }




def get_train_batch_old(
    key: jax.random.PRNGKey, training_set: Dict, batch_size: int
) -> Dict[str, jnp.ndarray]:
    """
    Generates a training batch of data.

    Args:
        key: JAX random key for reproducibility
        training_set: The dataset to sample from
        batch_size: Number of samples in the batch

    Returns:
        A dictionary containing the training batch
    """
    # Sample a batch from the training set
    indices = jax.random.choice(
        key, training_set["training_set_size"], (batch_size,), replace=False
    )
    batch_output = training_set["output"][indices]
    if isinstance(training_set["input"], Dict):
        batch_input = {k: v[indices] for k, v in training_set["input"].items()}
    else:
        batch_input = training_set["input"][indices]
    batch_data = {
        "input": batch_input,
        "output": batch_output,
        "n_simulations": 0,
    }
    return batch_data


def get_phi_from_batch(batch_data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Extract phi from the batch data.

    Args:
        batch_data: Dictionary containing batch data with keys 'input' and 'output'

    Returns:
        Extracted phi values as a jnp.ndarray
    """
    inputs = batch_data["input"]
    if type(inputs) is dict:
        return np.unique(inputs["theta"][:, 0])

    return np.unique(inputs[:, -1])


def run_training_epoch(
    epoch: int,
    training_components: Dict[str, Any],
    io_generator: Callable,
    training_config,
    stored_phi: Optional[np.ndarray] = None,
    training_set: Optional[Any] = None,
) -> Tuple[List[float], int, int, Optional[np.ndarray], Dict[str, float]]:
    """
    Run a single training epoch.

    Args:
        epoch: Current epoch number
        training_components: Dictionary with all training components
        io_generator: Function to generate training data batches
        training_config: Training configuration
        stored_phi: Current stored phi values (for phi storage feature)

    Returns:
        (epoch_losses, batch_count, simulations_this_epoch, updated_stored_phi, timing_data)
    """
    n_batch = training_components["n_batch"]
    batch_size = training_config.batch_size
    train_step = training_components["train_step"]
    params = training_components["params"]
    opt_state = training_components["opt_state"]
    key = training_components["key"]

    epoch_losses = []
    simulations_this_epoch = 0

    # Initialize timing trackers
    epoch_start_time = time.time()
    epoch_simulation_time = 0.0
    epoch_training_time = 0.0

    # Check if phi storage is enabled
    n_phi_to_store = getattr(training_config, "n_phi_to_store", 0)
    phi_storage_enabled = n_phi_to_store > 0 and stored_phi is not None
    
    

    for batch_idx in range(n_batch):
        # Generate batch data - TIME SIMULATION
        key, subkey = jax.random.split(key)
        batch_simulation_start_time = time.time()
        if training_config.use_presimulated_data and training_set:
            # Use pre-simulated data
            batch_data = get_train_batch_fast(subkey, training_set, batch_size)
            # No new simulations when using pre-simulated data
        else:
            # Generate new data on-the-fly
            batch_data = io_generator(subkey, batch_size)
            simulations_this_epoch += batch_data["n_simulations"]
        batch_simulation_time =  time.time() - batch_simulation_start_time
        # print(f"Batch {batch_idx} simulation time: {batch_simulation_time:.2f} seconds, total : {epoch_simulation_time:.2f} seconds")
        epoch_simulation_time += batch_simulation_time
        # Validate batch data format
        if not isinstance(batch_data, dict) or "n_simulations" not in batch_data:
            raise ValueError("io_generator must return dict with 'n_simulations' key")

        # Store phi values if requested and not yet full
        if phi_storage_enabled and stored_phi.shape[0] < n_phi_to_store:
            try:
                phis = get_phi_from_batch(batch_data)
                if phis is not None and len(phis) > 0:
                    # Determine how many more phi values we need
                    remaining_capacity = n_phi_to_store - stored_phi.shape[0]
                    phis_to_add = (
                        phis[:remaining_capacity]
                        if len(phis) > remaining_capacity
                        else phis
                    )

                    stored_phi = np.append(stored_phi, phis_to_add, axis=0)

                    if stored_phi.shape[0] >= n_phi_to_store:
                        logger.debug(
                            f"Phi storage completed: {stored_phi.shape[0]} values stored"
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to extract/store phi values in epoch {epoch}, batch {batch_idx}: {e}"
                )

        # Training step - TIME TRAINING
        try:
            # Generate new RNG key for each batch to ensure proper dropout randomness
            key, batch_key = jax.random.split(key)
            epoch_training_start_time = time.time()
            params, opt_state, loss = train_step(
                params, opt_state, batch_data, batch_key
            )
            epoch_training_time += time.time() - epoch_training_start_time
            epoch_losses.append(loss)

        except Exception as e:
            logger.error(
                f"Training step failed in epoch {epoch}, batch {batch_idx}: {e}"
            )
            raise

    # Update training components with new state
    training_components["params"] = params
    training_components["opt_state"] = opt_state
    training_components["key"] = key

    # Calculate timing data
    total_epoch_time = time.time() - epoch_start_time
    overhead_epoch_time = total_epoch_time - epoch_simulation_time - epoch_training_time

    timing_data = {
        "total_time": total_epoch_time,
        "simulation_time": epoch_simulation_time,
        "training_time": epoch_training_time,
        "overhead_time": overhead_epoch_time,
        "sim_to_train_ratio": epoch_simulation_time
        / max(epoch_training_time, 1e-6),  # Avoid division by zero
    }

    return epoch_losses, n_batch, simulations_this_epoch, stored_phi, timing_data


def initialize_phi_storage(training_config) -> Optional[np.ndarray]:
    """
    Initialize phi storage array if phi storage is enabled.

    Args:
        training_config: Training configuration

    Returns:
        Empty numpy array for phi storage, or None if disabled
    """
    n_phi_to_store = getattr(training_config, "n_phi_to_store", 0)

    if n_phi_to_store > 0:
        logger.info(f"Initialized phi storage for {n_phi_to_store} values")
        return np.array([])

    return None


def finalize_phi_storage(
    stored_phi: Optional[np.ndarray], training_config, verbose: bool = True
) -> Optional[np.ndarray]:
    """
    Finalize phi storage and log summary.

    Args:
        stored_phi: Array of stored phi values
        training_config: Training configuration
        verbose: Whether to log phi storage summary

    Returns:
        Final phi array or None
    """
    if stored_phi is None:
        return None

    n_phi_to_store = getattr(training_config, "n_phi_to_store", 0)

    if len(stored_phi) > 0:
        if verbose:
            logger.info(
                f"Phi storage completed: {len(stored_phi)} values stored (target: {n_phi_to_store})"
            )
        return stored_phi
    else:
        if verbose and n_phi_to_store > 0:
            logger.warning("Phi storage was enabled but no phi values were stored")
        return None


def compute_epoch_statistics(epoch_losses: List[float]) -> Dict[str, float]:
    """
    Compute statistics for an epoch's losses.

    Args:
        epoch_losses: List of loss values from the epoch

    Returns:
        Dictionary with loss statistics
    """
    if not epoch_losses:
        return {
            "mean": float("inf"),
            "std": 0.0,
            "min": float("inf"),
            "max": float("inf"),
        }

    losses_array = jnp.array(epoch_losses)

    return {
        "mean": float(jnp.mean(losses_array)),
        "std": float(jnp.std(losses_array)),
        "min": float(jnp.min(losses_array)),
        "max": float(jnp.max(losses_array)),
    }


def validate_training_components(training_components: Dict[str, Any]) -> bool:
    """
    Validate that training components contain all required elements.

    Args:
        training_components: Dictionary with training components

    Returns:
        True if valid, raises ValueError if invalid
    """
    required_keys = [
        "network",
        "params",
        "optimizer",
        "opt_state",
        "train_step",
        "val_data",
        "n_batch",
        "key",
    ]

    for key in required_keys:
        if key not in training_components:
            raise ValueError(f"Missing required training component: {key}")

    # Validate n_batch is positive
    if training_components["n_batch"] <= 0:
        raise ValueError(
            f"n_batch must be positive, got {training_components['n_batch']}"
        )

    return True


def estimate_training_time(
    epoch_times: List[float], current_epoch: int, total_epochs: int
) -> Dict[str, float]:
    """
    Estimate remaining training time based on previous epoch times.

    Args:
        epoch_times: List of time taken for each completed epoch
        current_epoch: Current epoch number (0-based)
        total_epochs: Total number of epochs

    Returns:
        Dictionary with time estimates
    """
    if not epoch_times:
        return {"eta_seconds": 0.0, "eta_minutes": 0.0, "avg_epoch_time": 0.0}

    # Use recent epochs for better estimate (last 10 or all if less)
    recent_times = epoch_times[-10:] if len(epoch_times) > 10 else epoch_times
    avg_epoch_time = sum(recent_times) / len(recent_times)

    remaining_epochs = total_epochs - (current_epoch + 1)
    eta_seconds = avg_epoch_time * remaining_epochs

    return {
        "eta_seconds": eta_seconds,
        "eta_minutes": eta_seconds / 60.0,
        "avg_epoch_time": avg_epoch_time,
    }


def check_training_health(
    epoch_losses: List[float], val_loss: float, learning_rate: float
) -> Dict[str, Any]:
    """
    Check training health and detect potential issues.

    Args:
        epoch_losses: Losses from current epoch
        val_loss: Validation loss
        learning_rate: Current learning rate

    Returns:
        Dictionary with health status and warnings
    """
    warnings = []
    status = "healthy"

    if not epoch_losses:
        return {
            "status": "error",
            "warnings": ["No losses computed"],
            "recommendations": [],
        }

    # Check for exploding gradients
    epoch_stats = compute_epoch_statistics(epoch_losses)
    mean_loss = epoch_stats["mean"]

    if mean_loss > 100:
        warnings.append("Very high training loss - possible exploding gradients")
        status = "warning"

    if jnp.isnan(mean_loss) or jnp.isinf(mean_loss):
        warnings.append("NaN or Inf detected in training loss")
        status = "error"

    if jnp.isnan(val_loss) or jnp.isinf(val_loss):
        warnings.append("NaN or Inf detected in validation loss")
        status = "error"

    # Check learning rate
    if learning_rate < 1e-8:
        warnings.append("Very small learning rate - training may be slow")
        status = "warning"

    if learning_rate > 1.0:
        warnings.append("Very large learning rate - may cause instability")
        status = "warning"

    # Check train/val gap
    if abs(val_loss - mean_loss) > 2.0:
        warnings.append("Large train/validation gap - possible overfitting")
        status = "warning"

    # Generate recommendations based on warnings
    recommendations = []
    for warning in warnings:
        if "exploding gradients" in warning:
            recommendations.append(
                "Consider reducing learning rate or adding gradient clipping"
            )
        elif "NaN or Inf" in warning:
            recommendations.append(
                "Check data for invalid values, reduce learning rate, or add regularization"
            )
        elif "small learning rate" in warning:
            recommendations.append(
                "Consider increasing learning rate or using learning rate scheduling"
            )
        elif "large learning rate" in warning:
            recommendations.append("Consider reducing learning rate")
        elif "overfitting" in warning:
            recommendations.append(
                "Consider adding regularization, reducing model complexity, or more data"
            )

    return {
        "status": status,
        "warnings": warnings,
        "recommendations": recommendations,
        "metrics": {
            "mean_train_loss": mean_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
            "loss_std": epoch_stats["std"],
        },
    }


def log_epoch_summary(
    epoch: int,
    epoch_stats: Dict[str, float],
    val_loss: float,
    health_check: Dict[str, Any],
    verbose: bool = True,
):
    """
    Log a summary of epoch results.

    Args:
        epoch: Epoch number
        epoch_stats: Epoch loss statistics
        val_loss: Validation loss
        health_check: Health check results
        verbose: Whether to log the summary
    """
    if not verbose:
        return

    logger.debug(f"Epoch {epoch} summary:")
    logger.debug(f"  Train loss: {epoch_stats['mean']:.6f} (±{epoch_stats['std']:.6f})")
    logger.debug(f"  Val loss: {val_loss:.6f}")
    logger.debug(f"  Loss range: [{epoch_stats['min']:.6f}, {epoch_stats['max']:.6f}]")
    logger.debug(f"  Health status: {health_check['status']}")

    if health_check["warnings"]:
        for warning in health_check["warnings"]:
            logger.warning(f"  {warning}")
