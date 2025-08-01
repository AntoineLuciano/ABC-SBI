"""
OUTDATED
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Dict, Any, Callable, Tuple, Union, NamedTuple, Optional
from pathlib import Path
import logging
import time

from .config import NNConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
        return np.unique(inputs["theta"][:,0])
    
    
    return np.unique(inputs[:, -1])

def create_loss_function(task_type: str, network) -> Callable:
    """Create the appropriate loss function for the task type."""

    if task_type == "classifier":

        def binary_crossentropy_loss(params, batch_data):
            batch_input = batch_data["input"]
            batch_output = batch_data["output"]
            logits = network.apply(params, batch_input, training=True)
            # Convert to probabilities and compute BCE
            probs = jax.nn.sigmoid(logits.squeeze(-1))
            # Clip to avoid log(0)
            probs = jnp.clip(probs, 1e-7, 1 - 1e-7)
            bce = -jnp.mean(batch_output * jnp.log(probs) + (1 - batch_output) * jnp.log(1 - probs))
            return bce

        return binary_crossentropy_loss

    elif task_type == "summary_learner":

        def mse_loss(params, batch_data):
            batch_input = batch_data["input"]
            batch_output = batch_data["output"]
            predictions = network.apply(params, batch_input, training=True)
            return jnp.mean((batch_output - predictions) ** 2)

        return mse_loss

    else:
        raise ValueError(f"Unknown task_type: {task_type}")
def create_learning_rate_schedule(
    schedule_name: str = "cosine",
    base_learning_rate: float = 1e-3,
    num_epochs: int = 100,
    num_steps_per_epoch: int = 100,
    **schedule_args,
) -> optax.Schedule:
    """
    Create learning rate schedule.

    Args:
        schedule_name: Name of the schedule ('cosine', 'exponential', 'constant')
        base_learning_rate: Initial learning rate
        num_epochs: Total number of training epochs
        num_steps_per_epoch: Number of steps per epoch
        **schedule_args: Additional arguments for the schedule

    Returns:
        Learning rate schedule
    """
    total_steps = num_epochs * num_steps_per_epoch

    if schedule_name == "cosine":
        # Filter out unsupported arguments for cosine_decay_schedule
        valid_args = {
            k: v for k, v in schedule_args.items() if k in ["alpha", "exponent"]
        }
        return optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=total_steps, **valid_args
        )
    elif schedule_name == "exponential":
        return optax.exponential_decay(
            init_value=base_learning_rate,
            transition_steps=total_steps // 4,
            decay_rate=schedule_args.get("decay_rate", 0.9),
        )
    elif schedule_name == "constant":
        return optax.constant_schedule(base_learning_rate)
    elif schedule_name == "reduce_on_plateau":
        # For reduce_on_plateau, return None as it's handled separately
        return None
    else:
        raise ValueError(f"Unknown schedule type: {schedule_name}")


def create_optimizer(
    learning_rate: Union[float, optax.Schedule],  # Accepte un float ou un schedule
    optimizer_type: str = "adam",
    weight_decay: float = 0.0,
    **kwargs,
) -> optax.GradientTransformation:
    """
    Create optimizer with specified configuration.

    Args:
        learning_rate: Learning rate or schedule
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
        weight_decay: Weight decay factor
        **kwargs: Additional optimizer arguments

    Returns:
        Configured optimizer
    """
    if optimizer_type == "adam":
        return optax.adam(learning_rate, **kwargs)
    elif optimizer_type == "sgd":
        return optax.sgd(learning_rate, **kwargs)
    elif optimizer_type == "adamw":
        return optax.adamw(learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class TrainingResult(NamedTuple):
    """Base result class for all training tasks."""

    params: Any
    config: NNConfig
    training_history: Dict[str, Any]
    network: Any
    stored_phi: Optional[jnp.ndarray] = None



def train_nn_old(
    key: jax.random.PRNGKey,
    config: NNConfig,
    io_generator: Callable[
        [jax.random.PRNGKey, int],
        Union[
            Tuple[jnp.ndarray, jnp.ndarray],
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ],
    ],
    network: Optional[Any] = None,
    val_io_generator: Optional[Callable] = None,
) -> TrainingResult:
    """
    Unified training function for neural networks.

    Args:
        key: JAX random key for initialization
        config: NNConfig containing all training parameters
        io_generator: Function that generates training data
                     - For classifier: (key, batch_size) -> (x, phi, s(x), labels)
                     - For summary_learner: (key, batch_size) -> (x, phi)
        network: Optional pre-initialized network. If None, will be created from config.
        val_io_generator: Optional validation data generator. If None, uses io_generator.

    Returns:
        Training result with trained parameters and inference function
    """
    training_config = config.training
    task_type = config.task_type

    # Extract training parameters
    batch_size = training_config.batch_size
    n_samples_per_epoch = training_config.n_samples_per_epoch
    num_epochs = training_config.num_epochs
    verbose = training_config.verbose
    if n_samples_per_epoch is None:
        raise ValueError("n_samples_per_epoch must be specified in the config")
    # Calculate number of batches per epoch
    n_batch = n_samples_per_epoch // batch_size

    if verbose:
        logger.info(f"Starting {task_type} training")
        logger.info(
            f"Epochs: {num_epochs}, Batch size: {batch_size}, Batches/epoch: {n_batch}"
        )
        

    # Generate small sample for network initialization
    key, subkey = jax.random.split(key)
    init_data = io_generator(subkey, 10)
    
    total_simulations = init_data["n_simulations"]
    total_sample = 10
    
    init_input = init_data["input"]
    init_output = init_data["output"] 
    

    # Create network if not provided
    if network is None:
        from .registry import create_network_from_config

        network = create_network_from_config(
            network_config=config.network, task_type=task_type
        )
        
    params = network.init(subkey, init_input)

    if val_io_generator is None:
        val_io_generator = io_generator

    # Generate validation data (fixed set for consistent evaluation)
    val_batch_size = min(batch_size * 10, 1024)  # Larger validation batches
    key, val_key = jax.random.split(key)
    val_data = val_io_generator(val_key, val_batch_size)

    total_simulations += val_data["n_simulations"]
    total_sample += val_batch_size
    
    # Setup learning rate schedule
    lr_schedule = create_learning_rate_schedule(
        schedule_name=training_config.lr_scheduler.schedule_name,
        base_learning_rate=training_config.learning_rate,
        num_epochs=num_epochs,
        num_steps_per_epoch=n_batch,
        **training_config.lr_scheduler.schedule_args,
    )

    # Create optimizer
    if lr_schedule is not None:
        optimizer_lr = lr_schedule
        current_lr = training_config.learning_rate
    else:
        optimizer_lr = training_config.learning_rate
        current_lr = training_config.learning_rate

    optimizer = create_optimizer(
        learning_rate=optimizer_lr,
        optimizer_type=training_config.optimizer,
        weight_decay=training_config.weight_decay,
    )
    opt_state = optimizer.init(params)

    # Create loss function
    loss_fn = create_loss_function(task_type, network)

    # Create validation evaluation function
    @jax.jit
    def evaluate_val(params, val_data):
        """Evaluate validation loss and metrics."""
        val_loss = loss_fn(params, val_data)
        
        val_output = val_data["output"]
        val_input = val_data["input"]

        
        if task_type == "classifier":
            logits = network.apply(params, val_input, training=False)
            probs = jax.nn.sigmoid(logits.squeeze(-1))
            predictions = (probs > 0.5).astype(jnp.float32)
            accuracy = jnp.mean(predictions == val_output)
            return val_loss, {"accuracy": accuracy}
        else:
            return val_loss, {}

    # Create training step function
    def create_train_step(current_optimizer):
        @jax.jit
        def train_step(params, opt_state, batch_data):
            loss, grads = jax.value_and_grad(loss_fn)(params, batch_data)
            updates, opt_state = current_optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        return train_step

    train_step = create_train_step(optimizer)

    # Training state for reduce_on_plateau
    best_loss = float("inf")
    lr_plateau_counter = 0
    step_counter = 0

    # Training metrics
    metrics = {
        "learning_rate": [],
        "total_simulations": [],
        "train_loss": [],
        "val_loss": [],
    }

    if task_type == "classifier":
        metrics["train_accuracy"] = []
        metrics["val_accuracy"] = []

    # Training loop
    start_time = time.time()
    if getattr(training_config, "n_phi_to_store", 0) > 0:
        stored_phi = np.array([])

    for epoch in range(num_epochs):
        # Check sample stopping rule BEFORE starting the epoch
        if (
            hasattr(training_config, "stopping_rules")
            and training_config.stopping_rules
        ):
            stopping_rules = training_config.stopping_rules

            # Check if we have a StoppingRulesConfig object or a dict
            if hasattr(stopping_rules, "sample_stopping"):
                sample_stopping = stopping_rules.sample_stopping
            elif isinstance(stopping_rules, dict):
                sample_stopping_dict = stopping_rules.get("sample_stopping", {})

                # Convert dict to object-like access
                class StoppingConfig:
                    def __init__(self, config_dict):
                        self.enabled = config_dict.get("enabled", False)
                        self.max_samples = config_dict.get("max_samples")

                sample_stopping = StoppingConfig(sample_stopping_dict)
            else:
                sample_stopping = None

            # Check if the next epoch would exceed the sample limit
            if (
                sample_stopping is not None
                and sample_stopping.enabled
                and sample_stopping.max_samples is not None
            ):
                next_epoch_total_samples = (epoch + 1) * n_samples_per_epoch

                if next_epoch_total_samples > sample_stopping.max_samples:
                    if verbose:
                        logger.info(
                            f"Sample stopping before epoch {epoch + 1}: "
                            f"Next epoch would use {next_epoch_total_samples} samples, "
                            f"exceeding max_samples {sample_stopping.max_samples}"
                        )
                    break

        epoch_losses = []

        for batch_idx in range(n_batch):
            # Generate batch data
            key, subkey = jax.random.split(key)
            batch_data = io_generator(subkey, batch_size)
            if getattr(training_config, "n_phi_to_store", 0) > 0:
                if stored_phi.shape[0] < training_config.n_phi_to_store:
                    phis = get_phi_from_batch(batch_data)
                    stored_phi = np.append(stored_phi, phis, axis=0)
                    

            # Training step
            params, opt_state, loss = train_step(params, opt_state, batch_data)
            epoch_losses.append(loss)
            step_counter += 1
            total_sample += batch_size
            total_simulations += batch_data["n_simulations"]
        # Calculate mean loss for the epoch
        if epoch_losses:
            mean_loss = jnp.mean(jnp.array(epoch_losses))
            metrics["train_loss"].append(float(mean_loss))

            # Evaluate on validation set
            val_loss, val_metrics = evaluate_val(params, val_data)
            metrics["val_loss"].append(float(val_loss))

            if task_type == "classifier":
                metrics["val_accuracy"].append(float(val_metrics["accuracy"]))
                # You can also compute train accuracy if needed
                train_val_loss, train_val_metrics = evaluate_val(params, batch_data)
                metrics["train_accuracy"].append(float(train_val_metrics["accuracy"]))

            # Handle reduce_on_plateau logic
            if training_config.lr_scheduler.schedule_name == "reduce_on_plateau":
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    lr_plateau_counter = 0
                else:
                    lr_plateau_counter += 1

                # Check if we should reduce LR
                lr_patience = training_config.lr_scheduler.schedule_args.get(
                    "patience", 5
                )
                lr_factor = training_config.lr_scheduler.schedule_args.get(
                    "factor", 0.5
                )
                min_lr = training_config.lr_scheduler.schedule_args.get("min_lr", 1e-8)

                if lr_plateau_counter >= lr_patience:
                    old_lr = current_lr
                    new_lr = current_lr * lr_factor

                    if new_lr < min_lr:
                        if current_lr <= min_lr * 1.001:
                            # Early stopping - already at minimum LR
                            if verbose:
                                logger.info(
                                    f"Early stopping at epoch {epoch + 1}: LR reached minimum"
                                )
                            break
                        else:
                            current_lr = min_lr
                    else:
                        current_lr = new_lr

                    # Recreate optimizer with new LR
                    optimizer = create_optimizer(
                        learning_rate=current_lr,
                        optimizer_type=training_config.optimizer,
                        weight_decay=training_config.weight_decay,
                    )
                    opt_state = optimizer.init(params)
                    train_step = create_train_step(optimizer)
                    lr_plateau_counter = 0

                    if verbose:
                        logger.info(
                            f"Learning rate reduced: {old_lr:.6f} → {current_lr:.6f}"
                        )

            # Get current effective LR for logging
            if (
                lr_schedule is not None
                and training_config.lr_scheduler.schedule_name != "reduce_on_plateau"
            ):
                effective_lr = lr_schedule(step_counter)
            else:
                effective_lr = current_lr

            metrics["learning_rate"].append(float(effective_lr))
            metrics["total_simulations"].append(total_simulations)

            # Check new stopping rules: sample_stopping and simulation_stopping
            if (
                hasattr(training_config, "stopping_rules")
                and training_config.stopping_rules
            ):
                stopping_rules = training_config.stopping_rules

                # Check if we have a StoppingRulesConfig object or a dict
                if hasattr(stopping_rules, "sample_stopping"):
                    sample_stopping = stopping_rules.sample_stopping
                    simulation_stopping = stopping_rules.simulation_stopping
                elif isinstance(stopping_rules, dict):
                    sample_stopping = stopping_rules.get("sample_stopping", {})
                    simulation_stopping = stopping_rules.get("simulation_stopping", {})

                    # Convert dict to object-like access
                    class StoppingConfig:
                        def __init__(self, config_dict):
                            self.enabled = config_dict.get("enabled", False)
                            self.max_samples = config_dict.get("max_samples")
                            self.max_simulations = config_dict.get("max_simulations")

                    sample_stopping = StoppingConfig(sample_stopping)
                    simulation_stopping = StoppingConfig(simulation_stopping)

                # Check sample stopping rule
                if sample_stopping.enabled and sample_stopping.max_samples is not None:
                    total_samples = (epoch + 1) * n_samples_per_epoch
                    if total_samples >= sample_stopping.max_samples:
                        if verbose:
                            logger.info(
                                f"Sample stopping at epoch {epoch + 1}: "
                                f"Total samples {total_samples} >= max_samples {sample_stopping.max_samples}"
                            )
                        break

                # Check simulation stopping rule
                if (
                    simulation_stopping.enabled
                    and simulation_stopping.max_simulations is not None
                ):
                    if total_simulations >= simulation_stopping.max_simulations:
                        if verbose:
                            logger.info(
                                f"Simulation stopping at epoch {epoch + 1}: "
                                f"Total simulations {total_simulations} >= max_simulations {simulation_stopping.max_simulations}"
                            )
                        break

            # Progress logging
            if verbose and epoch % 10 == 0:
                elapsed_time = time.time() - start_time
                log_msg = (
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {mean_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                )

                if task_type == "classifier":
                    train_acc = metrics["train_accuracy"][-1]
                    val_acc = metrics["val_accuracy"][-1]
                    log_msg += f"Train Acc: {train_acc:.0%} | Val Acc: {val_acc:.0%} | "

                log_msg += (
                    f"LR: {effective_lr:.6f} | "
                    f"Simulations: {total_simulations} | "
                    f"Samples: {total_sample} | "
                    f"Time: {elapsed_time:.1f}s"
                )
                logger.info(log_msg)

    # Final statistics
    final_loss = metrics["train_loss"][-1] if metrics["train_loss"] else float("inf")
    total_time = time.time() - start_time

    if verbose:
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Final train loss: {final_loss:.6f}")
        if metrics["val_loss"]:
            final_val_loss = metrics["val_loss"][-1]
            logger.info(f"Final val loss: {final_val_loss:.6f}")
        if task_type == "classifier" and metrics["val_accuracy"]:
            final_val_acc = metrics["val_accuracy"][-1]
            logger.info(f"Final val accuracy: {final_val_acc:.0%}")
        logger.info(f"Total simulations: {total_simulations}")

    

    # Prepare training history
    training_history = {
        "final_loss": final_loss,
        "total_simulations": total_simulations,
        "total_time": total_time,
        # "metrics": metrics,
    }
    if getattr(training_config, "n_phi_to_store", 0) > 0:
        print("Returning stored phi values")
        # Filter out None values before concatenation
        return TrainingResult(
            params=params,
            network=network,
            training_history=training_history,
            config=config,
            stored_phi=stored_phi
        )
    
    return TrainingResult(
        params = params, 
        network = network,
        training_history= training_history,
        config = config,
    )



# Convenience functions for backward compatibility
def train_classifier_old(
    key: jax.random.PRNGKey,
    config: NNConfig,
    io_generator: Callable,
    network: Optional[Any] = None,
    val_io_generator: Optional[Callable] = None,
) -> TrainingResult:
    """
    Train a classifier network.

    Args:
        key: JAX random key for initialization
        config: NNConfig with task_type="classifier"
        io_generator: Function that generates training data (theta, x, labels)
        network: Optional pre-initialized network
        val_io_generator: Optional validation data generator

    Returns:
        TrainingResult with trained parameters and classifier function
    """
    if config.task_type != "classifier":
        raise ValueError("config.task_type must be 'classifier' for train_classifier")

    return train_nn_old(key, config, io_generator, network, val_io_generator)


def train_summary_learner_old(
    key: jax.random.PRNGKey,
    config: NNConfig,
    io_generator: Callable,
    network: Optional[Any] = None,
    val_io_generator: Optional[Callable] = None,
) -> TrainingResult:
    """
    Train a summary learner network.

    Args:
        key: JAX random key for initialization
        config: NNConfig with task_type="summary_learner"
        io_generator: Function that generates training data (x, phi)
        network: Optional pre-initialized network
        val_io_generator: Optional validation data generator

    Returns:
        TrainingResult with trained parameters and summary function
    """
    if config.task_type != "summary_learner":
        raise ValueError(
            "config.task_type must be 'summary_learner' for train_summary_learner"
        )

    return train_nn_old(key, config, io_generator, network, val_io_generator)
