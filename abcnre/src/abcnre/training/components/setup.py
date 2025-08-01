"""
Setup and validation functions for training components in the ABC-NRE framework.

This module handles all setup and initialization logic with proper validation
using the sophisticated config.py structure.
"""

import jax
import jax.numpy as jnp
import logging
import optax
from typing import Dict, Any, Callable, Optional, Union

# Import from your existing modules
from ..config import NNConfig
from ..optimization import (
    create_loss_function,
    create_learning_rate_schedule,
    create_optimizer,
)

logger = logging.getLogger(__name__)


def validate_training_config(config: NNConfig):
    """
    Validate training configuration parameters using proper config structure.

    Args:
        config: NNConfig instance with complete configuration

    Raises:
        ValueError: If configuration is invalid
    """
    training_config = config.training

    # Basic parameter validation
    if training_config.n_samples_per_epoch is None:
        raise ValueError("n_samples_per_epoch must be specified in the config")

    if training_config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if training_config.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")

    if training_config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if not (0 <= training_config.validation_split <= 1):
        raise ValueError("validation_split must be between 0 and 1")

    # Check if n_samples_per_epoch is divisible by batch_size
    if training_config.n_samples_per_epoch % training_config.batch_size != 0:
        logger.warning(
            f"n_samples_per_epoch ({training_config.n_samples_per_epoch}) "
            f"not divisible by batch_size ({training_config.batch_size}). "
            f"Some samples may be unused."
        )

    # Validate task type
    if config.task_type not in ["classifier", "summary_learner"]:
        raise ValueError(
            f"Unknown task_type: {config.task_type}. Must be 'classifier' or 'summary_learner'"
        )

    # Validate optimizer type
    valid_optimizers = ["adam", "sgd", "adamw"]
    if training_config.optimizer not in valid_optimizers:
        raise ValueError(
            f"Unknown optimizer: {training_config.optimizer}. Must be one of {valid_optimizers}"
        )

    # Validate weight decay
    if training_config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    # Validate phi storage config if present
    n_phi_to_store = getattr(training_config, "n_phi_to_store", 0)
    if n_phi_to_store < 0:
        raise ValueError("n_phi_to_store must be non-negative")

    logger.debug(f"Validated training config for {config.task_type} task")


def setup_training_components(
    key: jax.random.PRNGKey,
    config: NNConfig,
    io_generator: Callable,
    network: Optional[Any] = None,
    val_io_generator: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Setup all training components (network, optimizer, data, etc.)

    Args:
        key: JAX random key for initialization
        config: NNConfig containing all training parameters
        io_generator: Function that generates training data
        network: Optional pre-initialized network
        val_io_generator: Optional validation data generator

    Returns:
        Dictionary with all setup components
    """
    training_config = config.training
    task_type = config.task_type

    # Validate configuration first
    validate_training_config(config)

    # Calculate derived parameters
    batch_size = training_config.batch_size
    n_samples_per_epoch = training_config.n_samples_per_epoch
    num_epochs = training_config.num_epochs
    n_batch = n_samples_per_epoch // batch_size

    if training_config.verbose:
        logger.info(f"Setting up {task_type} training")
        logger.info(
            f"Epochs: {num_epochs}, Batch size: {batch_size}, Batches/epoch: {n_batch}"
        )

    # Generate initialization data
    key, subkey = jax.random.split(key)
    init_data = io_generator(subkey, 10)

    # Validate io_generator output format
    if not isinstance(init_data, dict):
        raise ValueError("io_generator must return a dictionary")

    required_keys = ["input", "output", "n_simulations"]
    for req_key in required_keys:
        if req_key not in init_data:
            raise ValueError(f"io_generator output missing required key: '{req_key}'")

    init_input = init_data["input"]
    init_output = init_data["output"]

    # Create or use provided network
    if network is None:
        from ..registry import create_network_from_config

        network = create_network_from_config(
            network_config=config.network, task_type=task_type
        )
        if training_config.verbose:
            logger.info(f"Created network: {config.network.network_type}")
    else:
        if training_config.verbose:
            logger.info("Using provided pre-initialized network")

    # Initialize parameters
    key, subkey = jax.random.split(key)
    params = network.init(subkey, init_input)

    if training_config.verbose:
        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_flatten(params)[0])
        logger.info(f"Network initialized with {param_count:,} parameters")

    # Setup validation data
    if val_io_generator is None:
        val_io_generator = io_generator
        if training_config.verbose:
            logger.info("Using same generator for validation data")

    val_batch_size = min(batch_size * 10, 1024)  # Larger validation batches
    key, val_key = jax.random.split(key)
    val_data = val_io_generator(val_key, val_batch_size)

    # Validate validation data format
    for req_key in required_keys:
        if req_key not in val_data:
            raise ValueError(
                f"val_io_generator output missing required key: '{req_key}'"
            )

    # Setup learning rate schedule
    lr_schedule = create_learning_rate_schedule(
        schedule_name=training_config.lr_scheduler.schedule_name,
        base_learning_rate=training_config.learning_rate,
        num_epochs=num_epochs,
        num_steps_per_epoch=n_batch,
        **training_config.lr_scheduler.schedule_args,
    )

    # Create optimizer
    optimizer_lr = (
        lr_schedule if lr_schedule is not None else training_config.learning_rate
    )
    current_lr = training_config.learning_rate

    optimizer = create_optimizer(
        learning_rate=optimizer_lr,
        optimizer_type=training_config.optimizer,
        weight_decay=training_config.weight_decay,
    )
    opt_state = optimizer.init(params)

    # Create loss function
    loss_fn = create_loss_function(task_type, network)

    # Create validation function
    evaluate_val = create_validation_function(task_type, network, loss_fn)

    # Create training step function
    train_step = create_training_step_function(optimizer, loss_fn)

    # Initialize counters
    total_simulations = init_data["n_simulations"] + val_data["n_simulations"]
    total_samples = 10 + val_batch_size

    if training_config.verbose:
        logger.info(f"Setup complete. Initial simulations: {total_simulations}")

    return {
        "network": network,
        "params": params,
        "optimizer": optimizer,
        "opt_state": opt_state,
        "lr_schedule": lr_schedule,
        "current_lr": current_lr,
        "loss_fn": loss_fn,
        "evaluate_val": evaluate_val,
        "train_step": train_step,
        "val_data": val_data,
        "total_simulations": total_simulations,
        "total_samples": total_samples,
        "n_batch": n_batch,
        "key": key,
    }


def create_validation_function(task_type: str, network, loss_fn):
    """Create JIT-compiled validation function"""

    @jax.jit
    def evaluate_val(params, val_data):
        """Evaluate validation loss and metrics."""
        # For validation, we don't use dropout (training=False), so rng_key=None
        val_loss = loss_fn(params, val_data, rng_key=None)

        if task_type == "classifier":
            val_input = val_data["input"]
            val_output = val_data["output"]
            logits = network.apply(params, val_input, training=False)
            probs = jax.nn.sigmoid(logits.squeeze(-1))
            predictions = (probs > 0.5).astype(jnp.float32)
            accuracy = jnp.mean(predictions == val_output)
            return val_loss, {"accuracy": accuracy}
        else:
            return val_loss, {}

    return evaluate_val


def create_training_step_function(optimizer, loss_fn):
    """Create JIT-compiled training step function"""

    @jax.jit
    def train_step(params, opt_state, batch_data, rng_key):
        # Generate RNG key for dropout
        dropout_key = jax.random.fold_in(rng_key, 0)
        loss, grads = jax.value_and_grad(lambda p, b: loss_fn(p, b, dropout_key))(
            params, batch_data
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step


def validate_io_generator_compatibility(io_generator: Callable, config: NNConfig):
    """
    Validate that io_generator is compatible with the task type and config.

    Args:
        io_generator: The data generator function
        config: NNConfig with task configuration

    Raises:
        ValueError: If io_generator is incompatible
    """
    try:
        # Test with a small batch
        test_key = jax.random.PRNGKey(0)
        test_data = io_generator(test_key, 4)

        # Check basic format
        if not isinstance(test_data, dict):
            raise ValueError("io_generator must return a dictionary")

        required_keys = ["input", "output", "n_simulations"]
        for key in required_keys:
            if key not in test_data:
                raise ValueError(f"io_generator output missing key: '{key}'")

        # Check data types and shapes
        input_data = test_data["input"]
        output_data = test_data["output"]
        n_sims = test_data["n_simulations"]

        if not isinstance(input_data, (jnp.ndarray, dict)):
            raise ValueError("Input data must be jnp.ndarray or dict")

        if not isinstance(output_data, jnp.ndarray):
            raise ValueError("Output data must be jnp.ndarray")

        if not isinstance(n_sims, (int, jnp.ndarray)):
            raise ValueError("n_simulations must be int or jnp.ndarray")

        # Task-specific validation
        if config.task_type == "classifier":
            if output_data.ndim != 1:
                raise ValueError("Classifier output must be 1D (labels)")
            if not jnp.all((output_data == 0) | (output_data == 1)):
                logger.warning("Classifier output should be binary (0/1)")

        elif config.task_type == "summary_learner":
            if output_data.ndim != 2:
                raise ValueError("Summary learner output must be 2D (batch, features)")

        logger.debug("io_generator validation passed")

    except Exception as e:
        raise ValueError(f"io_generator validation failed: {e}")


# Utility functions
def get_parameter_count(params) -> int:
    """Get total number of parameters in the model."""
    return sum(x.size for x in jax.tree_util.tree_flatten(params)[0])


def log_setup_summary(config: NNConfig, components: Dict[str, Any]):
    """Log a summary of the setup configuration."""
    if not config.training.verbose:
        return

    param_count = get_parameter_count(components["params"])

    logger.info("=" * 50)
    logger.info("TRAINING SETUP SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Task type: {config.task_type}")
    logger.info(f"Network: {config.network.network_type}")
    logger.info(f"Parameters: {param_count:,}")
    logger.info(f"Optimizer: {config.training.optimizer}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"LR scheduler: {config.training.lr_scheduler.schedule_name}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Samples/epoch: {config.training.n_samples_per_epoch}")
    logger.info(f"Max epochs: {config.training.num_epochs}")

    # Phi storage info
    n_phi_to_store = getattr(config.training, "n_phi_to_store", 0)
    if n_phi_to_store > 0:
        logger.info(f"Phi storage: {n_phi_to_store} values")

    logger.info("=" * 50)
