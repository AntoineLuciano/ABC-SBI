import optax
import jax
import jax.numpy as jnp
from typing import Callable, Union
import numpy as np


def create_loss_function(task_type: str, network) -> Callable:
    """Create the appropriate loss function for the task type."""

    if task_type == "classifier":

        def binary_crossentropy_loss(params, batch_data, rng_key=None):
            batch_input = batch_data["input"]
            batch_output = batch_data["output"]
            # Use RNG key for dropout if provided
            rngs = {"dropout": rng_key} if rng_key is not None else None
            logits = network.apply(params, batch_input, training=True, rngs=rngs)
            # Convert to probabilities and compute BCE
            probs = jax.nn.sigmoid(logits.squeeze(-1))
            # Clip to avoid log(0)
            probs = jnp.clip(probs, 1e-7, 1 - 1e-7)
            bce = -jnp.mean(
                batch_output * jnp.log(probs) + (1 - batch_output) * jnp.log(1 - probs)
            )
            return bce

        return binary_crossentropy_loss

    elif task_type == "summary_learner":

        def mse_loss(params, batch_data, rng_key=None):
            batch_input = batch_data["input"]
            batch_output = batch_data["output"]
            # Use RNG key for dropout if provided
            rngs = {"dropout": rng_key} if rng_key is not None else None
            predictions = network.apply(params, batch_input, training=True, rngs=rngs)
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
