import optax
import jax
import jax.numpy as jnp
from typing import Callable, Union
import numpy as np


def create_loss_function(
    task_type: str, network, loss_name: str = "default", loss_args: dict = None
) -> Callable:
    """Create the appropriate loss function for the task type.

    Args:
        task_type: "classifier" or "regressor"
        network: Neural network instance
        loss_name: Name of loss function ("default", "bce", "focal", "label_smoothing", "mse", "huber", "mae", etc.)
        loss_args: Additional arguments for the loss function

    Returns:
        Loss function compatible with training loop
    """
    import logging

    logger = logging.getLogger(__name__)

    if loss_args is None:
        loss_args = {}

    if task_type == "classifier":
        # Determine which classifier loss to use
        if loss_name == "default" or loss_name == "bce":
            return _create_binary_crossentropy_loss(network)
        elif loss_name == "focal":
            alpha = loss_args.get("alpha", 0.25)
            gamma = loss_args.get("gamma", 2.0)
            logger.info(f"Using Focal Loss with alpha={alpha}, gamma={gamma}")
            return _create_focal_loss(network, alpha, gamma)
        elif loss_name == "label_smoothing":
            epsilon = loss_args.get("epsilon", 0.1)
            logger.info(f"Using Label Smoothing BCE with epsilon={epsilon}")
            return _create_label_smoothing_bce(network, epsilon)
        else:
            raise ValueError(f"Unknown classifier loss: {loss_name}")

    elif task_type == "regressor":
        # Determine which regression loss to use
        if loss_name == "default" or loss_name == "mse":
            return _create_mse_loss(network)
        elif loss_name == "huber":
            delta = loss_args.get("delta", 1.0)
            logger.info(f"Using Huber Loss with delta={delta}")
            return _create_huber_loss(network, delta)
        elif loss_name == "mae":
            logger.info("Using Mean Absolute Error Loss")
            return _create_mae_loss(network)
        elif loss_name == "pinball":
            tau = loss_args.get("tau", 0.5)
            logger.info(f"Using Pinball Loss (Quantile Loss) with tau={tau}")
            return _create_pinball_loss(network, tau)
        else:
            raise ValueError(f"Unknown regressor loss: {loss_name}")
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def _create_binary_crossentropy_loss(network) -> Callable:
    """Standard Binary Cross-Entropy loss for classification."""

    def binary_crossentropy_loss(params, batch_data, rng_key=None):
        batch_input = batch_data["input"]
        batch_output = batch_data["output"]
        # Use RNG key for dropout if provided, and set training mode accordingly
        if rng_key is not None:
            rngs = {"dropout": rng_key}
            training = True
        else:
            rngs = None
            training = False
        logits = network.apply(params, batch_input, training=training, rngs=rngs)
        # Convert to probabilities and compute BCE
        probs = jax.nn.sigmoid(logits.squeeze(-1))
        # Clip to avoid log(0)
        probs = jnp.clip(probs, 1e-7, 1 - 1e-7)
        bce = -jnp.mean(
            batch_output * jnp.log(probs) + (1 - batch_output) * jnp.log(1 - probs)
        )
        return bce

    return binary_crossentropy_loss


def _create_focal_loss(network, alpha: float = 0.25, gamma: float = 2.0) -> Callable:
    """Focal Loss for handling class imbalance."""

    def focal_loss(params, batch_data, rng_key=None):
        batch_input = batch_data["input"]
        batch_output = batch_data["output"]
        if rng_key is not None:
            rngs = {"dropout": rng_key}
            training = True
        else:
            rngs = None
            training = False
        logits = network.apply(params, batch_input, training=training, rngs=rngs)
        probs = jax.nn.sigmoid(logits.squeeze(-1))
        probs = jnp.clip(probs, 1e-7, 1 - 1e-7)

        # Focal loss computation
        ce = -(batch_output * jnp.log(probs) + (1 - batch_output) * jnp.log(1 - probs))
        p_t = jnp.where(batch_output == 1, probs, 1 - probs)
        alpha_t = jnp.where(batch_output == 1, alpha, 1 - alpha)
        focal = alpha_t * (1 - p_t) ** gamma * ce
        return jnp.mean(focal)

    return focal_loss


def _create_label_smoothing_bce(network, epsilon: float = 0.1) -> Callable:
    """Binary Cross-Entropy with Label Smoothing."""

    def label_smoothing_bce(params, batch_data, rng_key=None):
        batch_input = batch_data["input"]
        batch_output = batch_data["output"]
        if rng_key is not None:
            rngs = {"dropout": rng_key}
            training = True
        else:
            rngs = None
            training = False
        logits = network.apply(params, batch_input, training=training, rngs=rngs)
        probs = jax.nn.sigmoid(logits.squeeze(-1))
        probs = jnp.clip(probs, 1e-7, 1 - 1e-7)

        # Apply label smoothing
        targets_smooth = batch_output * (1 - epsilon) + 0.5 * epsilon
        bce_smooth = -jnp.mean(
            targets_smooth * jnp.log(probs) + (1 - targets_smooth) * jnp.log(1 - probs)
        )
        return bce_smooth

    return label_smoothing_bce


def _create_mse_loss(network) -> Callable:
    """Standard Mean Squared Error loss for regression."""

    def mse_loss(params, batch_data, rng_key=None):
        batch_input = batch_data["input"]
        batch_output = batch_data["output"]
        # Use RNG key for dropout if provided, and set training mode accordingly
        if rng_key is not None:
            rngs = {"dropout": rng_key}
            training = True
        else:
            rngs = None
            training = False
        predictions = network.apply(params, batch_input, training=training, rngs=rngs)
        return jnp.mean((batch_output - predictions) ** 2)

    return mse_loss


def _create_huber_loss(network, delta: float = 1.0) -> Callable:
    """Huber Loss for robust regression."""

    def huber_loss(params, batch_data, rng_key=None):
        batch_input = batch_data["input"]
        batch_output = batch_data["output"]
        if rng_key is not None:
            rngs = {"dropout": rng_key}
            training = True
        else:
            rngs = None
            training = False
        predictions = network.apply(params, batch_input, training=training, rngs=rngs)
        residual = jnp.abs(batch_output - predictions)
        condition = residual <= delta
        squared_loss = 0.5 * residual**2
        linear_loss = delta * residual - 0.5 * delta**2
        return jnp.mean(jnp.where(condition, squared_loss, linear_loss))

    return huber_loss


def _create_mae_loss(network) -> Callable:
    """Mean Absolute Error loss for regression."""

    def mae_loss(params, batch_data, rng_key=None):
        batch_input = batch_data["input"]
        batch_output = batch_data["output"]
        if rng_key is not None:
            rngs = {"dropout": rng_key}
            training = True
        else:
            rngs = None
            training = False
        predictions = network.apply(params, batch_input, training=training, rngs=rngs)
        return jnp.mean(jnp.abs(batch_output - predictions))

    return mae_loss


def _create_pinball_loss(network, tau: float = 0.5) -> Callable:
    """Pinball Loss (Quantile Loss) for quantile regression.

    Args:
        network: Neural network
        tau: Quantile level (0 < tau < 1). tau=0.5 gives median regression.
    """

    def pinball_loss(params, batch_data, rng_key=None):
        batch_input = batch_data["input"]
        batch_output = batch_data["output"]
        if rng_key is not None:
            rngs = {"dropout": rng_key}
            training = True
        else:
            rngs = None
            training = False
        predictions = network.apply(params, batch_input, training=training, rngs=rngs)
        residual = batch_output - predictions
        # Pinball loss: tau * max(residual, 0) + (1 - tau) * max(-residual, 0)
        positive_part = jnp.maximum(residual, 0)
        negative_part = jnp.maximum(-residual, 0)
        loss = tau * positive_part + (1 - tau) * negative_part
        return jnp.mean(loss)

    return pinball_loss


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
    import logging

    logger = logging.getLogger(__name__)

    if optimizer_type == "adam":
        if weight_decay > 0.0:
            logger.info(
                f"Adam optimizer with weight_decay={weight_decay} detected. "
                f"Automatically switching to AdamW for proper weight decay support."
            )
            return optax.adamw(learning_rate, weight_decay=weight_decay, **kwargs)
        else:
            return optax.adam(learning_rate, **kwargs)
    elif optimizer_type == "sgd":
        if weight_decay > 0.0:
            logger.info(
                f"Using SGD with weight_decay={weight_decay} via L2 regularization."
            )
            return optax.chain(
                optax.add_decayed_weights(weight_decay),
                optax.sgd(learning_rate, **kwargs),
            )
        else:
            return optax.sgd(learning_rate, **kwargs)
    elif optimizer_type == "adamw":
        if weight_decay > 0.0:
            logger.info(f"Using AdamW with native weight_decay={weight_decay} support.")
        return optax.adamw(learning_rate, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
