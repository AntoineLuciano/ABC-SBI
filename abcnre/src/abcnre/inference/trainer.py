"""
Training utilities for neural ratio estimation.

This module provides JAX-based training functions and utilities
for training neural networks for ABC inference.
"""

from typing import Dict, Any, Callable, Tuple, Union
import jax
import jax.numpy as jnp
from jax import random
import optax
import flax.linen as nn
from flax.training import train_state
from flax.core import freeze, unfreeze
import numpy as np


class TrainingState(train_state.TrainState):
    """
    Training state for neural ratio estimation.
    
    Extended TrainState with additional fields for NRE training.
    """
    key: random.PRNGKey
    batch_stats: Dict[str, Any] = None
    
    def __call__(self, *args, **kwargs):
        """Make TrainingState callable for convenience."""
        # Gérer les batch_stats correctement
        if self.batch_stats is not None:
            variables = {'params': self.params, 'batch_stats': self.batch_stats}
        else:
            variables = {'params': self.params}
        
        return self.apply_fn(variables, *args, **kwargs)


def binary_cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute binary cross-entropy loss.
    
    Args:
        logits: Network outputs of shape (batch_size, 1)
        labels: Binary labels of shape (batch_size,)
        
    Returns:
        Scalar loss value
    """
    # Ensure correct shapes
    logits = logits.flatten()
    labels = labels.flatten()
    
    # Compute binary cross-entropy with logits
    loss = optax.sigmoid_binary_cross_entropy(logits, labels)
    return jnp.mean(loss)


def compute_accuracy(
    logits: jnp.ndarray,
    labels: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute classification accuracy.
    
    Args:
        logits: Network outputs of shape (batch_size, 1)
        labels: Binary labels of shape (batch_size,)
        
    Returns:
        Scalar accuracy value
    """
    # Ensure correct shapes
    logits = logits.flatten()
    labels = labels.flatten()
    
    # Compute predictions
    predictions = nn.sigmoid(logits) > 0.5
    
    # Compute accuracy
    accuracy = jnp.mean(predictions == labels)
    return accuracy


def compute_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """
    Compute training metrics.
    
    Args:
        logits: Network outputs
        labels: True labels
        
    Returns:
        Dictionary of metrics
    """
    loss = binary_cross_entropy_loss(logits, labels)
    accuracy = compute_accuracy(logits, labels)
    
    return {
        'loss': loss,
        'accuracy': accuracy
    }


@jax.jit
def train_step(
    state: TrainingState,
    features: jnp.ndarray,
    labels: jnp.ndarray
) -> Tuple[TrainingState, Dict[str, Any]]:
    """
    Performs a single training step, properly handling RNGs for dropout.
    """
    # 1. Split the PRNG key. One part is used for dropout in this step,
    # and the other is stored back in the state for the next step.
    dropout_key, new_key = random.split(state.key)

    def loss_fn(params):
        variables = {'params': params}
        if state.batch_stats:
            variables['batch_stats'] = state.batch_stats

        # 2. Pass the `dropout_key` to the apply function via the `rngs` argument.
        # This is the standard Flax pattern for handling stochastic layers.
        result = state.apply_fn(
            variables,
            features,
            training=True,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_key} # This is the critical addition
        )

        if isinstance(result, tuple) and len(result) == 2:
            logits, updated_variables = result
            updated_batch_stats = updated_variables.get('batch_stats')
        else:
            logits = result
            updated_batch_stats = None
        
        loss = binary_cross_entropy_loss(logits, labels)
        return loss, (logits, updated_batch_stats)

    # Compute gradients
    (loss, (logits, updated_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Update parameters and batch stats
    new_state = state.apply_gradients(grads=grads)
    if updated_batch_stats is not None:
        new_state = new_state.replace(batch_stats=updated_batch_stats)
    
    # 3. Update the state with the new key for the next training iteration.
    new_state = new_state.replace(key=new_key)
    
    # Compute metrics for this step
    metrics = compute_metrics(logits, labels)
    
    return new_state, metrics



@jax.jit
def evaluate_step(
    state: TrainingState,
    features: jnp.ndarray,
    labels: jnp.ndarray
) -> Dict[str, Any]:
    """
    Perform a single evaluation step.
    
    Args:
        state: Current training state
        features: Input features
        labels: Target labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Créer les variables correctement
    if state.batch_stats is not None:
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
    else:
        variables = {'params': state.params}
    
    logits = state.apply_fn(variables, features, training=False)
    metrics = compute_metrics(logits, labels)
    
    return metrics


def create_learning_rate_schedule(
    base_learning_rate: float,
    num_epochs: int,
    decay_type: str = 'cosine'
) -> optax.Schedule:
    """
    Create learning rate schedule.
    
    Args:
        base_learning_rate: Initial learning rate
        num_epochs: Total number of training epochs
        decay_type: Type of decay ('cosine', 'exponential', 'constant')
        
    Returns:
        Learning rate schedule
    """
    if decay_type == 'cosine':
        return optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=num_epochs
        )
    elif decay_type == 'exponential':
        return optax.exponential_decay(
            init_value=base_learning_rate,
            transition_steps=num_epochs // 4,
            decay_rate=0.9
        )
    elif decay_type == 'constant':
        return optax.constant_schedule(base_learning_rate)
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")
    
    

def create_optimizer(
    learning_rate: Union[float, optax.Schedule], # Accepte un float ou un schedule
    optimizer_type: str = 'adam',
    weight_decay: float = 0.0,
    **kwargs
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
    if optimizer_type == 'adam':
        return optax.adam(learning_rate, **kwargs)
    elif optimizer_type == 'sgd':
        return optax.sgd(learning_rate, **kwargs)
    elif optimizer_type == 'adamw':
        return optax.adamw(learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class EarlyStopping:
    """
    Early stopping callback for training.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        restore_best_weights: Whether to restore best weights
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.best_params = None
        self.best_batch_stats = None
        self.wait = 0
        self.stopped_epoch = 0
    
    def __call__(
        self,
        epoch: int,
        val_loss: float,
        state: TrainingState
    ) -> Tuple[bool, TrainingState]:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch
            val_loss: Current validation loss
            state: Current training state
            
        Returns:
            Tuple of (should_stop, updated_state)
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_params = state.params
                self.best_batch_stats = state.batch_stats
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_params is not None:
                    state = state.replace(
                        params=self.best_params,
                        batch_stats=self.best_batch_stats
                    )
                return True, state
        
        return False, state


def train_with_validation(
    state: TrainingState,
    train_data_generator: Callable,
    val_data_generator: Callable,
    num_epochs: int,
    batch_size: int,
    early_stopping: EarlyStopping = None,
    verbose: bool = True
) -> Tuple[TrainingState, Dict[str, Any]]:
    """
    Train model with validation data.
    
    Args:
        state: Initial training state
        train_data_generator: Function to generate training batches
        val_data_generator: Function to generate validation batches
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        early_stopping: Early stopping callback
        verbose: Whether to print progress
        
    Returns:
        Trained state and training history
    """
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(num_epochs):
        # Training step
        train_batch = train_data_generator(batch_size)
        state, train_metrics = train_step(
            state, train_batch.features, train_batch.labels
        )
        
        # Validation step
        val_batch = val_data_generator(batch_size)
        val_metrics = evaluate_step(
            state, val_batch.features, val_batch.labels
        )
        
        # Update history
        history['train_loss'].append(float(train_metrics['loss']))
        history['train_accuracy'].append(float(train_metrics['accuracy']))
        history['val_loss'].append(float(val_metrics['loss']))
        history['val_accuracy'].append(float(val_metrics['accuracy']))
        
        # Early stopping check
        if early_stopping is not None:
            should_stop, state = early_stopping(epoch, val_metrics['loss'], state)
            if should_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
    
    return state, history


def compute_calibration_metrics(
    predicted_probs: jnp.ndarray,
    true_labels: jnp.ndarray,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics for the trained model.
    
    Args:
        predicted_probs: Predicted probabilities
        true_labels: True binary labels
        num_bins: Number of bins for calibration
        
    Returns:
        Dictionary of calibration metrics
    """
    # Convert to numpy for easier manipulation
    predicted_probs = np.array(predicted_probs)
    true_labels = np.array(true_labels)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Compute calibration metrics
    ece = 0.0  # Expected Calibration Error
    mce = 0.0  # Maximum Calibration Error
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Compute accuracy and confidence in this bin
            accuracy_in_bin = true_labels[in_bin].mean()
            avg_confidence_in_bin = predicted_probs[in_bin].mean()
            
            # Update calibration errors
            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * calibration_error
            mce = max(mce, calibration_error)
    
    return {
        'expected_calibration_error': ece,
        'maximum_calibration_error': mce
    }


def compute_ratio_statistics(
    log_ratios: jnp.ndarray
) -> Dict[str, float]:
    """
    Compute statistics for estimated log ratios.
    
    Args:
        log_ratios: Estimated log likelihood ratios
        
    Returns:
        Dictionary of ratio statistics
    """
    ratios = jnp.exp(log_ratios)
    
    return {
        'log_ratio_mean': float(jnp.mean(log_ratios)),
        'log_ratio_std': float(jnp.std(log_ratios)),
        'log_ratio_min': float(jnp.min(log_ratios)),
        'log_ratio_max': float(jnp.max(log_ratios)),
        'ratio_mean': float(jnp.mean(ratios)),
        'ratio_std': float(jnp.std(ratios)),
        'ratio_min': float(jnp.min(ratios)),
        'ratio_max': float(jnp.max(ratios))
    }