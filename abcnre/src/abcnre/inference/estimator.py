"""
Neural Ratio Estimator for ABC inference.

This module provides the main interface for training neural networks
to estimate likelihood ratios for ABC posterior inference.
"""

import os
import yaml
from typing import Dict, Any, Optional, Callable, Tuple, Union
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import random
import optax
import flax.linen as nn
from flax.training import train_state
import numpy as np

from .networks.base import NetworkBase
from .networks.mlp import MLPNetwork
from .networks.deepset import DeepSetNetwork
from .utils import save_model_config, load_model_config, compute_metrics
from ..simulation.base import ABCTrainingResult
from .trainer import TrainingState, train_step, evaluate_step, create_optimizer,  create_learning_rate_schedule
from .config import TrainingConfig, LRSchedulerConfig
import time
from ..simulation.simulator import ABCSimulator

class NeuralRatioEstimator:
    """
    Neural Ratio Estimator for ABC posterior inference.
    
    This class provides a high-level interface for training neural networks
    to estimate likelihood ratios, which can then be used to approximate
    the posterior distribution in ABC inference.
    
    The estimator trains a binary classifier to distinguish between samples
    from the joint distribution p(x,θ) and the marginal distributions p(x)p(θ).
    The trained classifier's output can be transformed into likelihood ratios
    using: r(x,θ) = σ(f(x,θ)) / (1 - σ(f(x,θ))), where σ is the sigmoid function.
    
    Args:
        network: Neural network architecture
        learning_rate: Learning rate for optimization
        optimizer: Optax optimizer (default: Adam)
        random_seed: Random seed for reproducibility
        
    Example:
        # Create estimator with MLP network
        network = MLPNetwork(hidden_dims=[128, 64, 32])
        estimator = NeuralRatioEstimator(network, learning_rate=1e-3)
        
        # Train on ABC samples
        estimator.train(abc_simulator, num_epochs=100, batch_size=256)
        
        # Estimate posterior
        log_ratios = estimator.log_ratio(features)
        posterior_weights = jnp.exp(log_ratios)
    """
    
    def __init__(
        self,
        network: NetworkBase,
        training_config: TrainingConfig,
        random_seed: int = 42
    ):
        """Initializes the Neural Ratio Estimator."""
        self.network = network
        self.training_config = training_config
        self.random_seed = random_seed
        self.key = random.PRNGKey(random_seed)
        self.accumulated_phi_samples: list = []
        self.total_simulation_count: int = 0
        
        schedule_params = training_config.lr_scheduler
        
        # Create the learning rate schedule object
        if schedule_params.schedule_name != 'reduce_on_plateau':
            num_steps_per_epoch = training_config.n_samples_per_epoch // training_config.batch_size
            lr_schedule = create_learning_rate_schedule(
                schedule_name=schedule_params.schedule_name,
                base_learning_rate=training_config.learning_rate,
                num_epochs=training_config.num_epochs,
                num_steps_per_epoch=num_steps_per_epoch,
                **schedule_params.schedule_args
            )
        else:
            # For 'reduce_on_plateau', the schedule is stateful and managed in the train loop
            # We set it to None here to indicate this.
            lr_schedule = None

        # --- THE FIX ---
        # Store the schedule object as an instance attribute so other methods can access it.
        self.lr_schedule = lr_schedule
        
        # The optimizer needs an initial learning rate (float) or a schedule object.
        optimizer_lr = lr_schedule if lr_schedule is not None else training_config.learning_rate
        self.optimizer = create_optimizer(
            learning_rate=optimizer_lr,
            optimizer_type=training_config.optimizer,
            weight_decay=training_config.weight_decay
        )
        
        self.state: Optional[TrainingState] = None
        self.is_trained = False
        self.training_history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': []
        }
            
    def initialize_training(self, input_shape: Tuple[int, ...]) -> None:
        """
        Initialize training state with network parameters.
        
        Args:
            input_shape: Shape of input features (batch_size, feature_dim)
        """
        self.key, init_key = random.split(self.key)
        
        # Initialize network parameters avec batch_stats
        dummy_input = jnp.ones(input_shape)
        variables = self.network.init(init_key, dummy_input, training=False)
        params = variables['params']
        batch_stats = variables.get('batch_stats', {})
        
        # Créer apply_fn qui gère les batch_stats avec tous les paramètres
        def apply_fn(variables, x, training=True, **kwargs):
            if 'batch_stats' in variables and variables['batch_stats']:
                if training and 'mutable' in kwargs:
                    return self.network.apply(variables, x, training=training, **kwargs)
                else:
                    return self.network.apply(variables, x, training=training)
            else:
                return self.network.apply(variables, x, training=training)
        
        # Create training state
        self.state = TrainingState.create(
            apply_fn=apply_fn,
            params=params,
            tx=self.optimizer,
            key=init_key,
            batch_stats=batch_stats
        )
        
        print(f"Initialized network with {self.network.count_parameters(params):,} parameters")
    
    def train(
        self,
        simulator: 'ABCSimulator', 
        output_dir: Path, 
        num_epochs: int,
        n_samples_per_epoch: int,
        batch_size: int,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Trains the neural ratio estimator.

        Args:
            data_generator: A function that generates training data on demand.
            num_epochs: The number of epochs to train for.
            n_samples_per_epoch: The total number of samples to generate at the
                                 start of each epoch.
            batch_size: The size of mini-batches for each training step.
            validation_split: The fraction of data to use for validation each epoch.
            early_stopping_patience: The number of epochs to wait for improvement
                                     before stopping early.
            verbose: Whether to print training progress.

        Returns:
            A dictionary containing training metrics and history.
        """
        total_sim_time = 0.0
        total_train_time = 0.0
        if self.training_config.store_thetas:
            self.accumulated_phi_samples = []
        self.total_simulation_count = 0

        # Initialize the network if it hasn't been done yet
        if self.state is None:
            # Determine input shape from a dummy data generation
            phi_dim = 1 # Assuming phi is a scalar for now
            summary_stat_dim = simulator.observed_summary_stats.shape[0]
            input_shape = (batch_size, phi_dim + summary_stat_dim)
            self.initialize_training(input_shape)

        data_generator = lambda key, n: simulator.generate_training_samples(key, n)
        best_val_loss = float('inf')
        early_stop_counter = 0
        lr_plateau_counter = 0
        current_lr = self.training_config.learning_rate 
        schedule_name = self.training_config.lr_scheduler.schedule_name
        lr_schedule_patience = self.training_config.lr_scheduler.schedule_args.get('patience', 10)
        lr_schedule_factor = self.training_config.lr_scheduler.schedule_args.get('factor', 0.5)

        if self.training_config.store_thetas: 
            self.accumulated_phi_samples = []
        self.total_simulation_count = 0

        for epoch in range(num_epochs):
            # 1. Generate a fresh dataset for the entire epoch
            self.key, epoch_key = random.split(self.key)
            time_start_sim = time.time()
            epoch_data = data_generator(epoch_key, n_samples_per_epoch)
            total_sim_time += time.time() - time_start_sim

            
            if verbose and epoch == 0: # On l'affiche une seule fois
                print("\n--- DEBUG INFO ---")
                has_phis = hasattr(epoch_data, 'phi_samples') and epoch_data.phi_samples is not None
                print(f"Does epoch_data contain phi_samples? {has_phis}")
                if has_phis:
                    print(f"Shape of phi_samples: {epoch_data.phi_samples.shape}")
                print("--- END DEBUG INFO ---\n")
                
                
            self.total_simulation_count += getattr(epoch_data, 'total_sim_count', 0)
            
            if self.training_config.store_thetas:
                current_len = len(self.accumulated_phi_samples)
                needed = self.training_config.num_thetas_to_store - current_len
                if needed > 0 and hasattr(epoch_data, 'phi_samples') and epoch_data.phi_samples is not None:
                    self.accumulated_phi_samples.extend(epoch_data.phi_samples[:needed])
        

            # 2. Split the epoch data into a training set and a validation set
            val_size = int(n_samples_per_epoch * validation_split)
            train_size = n_samples_per_epoch - val_size
            
            # Shuffle the epoch data before splitting for better generalization
            self.key, perm_key = random.split(self.key)
            perm = jax.random.permutation(perm_key, n_samples_per_epoch)
            shuffled_features = epoch_data.features[perm]
            shuffled_labels = epoch_data.labels[perm]

            epoch_train_features, epoch_val_features = jnp.split(shuffled_features, [train_size])
            epoch_train_labels, epoch_val_labels = jnp.split(shuffled_labels, [train_size])
            
            num_batches = train_size // batch_size
            
            # 3. Inner loop to iterate over the batches for this epoch
            for i in range(num_batches):
                # Slice the current batch
                start = i * batch_size
                end = start + batch_size
                batch_features = epoch_train_features[start:end]
                batch_labels = epoch_train_labels[start:end]
                
                time_start_train = time.time()
                # Perform one training step
                self.state, train_metrics = train_step(self.state, batch_features, batch_labels)
                total_train_time+= time.time()-time_start_train
            # 4. Evaluate on the validation set at the end of the epoch
            val_metrics = evaluate_step(self.state, epoch_val_features, epoch_val_labels)

            # Update history and check for early stopping
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            if verbose:
                if self.lr_schedule:
                    # For stateless schedules, calculate LR from the current step
                    effective_lr = self.lr_schedule(self.state.step)
                else:
                    # For stateful schedules like reduce_on_plateau
                    effective_lr = current_lr
                print(f"Epoch {epoch + 1}/{num_epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                      f"Train Acc {train_metrics['accuracy']:.2%}, Val Acc: {val_metrics['accuracy']:.2%}, "
                      f"Learning rate = {effective_lr:.6f}")

        self.is_trained = True
        final_results = {
            'history': self.training_history,
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1],
            'final_train_accuracy': self.training_history['train_accuracy'][-1],
            'final_val_accuracy': self.training_history['val_accuracy'][-1],
            'epochs_trained': len(self.training_history['train_loss']),
            'total_simulation_count': self.total_simulation_count,
        }
    
    def predict(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Predict class probabilities for given features.
        
        Args:
            features: Input features of shape (batch_size, feature_dim)
            
        Returns:
            Predicted probabilities of shape (batch_size, 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Utiliser la méthode __call__ de TrainingState qui gère les batch_stats
        if self.state.batch_stats is not None:
            variables = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
        else:
            variables = {'params': self.state.params}
        
        logits = self.state.apply_fn(variables, features, training=False)
        probabilities = nn.sigmoid(logits)
        return probabilities
    
    def log_ratio(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log likelihood ratios for given features.
        
        The log ratio is computed as: log(p(x,θ)/p(x)p(θ)) = log(σ(f(x,θ))/(1-σ(f(x,θ))))
        where σ is the sigmoid function and f is the network output.
        
        Args:
            features: Input features of shape (batch_size, feature_dim)
            
        Returns:
            Log likelihood ratios of shape (batch_size,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before computing log ratios")
        
        # Utiliser la méthode __call__ de TrainingState qui gère les batch_stats
        if self.state.batch_stats is not None:
            variables = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
        else:
            variables = {'params': self.state.params}
        
        logits = self.state.apply_fn(variables, features, training=False)
        log_ratios = logits.flatten()
        
        return log_ratios
    
    def posterior_weights(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Compute posterior weights for given features.
        
        The posterior weights are computed as: w = exp(log_ratio) * prior
        Since we assume uniform prior, this simplifies to: w = exp(log_ratio)
        
        Args:
            features: Input features of shape (batch_size, feature_dim)
            
        Returns:
            Posterior weights of shape (batch_size,)
        """
        log_ratios = self.log_ratio(features)
        weights = jnp.exp(log_ratios)
        return weights
    
    def save_model(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save the model
            metadata: Additional metadata to save
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare save data
        config_to_save = {
            'network_config': self.network.get_config(),
            'network_class': self.network.__class__.__name__,
            'training_config': self.training_config.to_dict() # Save the full training config
        }
        
        save_data = {
            'params': self.state.params,
            'batch_stats': self.state.batch_stats,
            'config': config_to_save, # <-- Use the dictionary we just made
            'training_history': self.training_history,
            'metadata': metadata or {}
        }
        
        
        # Save using JAX serialization
        with open(filepath, 'wb') as f:
            np.savez_compressed(f, **save_data)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        Loads a trained model state from a compressed .npz file.

        Args:
            filepath: Path to the .npz model file.

        Returns:
            A dictionary containing the metadata saved with the model.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            # The npz file is opened and lazily loaded.
            data = np.load(f, allow_pickle=True)

            # --- All logic that accesses `data` must now be INSIDE this 'with' block ---

            # Reconstruct the network if needed.
            loaded_config = data['config'].item()
            if loaded_config['network_class'] != self.network.__class__.__name__:
                raise ValueError(
                    f"Network class mismatch: expected {self.network.__class__.__name__}, "
                    f"but file was saved with {loaded_config['network_class']}"
                )

            # Define the apply function, which is part of the state.
            def apply_fn(variables, x, training=True, **kwargs):
                if 'batch_stats' in variables and variables['batch_stats']:
                    if training and 'mutable' in kwargs:
                        return self.network.apply(variables, x, training=training, **kwargs)
                    else:
                        return self.network.apply(variables, x, training=training)
                else:
                    return self.network.apply(variables, x, training=training)

            # Restore the training state from the saved components.
            self.state = TrainingState.create(
                apply_fn=apply_fn,
                params=data['params'].item(),
                tx=self.optimizer,
                key=self.key,  # Re-initialize key from the estimator's seed
                batch_stats=data.get('batch_stats', {}).item() if 'batch_stats' in data else {}
            )

            # Restore the estimator's configuration and history.
            self.config = loaded_config
            self.training_history = data['training_history'].item()
            self.is_trained = True
            
            metadata = data['metadata'].item() if 'metadata' in data else {}
        
        # The file is automatically and safely closed here, after all data is loaded.
        
        print(f"Model loaded successfully from {filepath}")
        return metadata
    
    def save_config(self, filepath: str) -> None:
        """
        Save estimator configuration to YAML file.
        
        Args:
            filepath: Path to save configuration
        """
        save_model_config(self.config, filepath)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_trained:
            return {'status': 'untrained'}
        
        return {
            'status': 'trained',
            'network_class': self.config['network_class'],
            'network_config': self.config['network_config'],
            'num_parameters': self.network.count_parameters(self.state.params),
            'learning_rate': self.config['learning_rate'],
            'epochs_trained': len(self.training_history['train_loss']),
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1],
            'final_train_accuracy': self.training_history['train_accuracy'][-1],
            'final_val_accuracy': self.training_history['val_accuracy'][-1]
        }


# Backward compatibility
ABCClassifier = NeuralRatioEstimator