# src/abcnre/preprocessing/statistic_learner.py

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
import flax.serialization
from flax.training import train_state
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional

from ..inference.networks.base import NetworkBase, create_network_from_config
from ..simulation.simulator import ABCSimulator
from ..inference.trainer import TrainingState

@jax.jit
def regression_train_step(state: TrainingState, batch_data: jnp.ndarray, batch_phi: jnp.ndarray) -> (TrainingState, Dict):
    """Performs a single training step for the regression task."""
    def loss_fn(params):
        predicted_phi = state.apply_fn({'params': params}, batch_data)
        loss = optax.squared_error(predicted_phi.flatten(), batch_phi.flatten()).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    metrics = {'mse_loss': loss}
    return new_state, metrics

class StatisticLearner:
    """Trains a neural network to learn optimal summary statistics."""
    def __init__(self, summary_network: NetworkBase, learning_rate: float = 1e-3):
        self.network = summary_network
        self.optimizer = optax.adam(learning_rate)
        self.state: Optional[TrainingState] = None
        self.input_shape: Optional[tuple] = None # To store input shape for reloading

    def train(self, simulator: ABCSimulator, n_samples: int, num_epochs: int, batch_size: int):
        """Trains the summary network."""
        print(f"Generating {n_samples} simulation pairs for statistic learning...")
        key = jax.random.PRNGKey(0)
        
        # 1. Generate (theta, data) pairs
        key, prior_key = jax.random.split(key)
        thetas = simulator.model.get_prior_samples(prior_key, n_samples)
        phis = simulator.model.transform_phi(thetas)

        sim_keys = jax.random.split(key, n_samples)
        n_obs = simulator.observed_data.shape[0]
        all_data = jax.vmap(simulator.model.simulate, in_axes=(0, 0, None))(sim_keys, thetas, n_obs)
        all_features = jnp.mean(all_data, axis=1, keepdims=True)
        
        # --- Store input shape ---
        self.input_shape = all_features.shape[1:]

        # 2. Initialize network state
        self.state = TrainingState.create(
            apply_fn=self.network.apply,
            params=self.network.init(jax.random.PRNGKey(1), all_features[:1])['params'],
            tx=self.optimizer
        )
        print("Summary network initialized.")

        # 3. Training loop
        num_batches = n_samples // batch_size
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                start = i * batch_size
                end = start + batch_size
                batch_features = all_features[start:end]
                batch_phis = phis[start:end]
                
                self.state, metrics = regression_train_step(self.state, batch_features, batch_phis)
                epoch_loss += metrics['mse_loss']
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs} - Avg. MSE Loss: {avg_loss:.6f}")
            
        print("✅ Summary network training complete.")

    def transform(self, data: jnp.ndarray) -> jnp.ndarray:
        """Computes the learned summary statistic for new data."""
        if self.state is None:
            raise RuntimeError("Learner must be trained before transforming data.")
        
        features = jnp.mean(data, axis=0, keepdims=True)
        embedding = self.network.apply(
            {'params': self.state.params}, 
            features,
            method=self.network.embed
        )
        return embedding.flatten()
        
    def save(self, filepath: Path):
        """Saves the trained StatisticLearner to a YAML and weights file."""
        if self.state is None or self.input_shape is None:
            raise RuntimeError("Learner must be trained before it can be saved.")
            
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Define paths
        weights_path = filepath.with_suffix('.npy')
        
        # 2. Save weights
        params_bytes = flax.serialization.to_bytes(self.state.params)
        with open(weights_path, 'wb') as f:
            f.write(params_bytes)
            
        # 3. Create and save YAML config
        config = {
            'network_config': {
                'network_type': self.network.__class__.__name__,
                'network_args': self.network.get_config()
            },
            'input_shape': self.input_shape,
            'weights_path': str(weights_path.resolve())
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"✅ StatisticLearner saved. Master config: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'StatisticLearner':
        """Loads a StatisticLearner from a YAML configuration file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
            
        # 1. Recreate network and learner
        network = create_network_from_config(config['network_config'])
        # Note: learning rate is not saved, as it's only for training.
        learner = cls(summary_network=network)
        
        # 2. Initialize state with the saved shape
        learner.input_shape = config['input_shape']
        dummy_input = jnp.ones((1, *learner.input_shape))
        
        learner.state = TrainingState.create(
            apply_fn=learner.network.apply,
            params=learner.network.init(jax.random.PRNGKey(0), dummy_input)['params'],
            tx=learner.optimizer
        )
        
        # 3. Load and restore weights
        weights_path = Path(config['weights_path'])
        with open(weights_path, 'rb') as f:
            params_bytes = f.read()
        
        loaded_params = flax.serialization.from_bytes(learner.state.params, params_bytes)
        learner.state = learner.state.replace(params=loaded_params)
        
        print(f"✅ StatisticLearner loaded from {filepath}")
        return learner
    
    
    