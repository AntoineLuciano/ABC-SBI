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
import numpy as np
import matplotlib.pyplot as plt
from ..training import (
    NNConfig,
    get_nn_config,
    train_classifier,
    TrainingResult,
    create_network_from_nn_config,
)
from ..simulation.base import ABCTrainingResult
from ..simulation.simulator import ABCSimulator
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_log_ratio_function(
    network: nn.Module,
    params: Dict[str, Any],
    network_type: str,
    summary_as_input: bool = False,
) -> Callable[[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]:
    """
    Create log-ratio function adapted to different network architectures.
    
    Args:
        network: Trained network instance
        params: Trained parameters
        network_type: Type of network ("conditioned_deepset", "deepset", "MLP")
        summary_as_input: Whether summary statistics are used as input
        
    Returns:
        Log-ratio function that takes (phi, x, s_x) and returns log-ratio estimates
    """
    
    def log_ratio_fn(
        phi: jnp.ndarray,  # Parameters theta: (batch, k)
        x: jnp.ndarray,    # Observations: (batch, n, d) or (batch, d)
        s_x: Optional[jnp.ndarray] = None  # Summary stats: (batch, s)
    ) -> jnp.ndarray:
        """
        Compute log-ratio log[f(x|θ) / p(x)] for given parameters and observations.
        
        Args:
            phi: Parameter values θ, shape (batch, k)
            x: Observations, shape (batch, n, d) or (batch, d)  
            s_x: Optional summary statistics, shape (batch, s)
            
        Returns:
            Log-ratio estimates, shape (batch,)
        """
        
        # Prepare inputs based on network type
        if network_type == "conditioned_deepset":
            # ConditionedDeepSet expects dict input: {'theta': ..., 'x': ...}
            theta = phi
            if summary_as_input and s_x is not None:
                # Concatenate summary stats to theta
                theta = jnp.concatenate([phi, s_x], axis=-1)
            
            network_input = {
                'theta': theta,  # (batch, k) or (batch, k + s)
                'x': x          # (batch, n, d) - preserves structure for deep set
            }
            
        else:
            # DeepSet or MLP expect flattened concatenated input
            # Flatten observations if multi-dimensional
            if x.ndim > 2:
                batch_size = x.shape[0]
                x_flat = x.reshape(batch_size, -1)  # (batch, n*d)
            else:
                x_flat = x  # Already (batch, d)
            
            # Build input components
            input_components = [x_flat, phi]  # [observations, parameters]
            
            if summary_as_input and s_x is not None:
                input_components.append(s_x)  # Add summary stats
            
            # Concatenate all components
            network_input = jnp.concatenate(input_components, axis=-1)
            # Shape: (batch, n*d + k) or (batch, n*d + k + s)
        
        # Forward pass through network
        logits = network.apply(params, network_input, training=False)
        
        # Return log-ratio (squeeze to remove last dimension if output_dim=1)
        log_ratio = logits.squeeze(-1)  # (batch,)
        
        return log_ratio
    
    return log_ratio_fn
    

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
        nn_config: Neural network configuration for training
        random_seed: Random seed for reproducibility

    Example:
        # Create estimator with MLP network
        nn_config = get_nn_config(network_name="MLP", task_type="classifier")
        estimator = NeuralRatioEstimator(nn_config, random_seed=42)

        # Train on ABC simulator
        result = estimator.train(simulator)

        # Estimate posterior
        log_ratios = estimator.log_ratio(features)
        posterior_weights = jnp.exp(log_ratios)
    """

    def __init__(
        self,
        simulator : ABCSimulator,
        nn_config: Optional[Union[NNConfig, Dict[str, Any]]] = None,
        summary_as_input: bool = False,
    ):
        """Initializes the Neural Ratio Estimator."""

        # Handle configuration input
        if nn_config is None:
            # Create default classifier configuration
            self.nn_config = get_nn_config(network_name="MLP", task_type="classifier")
        elif isinstance(nn_config, dict):
            self.nn_config = NNConfig.from_dict(nn_config)
        else:
            self.nn_config = nn_config

        # Ensure task_type is classifier
        if self.nn_config.task_type != "classifier":
            raise ValueError("nn_config.task_type must be 'classifier'")

        self.simulator = simulator
        self.classifier_result: Optional[ClassifierResult] = None
        self.is_trained = False
        self.summary_as_input = summary_as_input

    def train(
        self,  key : jax.random.PRNGKey, n_samples_max: Optional[int] = None, n_sim_max: Optional[int] = None, n_phi_to_store : Optional[int] = 0, **kwargs
    ) -> TrainingResult:
        """
        Trains the neural ratio estimator using the unified training system.

        Args:
            n_samples_max: Maximum number of samples for training (optional stopping rule)
            n_sim_max: Maximum number of simulations to run (optional stopping rule)
            **kwargs: Additional training parameters (will override config if provided)

        Returns:
            ClassifierResult containing trained network and training history
        """

        if n_sim_max is not None:
            if hasattr(self.nn_config.training, "stopping_rules"):
                # Check if stopping_rules is a dict or StoppingRulesConfig object
                if isinstance(self.nn_config.training.stopping_rules, dict):
                    if "simulation_stopping" not in self.nn_config.training.stopping_rules:
                        self.nn_config.training.stopping_rules["simulation_stopping"] = {}

                    self.nn_config.training.stopping_rules["simulation_stopping"][
                        "enabled"
                    ] = True
                    self.nn_config.training.stopping_rules["simulation_stopping"][
                        "max_simulations"
                    ] = n_sim_max

                elif hasattr(self.nn_config.training.stopping_rules, "simulation_stopping"):
                    self.nn_config.training.stopping_rules.simulation_stopping.enabled = (
                        True
                    )
                    self.nn_config.training.stopping_rules.simulation_stopping.max_simulations = (
                        n_sim_max
                    )
            else:
                # Create stopping rules if they don't exist
                self.nn_config.training.stopping_rules = {
                    "simulation_stopping": {"enabled": True, "max_simulations": n_sim_max}
                }
            
        
        
        def create_io_generator(network_type: str, use_summary: bool):
            """Create IO generator based on network type and summary usage."""

            def io_generator(key: random.PRNGKey, batch_size: int):
                """Unified IO generator for all network types."""
                training_result = self.simulator.generate_training_samples(key, batch_size)
                
                if network_type == "conditioned_deepset":
                    # ConditionedDeepSet: dict input format
                    theta = training_result.phi
                    if use_summary:
                        theta = jnp.concatenate([theta, training_result.summary_stats], axis=1)
                    
                    training_input = {
                        'x': training_result.data,
                        'theta': theta
                    }
                else:
                    # DeepSet/MLP: flattened concatenated input
                    inputs = jnp.concatenate([training_result.data.reshape(batch_size, -1), training_result.phi], axis=1)
                    if use_summary:
                        inputs = jnp.concatenate([inputs, training_result.summary_stats], axis=1)
                    training_input = inputs

                return {
                    "input": training_input,
                    "output": training_result.labels,
                    "n_simulations": training_result.total_sim_count,
                }
            return io_generator
    
            
        network_type = self.nn_config.network.network_type
        io_generator = create_io_generator(network_type, self.summary_as_input)

        logger.info(f"Using {network_type} {'with' if self.summary_as_input else 'without'} summary statistics")
        
        if n_phi_to_store is not None and n_phi_to_store > 0:
            logger.info(f"Storing {n_phi_to_store} phi during training")
            self.nn_config.training.n_phi_to_store = n_phi_to_store 
        # Train using the unified system
        key, train_key = random.split(key)
        self.classifier_result = train_classifier(
            key=train_key, config=self.nn_config, io_generator=io_generator
        )

        self.is_trained = True

        if self.nn_config.training.verbose:
            print("✅ Neural Ratio Estimator training completed successfully!")
            print(
                f"   - Final train loss: {self.classifier_result.training_history.get('final_loss', 'N/A')}"
            )
            print(
                f"   - Total simulations: {self.classifier_result.training_history.get('total_simulations', 'N/A')}"
            )
        self.trained_params = self.classifier_result.params
        self.trained_network = self.classifier_result.network
        self._training_history = self.classifier_result.training_history
        self._trained_config = self.nn_config
        self.stored_phis = self.classifier_result.stored_phi if hasattr(self.classifier_result, 'stored_phi') else None

        self.log_ratio_fn = create_log_ratio_function(
            network=self.classifier_result.network,
            params=self.classifier_result.params,
            network_type=self.nn_config.network.network_type,
            summary_as_input=self.summary_as_input
        )
        
        return self.classifier_result

    def predict(
        self,
        phi: jnp.ndarray,
        x: jnp.ndarray,
        s_x: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Predict log-ratio estimates for given parameters and observations.

        Args:
            phi: Parameters θ, shape (batch, k)
            x: Observations, shape (batch, n, d) or (batch, d)
            s_x: Optional summary statistics, shape (batch, s)

        Returns:
            Log-ratio estimates, shape (batch,)
        """
        if not self.is_trained:
            raise RuntimeError("Estimator is not trained. Call train() first.")

        return nn.sigmoid(
            self.log_ratio_fn(phi, x, s_x)
        )
        
