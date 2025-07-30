"""
Inference module registry for models, optimizers, and schedulers.

This registry manages components specific to inference:
- Neural network models (NPE, NLE, NRE)
- Optimizers (Adam, SGD, etc.)
- Learning rate schedulers
- Activation functions
"""

from typing import Any, Dict, Type, Callable, Optional
import jax.numpy as jnp
import jax.nn as jnn
import optax
from flax import linen as nn

from ..shared.base_registry import BaseRegistry


class InferenceRegistry(BaseRegistry):
    """Registry for inference-specific components."""

    def __init__(self):
        super().__init__()
        self._register_default_components()

    def _register_default_components(self):
        """Register default inference components."""
        # Neural network models
        self.register("MLP", self._create_mlp, "model")
        self.register("DeepSet", self._create_deepset, "model")

        # Optimizers
        self.register("adam", optax.adam, "optimizer")
        self.register("sgd", optax.sgd, "optimizer")
        self.register("adamw", optax.adamw, "optimizer")

        # Learning rate schedulers
        self.register("constant", optax.constant_schedule, "scheduler")
        self.register("exponential_decay", optax.exponential_decay, "scheduler")
        self.register("cosine_decay", optax.cosine_decay_schedule, "scheduler")

        # Activation functions
        self.register("relu", jnn.relu, "activation")
        self.register("tanh", jnn.tanh, "activation")
        self.register("elu", jnn.elu, "activation")
        self.register("swish", jnn.swish, "activation")

    def _create_mlp(self, features: list, activation: str = "relu") -> nn.Module:
        """Create MLP model."""
        activation_fn = self.get(activation, "activation")

        class MLP(nn.Module):
            @nn.compact
            def __call__(self, x):
                for feat in features[:-1]:
                    x = nn.Dense(feat)(x)
                    x = activation_fn(x)
                x = nn.Dense(features[-1])(x)
                return x

        return MLP

    def _create_deepset(self, features: list, activation: str = "relu") -> nn.Module:
        """Create DeepSet model for set-valued inputs."""
        activation_fn = self.get(activation, "activation")

        class DeepSet(nn.Module):
            @nn.compact
            def __call__(self, x):
                # Encoder
                for feat in features[:-1]:
                    x = nn.Dense(feat)(x)
                    x = activation_fn(x)

                # Pooling operation (mean)
                x = jnp.mean(x, axis=-2, keepdims=False)

                # Decoder
                x = nn.Dense(features[-1])(x)
                return x

        return DeepSet

    def create_optimizer(
        self, name: str, learning_rate: float, **kwargs
    ) -> optax.GradientTransformation:
        """Create optimizer with given parameters."""
        optimizer_fn = self.get(name, "optimizer")
        return optimizer_fn(learning_rate, **kwargs)

    def create_scheduler(self, name: str, **kwargs) -> Callable:
        """Create learning rate scheduler."""
        scheduler_fn = self.get(name, "scheduler")
        return scheduler_fn(**kwargs)

    def create_model(self, name: str, **kwargs) -> nn.Module:
        """Create neural network model."""
        model_fn = self.get(name, "model")
        return model_fn(**kwargs)


# Global inference registry instance
inference_registry = InferenceRegistry()
