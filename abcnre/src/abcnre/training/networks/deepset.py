import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class DeepSet(nn.Module):
    """
    DeepSet for learning classification or regression tasks.

    Used when sample_is_iid = True. Guarantees permutation invariance
    of samples, crucial property for i.i.d. data.

    Architecture: φ(x_i) → pooling → ρ(pooled)

    Args:
        output_dim: Output dimension
        phi_hidden_dims: Dimensions of φ function layers (per-element)
        rho_hidden_dims: Dimensions of ρ function layers (post-pooling)
        pooling_type: Pooling type ('mean', 'sum', 'max', 'attention')
        activation: Activation function
        use_layer_norm: Use LayerNorm
        dropout_rate: Dropout rate
    """

    output_dim: int
    phi_hidden_dims: Optional[List[int]] = None
    rho_hidden_dims: Optional[List[int]] = None
    pooling_type: str = "mean"
    activation: Any = nn.relu
    use_layer_norm: bool = True
    dropout_rate: float = 0.0

    def setup(self):
        if self.phi_hidden_dims is None:
            self.phi_hidden_dims = [32, 16]
        if self.rho_hidden_dims is None:
            self.rho_hidden_dims = [32, 16]

        # Convert activation string to function if needed
        if isinstance(self.activation, str):
            activation_map = {
                "relu": nn.relu,
                "tanh": nn.tanh,
                "sigmoid": nn.sigmoid,
                "gelu": nn.gelu,
                "swish": nn.swish,
                "elu": nn.elu,
                "leaky_relu": nn.leaky_relu,
            }
            if self.activation in activation_map:
                self.activation_fn = activation_map[self.activation]
            else:
                raise ValueError(f"Unknown activation function: {self.activation}")
        else:
            self.activation_fn = self.activation

        # φ network (per-element processing) - use local variables
        phi_layers = []
        for i, dim in enumerate(self.phi_hidden_dims):
            phi_layers.append(nn.Dense(dim, name=f"phi_dense_{i}"))
            if self.use_layer_norm:
                phi_layers.append(nn.LayerNorm(name=f"phi_norm_{i}"))
            if self.dropout_rate > 0.0:
                phi_layers.append(
                    nn.Dropout(self.dropout_rate, name=f"phi_dropout_{i}")
                )

        self.phi_layers = nn.Sequential(phi_layers)

        # φ output dimension
        self.phi_output_dim = self.phi_hidden_dims[-1]

        # Attention if used for pooling
        if self.pooling_type == "attention":
            self.attention_layer = nn.Dense(1, name="attention")

        # ρ network (post-pooling processing) - use local variables
        rho_layers = []
        for i, dim in enumerate(self.rho_hidden_dims):
            rho_layers.append(nn.Dense(dim, name=f"rho_dense_{i}"))
            if self.use_layer_norm:
                rho_layers.append(nn.LayerNorm(name=f"rho_norm_{i}"))
            if self.dropout_rate > 0.0:
                rho_layers.append(
                    nn.Dropout(self.dropout_rate, name=f"rho_dropout_{i}")
                )
        self.rho_layers = nn.Sequential(rho_layers)

        # Final output layer
        self.output_layer = nn.Dense(self.output_dim, name="output")

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass of the DeepSet.

        Args:
            x: Input data, shape (batch_size, n_samples, feature_dim)
            training: Training mode

        Returns:
            output: Summary statistics, shape (batch_size, output_dim)
        """
        
        
        batch_size, n_samples, feature_dim = x.shape

        # φ phase: per-element processing
        # Reshape to process all elements in parallel
        x_flat = x.reshape(-1, feature_dim)  # (batch_size * n_samples, feature_dim)

        # Manual forward pass through phi layers to handle deterministic correctly
        phi_output = x_flat
        for layer in self.phi_layers.layers:
            if isinstance(layer, nn.Dense):
                phi_output = layer(phi_output)
            elif isinstance(layer, nn.LayerNorm):
                phi_output = layer(phi_output)
                phi_output = self.activation_fn(phi_output)
            elif isinstance(layer, nn.Dropout):
                phi_output = layer(phi_output, deterministic=not training)

        # Reshape back
        phi_output = phi_output.reshape(batch_size, n_samples, -1)

        # Pooling phase
        pooled = self._apply_pooling(phi_output, self.pooling_type)

        # ρ phase: post-pooling processing
        rho_output = pooled
        for layer in self.rho_layers.layers:
            if isinstance(layer, nn.Dense):
                rho_output = layer(rho_output)
            elif isinstance(layer, nn.LayerNorm):
                rho_output = layer(rho_output)
                rho_output = self.activation_fn(rho_output)
            elif isinstance(layer, nn.Dropout):
                rho_output = layer(rho_output, deterministic=not training)

        # Final output layer
        output = self.output_layer(rho_output)

        return output

    def _apply_pooling(self, x: jnp.ndarray, pooling_type: str) -> jnp.ndarray:
        """
        Apply pooling operation on the phi outputs.

        Args:
            x: Input tensor of shape (batch, n_samples, phi_dim)
            pooling_type: Type of pooling ('mean', 'max', 'sum', 'attention')

        Returns:
            Pooled tensor of shape (batch, phi_dim)
        """
        if pooling_type == "mean":
            return jnp.mean(x, axis=1)
        elif pooling_type == "max":
            return jnp.max(x, axis=1)
        elif pooling_type == "sum":
            return jnp.sum(x, axis=1)
        elif pooling_type == "attention":
            # Attention-based pooling
            attention_weights = self.attention_layer(x)  # (batch, n_samples, 1)
            attention_weights = nn.softmax(attention_weights, axis=1)
            return jnp.sum(x * attention_weights, axis=1)  # (batch, phi_dim)
        else:
            raise ValueError(
                f"Unsupported pooling type: {pooling_type}. "
                f"Supported types: 'mean', 'max', 'sum', 'attention'"
            )
