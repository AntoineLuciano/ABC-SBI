import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, List, Tuple, Any
import logging
from ..components.loop import Inputs

logger = logging.getLogger(__name__)

debug = False


class MLP(nn.Module):
    """
    MLP for learning classification or regression tasks.

    Used when sample_is_iid = False, i.e. when samples
    present temporal or spatial correlations.

    Args:
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function to use
        use_layer_norm: Use LayerNorm after each layer
        dropout_rate: Dropout rate (0.0 = no dropout)
        sequence_pooling: How to handle 3D inputs ('flatten', 'mean', 'max', 'sum')
        flatten_threshold: Max features before switching to pooling (default: 1000)
    """

    output_dim: int = 1
    hidden_dims: Optional[List[int]] = None
    activation: Any = nn.relu
    use_layer_norm: bool = True
    dropout_rate: float = 0.0
    sequence_pooling: str = "adaptive"  # 'flatten', 'mean', 'max', 'sum', 'adaptive'
    flatten_threshold: int = 1000
    use_layer_norm: bool = True
    dropout_rate: float = 0.0

    def setup(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]

        if debug:
            print(f"DEBUG MLP: setup hidden_dims={self.hidden_dims}")
        if debug:
            print(f"DEBUG MLP: setup output_dim={self.output_dim}")

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

        # Hidden layers - use local variable
        layers = []
        for i, dim in enumerate(self.hidden_dims):
            layers.append(nn.Dense(dim, name=f"dense_{i}"))

            if self.use_layer_norm:
                layers.append(nn.LayerNorm(name=f"norm_{i}"))

            # Activation will be applied in __call__

            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(self.dropout_rate, name=f"dropout_{i}"))

        self.layers = layers

        # Output layer
        self.output_layer = nn.Dense(self.output_dim, name="output")

    def __call__(self, inputs: Inputs, training: bool = True) -> jnp.ndarray:
        """
        Forward pass of the MLP.

        Args:
            x: Input data, shape (batch_size, seq_len, feature_dim)
               or (batch_size, total_features)
            training: Training mode for dropout

        Returns:
            output: Summary statistics, shape (batch_size, output_dim)
        """
        x = inputs.raw
        if debug:
            print(f"DEBUG MLP.__call__: input x.shape={x.shape}")
        if debug:
            print(f"DEBUG MLP.__call__: input x.ndim={x.ndim}")

        # Handle sequence data with configurable pooling strategy
        if x.ndim == 3:
            batch_size, n_samples, feature_dim = x.shape
            total_features = n_samples * feature_dim

            if debug:
                print(
                    f"DEBUG MLP.__call__: 3D input detected: batch_size={batch_size}, n_samples={n_samples}, feature_dim={feature_dim}"
                )
                print(
                    f"DEBUG MLP.__call__: total_features={total_features}, threshold={self.flatten_threshold}"
                )

            # Decide pooling strategy
            if self.sequence_pooling == "adaptive":
                use_pooling = total_features > self.flatten_threshold
                pooling_method = "mean"
            elif self.sequence_pooling == "flatten":
                use_pooling = False
                pooling_method = None
            else:
                use_pooling = True
                pooling_method = self.sequence_pooling

            if use_pooling:
                if debug:
                    print(f"DEBUG MLP.__call__: Using {pooling_method} pooling")

                if pooling_method == "mean":
                    x = jnp.mean(x, axis=1)  # (B, n, d) -> (B, d)
                elif pooling_method == "max":
                    x = jnp.max(x, axis=1)  # (B, n, d) -> (B, d)
                elif pooling_method == "sum":
                    x = jnp.sum(x, axis=1)  # (B, n, d) -> (B, d)
                else:
                    raise ValueError(f"Unknown pooling method: {pooling_method}")

                if debug:
                    print(
                        f"DEBUG MLP.__call__: after {pooling_method} pooling x.shape={x.shape}"
                    )
            else:
                # Standard flattening
                x = x.reshape(batch_size, -1)
                if debug:
                    print(f"DEBUG MLP.__call__: flattened x.shape={x.shape}")
        elif x.ndim > 3:
            # For higher dimensions, always flatten
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
            if debug:
                print(f"DEBUG MLP.__call__: >3D input flattened x.shape={x.shape}")
        elif x.ndim == 2:
            # Already in correct format (B, features)
            if debug:
                print(f"DEBUG MLP.__call__: 2D input, no reshaping needed")

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Dense):
                x = layer(x)
                if debug:
                    print(f"DEBUG MLP.__call__: after_dense_{i}.shape={x.shape}")
                x = self.activation_fn(x)
                if debug:
                    print(f"DEBUG MLP.__call__: after_activation_{i}.shape={x.shape}")
            elif isinstance(layer, nn.LayerNorm):
                x = layer(x)
                if debug:
                    print(f"DEBUG MLP.__call__: after_layernorm_{i}.shape={x.shape}")
            elif isinstance(layer, nn.Dropout):
                x = layer(x, deterministic=not training)
                if debug:
                    print(f"DEBUG MLP.__call__: after_dropout_{i}.shape={x.shape}")

        # Output layer
        output = self.output_layer(x)
        if debug:
            print(f"DEBUG MLP.__call__: final_output.shape={output.shape}")

        return output
