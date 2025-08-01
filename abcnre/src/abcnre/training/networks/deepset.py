import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

debug = False  # Set to True for detailed debug output

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

        if debug: print(f"DEBUG DeepSet: setup phi_hidden_dims={self.phi_hidden_dims}")
        if debug: print(f"DEBUG DeepSet: setup rho_hidden_dims={self.rho_hidden_dims}")

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
        if debug: print(f"DEBUG DeepSet.__call__: input x.shape={x.shape}")
        
        batch_size, n_samples, feature_dim = x.shape
        if debug: print(f"DEBUG DeepSet.__call__: batch_size={batch_size}, n_samples={n_samples}, feature_dim={feature_dim}")

        # φ phase: per-element processing
        # Reshape to process all elements in parallel
        x_flat = x.reshape(-1, feature_dim)  # (batch_size * n_samples, feature_dim)
        if debug: print(f"DEBUG DeepSet.__call__: x_flat.shape={x_flat.shape}")

        # Manual forward pass through phi layers to handle deterministic correctly
        phi_output = x_flat
        for i, layer in enumerate(self.phi_layers.layers):
            if isinstance(layer, nn.Dense):
                phi_output = layer(phi_output)
                if debug: print(f"DEBUG DeepSet.__call__: phi after_dense_{i}.shape={phi_output.shape}")
            elif isinstance(layer, nn.LayerNorm):
                phi_output = layer(phi_output)
                if debug: print(f"DEBUG DeepSet.__call__: phi after_layernorm_{i}.shape={phi_output.shape}")
                phi_output = self.activation_fn(phi_output)
                if debug: print(f"DEBUG DeepSet.__call__: phi after_activation_{i}.shape={phi_output.shape}")
            elif isinstance(layer, nn.Dropout):
                phi_output = layer(phi_output, deterministic=not training)
                if debug: print(f"DEBUG DeepSet.__call__: phi after_dropout_{i}.shape={phi_output.shape}")

        # Reshape back
        phi_output = phi_output.reshape(batch_size, n_samples, -1)
        if debug: print(f"DEBUG DeepSet.__call__: phi_output_reshaped.shape={phi_output.shape}")

        # Pooling phase
        pooled = self._apply_pooling(phi_output, self.pooling_type)

        # ρ phase: post-pooling processing
        rho_output = pooled
        if debug: print(f"DEBUG DeepSet.__call__: rho_input.shape={rho_output.shape}")
        
        for i, layer in enumerate(self.rho_layers.layers):
            if isinstance(layer, nn.Dense):
                rho_output = layer(rho_output)
                if debug: print(f"DEBUG DeepSet.__call__: rho after_dense_{i}.shape={rho_output.shape}")
            elif isinstance(layer, nn.LayerNorm):
                rho_output = layer(rho_output)
                if debug: print(f"DEBUG DeepSet.__call__: rho after_layernorm_{i}.shape={rho_output.shape}")
                rho_output = self.activation_fn(rho_output)
                if debug: print(f"DEBUG DeepSet.__call__: rho after_activation_{i}.shape={rho_output.shape}")
            elif isinstance(layer, nn.Dropout):
                rho_output = layer(rho_output, deterministic=not training)
                if debug: print(f"DEBUG DeepSet.__call__: rho after_dropout_{i}.shape={rho_output.shape}")

        # Final output layer
        output = self.output_layer(rho_output)
        if debug: print(f"DEBUG DeepSet.__call__: final_output.shape={output.shape}")

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
        if debug: print(f"DEBUG DeepSet._apply_pooling: input.shape={x.shape}, pooling_type={pooling_type}")
        
        if pooling_type == "mean":
            result = jnp.mean(x, axis=1)
        elif pooling_type == "max":
            result = jnp.max(x, axis=1)
        elif pooling_type == "sum":
            result = jnp.sum(x, axis=1)
        elif pooling_type == "attention":
            # Attention-based pooling
            attention_weights = self.attention_layer(x)  # (batch, n_samples, 1)
            if debug: print(f"DEBUG DeepSet._apply_pooling: attention_weights.shape={attention_weights.shape}")
            attention_weights = nn.softmax(attention_weights, axis=1)
            if debug: print(f"DEBUG DeepSet._apply_pooling: attention_weights_softmax.shape={attention_weights.shape}")
            result = jnp.sum(x * attention_weights, axis=1)  # (batch, phi_dim)
        else:
            raise ValueError(
                f"Unsupported pooling type: {pooling_type}. "
                f"Supported types: 'mean', 'max', 'sum', 'attention'"
            )
        
        if debug: print(f"DEBUG DeepSet._apply_pooling: output.shape={result.shape}")
        return result