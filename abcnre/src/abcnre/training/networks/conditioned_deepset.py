import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ConditionedDeepSet(nn.Module):
    """
    Conditioned DeepSet for learning classification or regression tasks.

    Used when samples need to be conditioned on parameters θ.
    Guarantees permutation invariance while incorporating parameter conditioning.

    Architecture: φ(θ, x_i) → pooling → ρ(θ, pooled)

    Args:
        output_dim: Output dimension
        phi_hidden_dims: Dimensions of φ function layers (per-element)
        rho_hidden_dims: Dimensions of ρ function layers (post-pooling)
        pooling_type: Pooling type ('mean', 'sum', 'max', 'attention')
        conditioning_mode: How to combine θ with x ('concat', 'film')
        activation: Activation function
        use_layer_norm: Use LayerNorm
        dropout_rate: Dropout rate
    """

    output_dim: int = 1
    phi_hidden_dims: Optional[List[int]] = None
    rho_hidden_dims: Optional[List[int]] = None
    pooling_type: str = "mean"
    conditioning_mode: str = "concat"
    activation: Any = nn.relu
    use_layer_norm: bool = True
    dropout_rate: float = 0.0

    def setup(self):
        if self.phi_hidden_dims is None:
            self.phi_hidden_dims = [64, 32]
        if self.rho_hidden_dims is None:
            self.rho_hidden_dims = [32, 16]

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

        if self.conditioning_mode == "film":
            self.film_layers = []
            for i in range(len(self.phi_hidden_dims)):
                self.film_layers.append(nn.Dense(2, name=f"film_{i}"))

        if self.pooling_type == "attention":
            self.attention_layer = nn.Dense(1, name="attention")

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

    def _apply_conditioned_phi(
        self, theta: jnp.ndarray, observations: jnp.ndarray, training: bool
    ) -> jnp.ndarray:
        """Apply conditioned φ network - COHÉRENT avec DeepSet."""
        batch_size, n_samples, feature_dim = observations.shape
        theta_dim = theta.shape[1]

        if self.conditioning_mode == "concat":
            theta_expanded = jnp.expand_dims(theta, axis=1)
            theta_expanded = jnp.tile(theta_expanded, (1, n_samples, 1))
            conditioned_input = jnp.concatenate([theta_expanded, observations], axis=-1)
            phi_output = conditioned_input.reshape(-1, theta_dim + feature_dim)
        elif self.conditioning_mode == "film":
            phi_output = observations.reshape(-1, feature_dim)
        else:
            raise ValueError(f"Unsupported conditioning mode: {self.conditioning_mode}")

        film_layer_idx = 0
        for layer in self.phi_layers.layers:
            if isinstance(layer, nn.Dense):
                phi_output = layer(phi_output)
            elif isinstance(layer, nn.LayerNorm):
                phi_output = layer(phi_output)

                if self.conditioning_mode == "film":
                    film_params = self.film_layers[film_layer_idx](theta)
                    scale, shift = jnp.split(film_params, 2, axis=-1)
                    scale = jnp.repeat(scale, n_samples, axis=0)
                    shift = jnp.repeat(shift, n_samples, axis=0)
                    phi_output = scale * phi_output + shift
                    film_layer_idx += 1

                phi_output = self.activation_fn(phi_output)

            elif isinstance(layer, nn.Dropout):
                phi_output = layer(phi_output, deterministic=not training)

        # Reshape back
        phi_output = phi_output.reshape(batch_size, n_samples, -1)
        return phi_output

    def _apply_conditioned_rho(
        self, theta: jnp.ndarray, pooled: jnp.ndarray, training: bool
    ) -> jnp.ndarray:
        """Apply conditioned ρ network - COHÉRENT avec DeepSet."""
        rho_input = jnp.concatenate([theta, pooled], axis=-1)

        rho_output = rho_input
        for layer in self.rho_layers.layers:
            if isinstance(layer, nn.Dense):
                rho_output = layer(rho_output)
            elif isinstance(layer, nn.LayerNorm):
                rho_output = layer(rho_output)
                rho_output = self.activation_fn(rho_output)
            elif isinstance(layer, nn.Dropout):
                rho_output = layer(rho_output, deterministic=not training)

        return rho_output

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

    def __call__(self, inputs, training: bool = True) -> jnp.ndarray:
        """
        Forward pass of the conditioned DeepSet network.

        Args:
            inputs: Can be either:
                    1. Dictionary with structured data: {'raw_data': (batch, n_obs, d), 'phi': (batch, n_phi)}
                    2. Tuple: (theta, observations) where theta: (batch, n_theta), observations: (batch, n_obs, d)
                    3. Flattened array: (batch, total_features) - for backward compatibility/initialization
            training: Whether in training mode

        Returns:
            Network output of shape (batch_size, output_dim) representing log ratios
        """

        # Parse inputs to extract θ and observations
        if isinstance(inputs, dict):
            # Dictionary format - structured data (preferred)
            theta = inputs["phi"]  # Parameters: (batch, n_theta)
            observations = inputs["raw_data"]  # Observations: (batch, n_obs, d)

            # Include summary stats if available
            if "summary_stats" in inputs:
                theta = jnp.concatenate([theta, inputs["summary_stats"]], axis=1)

        elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            # Tuple format: (theta, observations)
            theta, observations = inputs

        elif isinstance(inputs, jnp.ndarray) and inputs.ndim == 2:
            # Flattened format - for initialization and backward compatibility
            batch_size, total_features = inputs.shape

            # For initialization, make simple assumptions about data structure
            # Assume first few features are theta, rest are flattened observations
            if total_features >= 4:  # Minimum viable split
                # Take first 2 features as theta (common case)
                theta_dim = 2
                theta = inputs[:, :theta_dim]

                # Remaining features as observations - reshape to 3D
                obs_features = inputs[:, theta_dim:]
                n_obs_features = obs_features.shape[1]

                # Infer observation structure
                if n_obs_features >= 20:  # Many features, likely 2D observations
                    # Assume square-ish structure for 2D observations
                    obs_dim = 2
                    n_obs = n_obs_features // obs_dim
                elif n_obs_features % 2 == 0 and n_obs_features >= 6:
                    # Even number, likely 2D observations
                    obs_dim = 2
                    n_obs = n_obs_features // obs_dim
                else:
                    # Assume 1D observations
                    obs_dim = 1
                    n_obs = n_obs_features

                observations = obs_features.reshape(batch_size, n_obs, obs_dim)
            else:
                # Very minimal case for initialization - split roughly in half
                mid = total_features // 2
                theta = inputs[:, :mid] if mid > 0 else inputs[:, :1]
                obs_flat = inputs[:, mid:] if mid < total_features else inputs[:, 1:]
                observations = obs_flat.reshape(batch_size, obs_flat.shape[1], 1)

        else:
            raise ValueError(
                f"Input format not supported. Expected dict with 'phi' and 'raw_data' keys, "
                f"tuple (theta, observations), or flattened array. Got: {type(inputs)} "
                f"with shape {getattr(inputs, 'shape', 'N/A')}"
            )

        # Apply conditioned phi network to each observation
        phi_outputs = self._apply_conditioned_phi(theta, observations, training)

        # Apply pooling
        pooled_features = self._apply_pooling(phi_outputs, self.pooling_type)

        # Apply conditioned rho network
        rho_output = self._apply_conditioned_rho(theta, pooled_features, training)

        # Final output layer
        output = self.output_layer(rho_output)

        return output
