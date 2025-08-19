import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

debug = False


class ConditionedDeepSet(nn.Module):
    """
    Conditioned Deep Set Network with flexible conditioning modes.

    This module implements a Deep Set architecture that can condition on
    additional parameters (theta) using different modes:
    - "concat": Concatenates theta with observations
    - "film": Uses FiLM layers to modulate the network based on theta
    """

    output_dim: int 
    phi_hidden_dims: Optional[List[int]] = None
    rho_hidden_dims: Optional[List[int]] = None
    pooling_type: str = "mean"
    conditioning_mode: str = "concat"
    activation: Any = nn.relu
    use_layer_norm: bool = True
    dropout_rate: float = 0.0
    film_hidden_dim: int = 64

    def setup(self):
        if self.phi_hidden_dims is None:
            self.phi_hidden_dims = [64, 32]
        if self.rho_hidden_dims is None:
            self.rho_hidden_dims = [32, 16]

        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed: setup phi_hidden_dims={self.phi_hidden_dims}"
            )
        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed: setup rho_hidden_dims={self.rho_hidden_dims}"
            )
        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed: conditioning_mode={self.conditioning_mode}"
            )

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
            for i, dim in enumerate(self.phi_hidden_dims):
                setattr(
                    self,
                    f"film_intermediate_{i}",
                    nn.Dense(self.film_hidden_dim, name=f"film_{i}_intermediate"),
                )
                setattr(
                    self, f"film_output_{i}", nn.Dense(2 * dim, name=f"film_{i}_output")
                )
                if debug:
                    print(
                        f"DEBUG ConditionedDeepSetFixed: FiLM layer {i} -> intermediate({self.film_hidden_dim}) -> output({2 * dim})"
                    )

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
        """Apply conditioned φ network - VERSION CORRIGÉE."""
        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: theta.shape={theta.shape}"
            )
        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: observations.shape={observations.shape}"
            )

        batch_size, n_samples, feature_dim = observations.shape
        theta_dim = theta.shape[1]

        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: batch_size={batch_size}, n_samples={n_samples}, feature_dim={feature_dim}"
            )
        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: theta_dim={theta_dim}"
            )

        if self.conditioning_mode == "concat":
            theta_expanded = jnp.expand_dims(theta, axis=1)
            theta_expanded = jnp.tile(theta_expanded, (1, n_samples, 1))
            conditioned_input = jnp.concatenate([theta_expanded, observations], axis=-1)
            phi_output = conditioned_input.reshape(-1, theta_dim + feature_dim)
            if debug:
                print(
                    f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: concat mode, phi_input.shape={phi_output.shape}"
                )

        elif self.conditioning_mode == "film":
            phi_output = observations.reshape(-1, feature_dim)
            if debug:
                print(
                    f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: film mode, phi_input.shape={phi_output.shape}"
                )
        else:
            raise ValueError(f"Unsupported conditioning mode: {self.conditioning_mode}")

        film_layer_idx = 0
        for i, layer in enumerate(self.phi_layers.layers):
            if isinstance(layer, nn.Dense):
                phi_output = layer(phi_output)
                if debug:
                    print(
                        f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: after_dense_{i}.shape={phi_output.shape}"
                    )

                # Apply after each Dense layer
                if self.conditioning_mode == "film" and film_layer_idx < len(
                    self.phi_hidden_dims
                ):
                    # Utiliser les couches FiLM individuelles
                    film_intermediate = getattr(
                        self, f"film_intermediate_{film_layer_idx}"
                    )
                    film_output = getattr(self, f"film_output_{film_layer_idx}")

                    # Apply FiLM layers in sequence
                    film_params = film_intermediate(theta)
                    film_params = film_output(film_params)
                    if debug:
                        print(
                            f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: film_params.shape={film_params.shape}"
                        )

                    phi_dim = phi_output.shape[-1]
                    expected_film_size = 2 * phi_dim

                    if film_params.shape[-1] != expected_film_size:
                        raise ValueError(
                            f"FiLM dimension mismatch: expected {expected_film_size}, got {film_params.shape[-1]}"
                        )

                    scale, shift = jnp.split(
                        film_params, 2, axis=-1
                    )  # (batch_size, phi_dim) each
                    if debug:
                        print(
                            f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: scale.shape={scale.shape}, shift.shape={shift.shape}"
                        )

                    # Expand to match phi_output dimensions
                    scale_expanded = jnp.repeat(
                        scale, n_samples, axis=0
                    )  # (batch_size * n_samples, phi_dim)
                    shift_expanded = jnp.repeat(
                        shift, n_samples, axis=0
                    )  # (batch_size * n_samples, phi_dim)
                    if debug:
                        print(
                            f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: scale_expanded.shape={scale_expanded.shape}, shift_expanded.shape={shift_expanded.shape}"
                        )

                    phi_output = scale_expanded * phi_output + shift_expanded
                    if debug:
                        print(
                            f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: after_film_{film_layer_idx}.shape={phi_output.shape}"
                        )
                    film_layer_idx += 1

            elif isinstance(layer, nn.LayerNorm):
                phi_output = layer(phi_output)
                if debug:
                    print(
                        f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: after_layernorm_{i}.shape={phi_output.shape}"
                    )
                phi_output = self.activation_fn(phi_output)
                if debug:
                    print(
                        f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: after_activation_{i}.shape={phi_output.shape}"
                    )

            elif isinstance(layer, nn.Dropout):
                phi_output = layer(phi_output, deterministic=not training)
                if debug:
                    print(
                        f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: after_dropout_{i}.shape={phi_output.shape}"
                    )

        # Reshape back
        phi_output = phi_output.reshape(batch_size, n_samples, -1)
        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed._apply_conditioned_phi: final_output.shape={phi_output.shape}"
            )
        return phi_output

    def _apply_conditioned_rho(
        self, theta: jnp.ndarray, pooled: jnp.ndarray, training: bool
    ) -> jnp.ndarray:
        """Apply conditioned ρ network."""
        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed._apply_conditioned_rho: theta.shape={theta.shape}"
            )
        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed._apply_conditioned_rho: pooled.shape={pooled.shape}"
            )

        rho_input = jnp.concatenate([theta, pooled], axis=-1)
        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed._apply_conditioned_rho: rho_input.shape={rho_input.shape}"
            )

        rho_output = rho_input
        for i, layer in enumerate(self.rho_layers.layers):
            if isinstance(layer, nn.Dense):
                rho_output = layer(rho_output)
                if debug:
                    print(
                        f"DEBUG ConditionedDeepSetFixed._apply_conditioned_rho: after_dense_{i}.shape={rho_output.shape}"
                    )
            elif isinstance(layer, nn.LayerNorm):
                rho_output = layer(rho_output)
                if debug:
                    print(
                        f"DEBUG ConditionedDeepSetFixed._apply_conditioned_rho: after_layernorm_{i}.shape={rho_output.shape}"
                    )
                rho_output = self.activation_fn(rho_output)
                if debug:
                    print(
                        f"DEBUG ConditionedDeepSetFixed._apply_conditioned_rho: after_activation_{i}.shape={rho_output.shape}"
                    )
            elif isinstance(layer, nn.Dropout):
                rho_output = layer(rho_output, deterministic=not training)
                if debug:
                    print(
                        f"DEBUG ConditionedDeepSetFixed._apply_conditioned_rho: after_dropout_{i}.shape={rho_output.shape}"
                    )

        return rho_output

    def _apply_pooling(self, x: jnp.ndarray, pooling_type: str) -> jnp.ndarray:
        """Apply pooling operation on the phi outputs."""
        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed._apply_pooling: input.shape={x.shape}, pooling_type={pooling_type}"
            )

        if pooling_type == "mean":
            result = jnp.mean(x, axis=1)
        elif pooling_type == "max":
            result = jnp.max(x, axis=1)
        elif pooling_type == "sum":
            result = jnp.sum(x, axis=1)
        elif pooling_type == "attention":
            attention_weights = self.attention_layer(x)
            if debug:
                print(
                    f"DEBUG ConditionedDeepSetFixed._apply_pooling: attention_weights.shape={attention_weights.shape}"
                )
            attention_weights = nn.softmax(attention_weights, axis=1)
            result = jnp.sum(x * attention_weights, axis=1)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed._apply_pooling: output.shape={result.shape}"
            )
        return result

    def __call__(self, inputs, training: bool = True) -> jnp.ndarray:
        """Forward pass"""
        if debug:
            print(f"DEBUG ConditionedDeepSetFixed.__call__: inputs type={type(inputs)}")

        if isinstance(inputs, dict):
            theta = inputs["theta"]
            observations = inputs["x"]

            if debug:
                print(
                    f"DEBUG ConditionedDeepSetFixed.__call__: flattened parsing - theta.shape={theta.shape}, observations.shape={observations.shape}"
                )
        else:
            raise ValueError(f"Unsupported input format: {type(inputs)}")

        if debug:
            print(f"DEBUG ConditionedDeepSetFixed.__call__: theta.shape={theta.shape}")
        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed.__call__: observations.shape={observations.shape}"
            )

        # Apply conditioned phi network
        phi_outputs = self._apply_conditioned_phi(theta, observations, training)

        # Apply pooling
        pooled_features = self._apply_pooling(phi_outputs, self.pooling_type)

        # Apply conditioned rho network
        rho_output = self._apply_conditioned_rho(theta, pooled_features, training)

        # Final output
        output = self.output_layer(rho_output)
        if debug:
            print(
                f"DEBUG ConditionedDeepSetFixed.__call__: final_output.shape={output.shape}"
            )

        return output
