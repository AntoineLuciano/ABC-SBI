"""
DeepSet network architecture for NRE.

DeepSets are particularly well-suited for ABC inference as they are
permutation-invariant, making them ideal for handling sets of observations
or summary statistics.

Reference: Zaheer et al. "Deep Sets" (2017)
"""

from typing import Sequence, Dict, Any, Optional
import flax.linen as nn
import jax.numpy as jnp

from .base import NetworkBase


class DeepSetNetwork(NetworkBase):
    """
    DeepSet network for permutation-invariant neural ratio estimation.
    
    The DeepSet architecture consists of:
    1. Phi network: Applied to each element independently
    2. Pooling operation: Aggregates across elements (mean, max, sum)
    3. Rho network: Processes the aggregated representation
    
    This is ideal for ABC when dealing with multiple observations or
    when the order of summary statistics shouldn't matter.
    
    Args:
        phi_hidden_dims: Hidden dimensions for phi network
        rho_hidden_dims: Hidden dimensions for rho network
        output_dim: Output dimension (default: 1 for binary classification)
        pooling: Pooling operation ('mean', 'max', 'sum')
        activation: Activation function
        dropout_rate: Dropout rate (0.0 = no dropout)
        use_batch_norm: Whether to use batch normalization
        
    Example:
        # Standard DeepSet for ABC
        network = DeepSetNetwork(
            phi_hidden_dims=[64, 64],
            rho_hidden_dims=[128, 64],
            pooling='mean'
        )
        
        # For set of observations (batch_size, n_obs, obs_dim)
        x = jnp.ones((32, 100, 10))  # 32 batches, 100 obs, 10-dim each
        output = network(x)  # Shape: (32, 1)
    """
    
    phi_hidden_dims: Sequence[int] = (64, 64)
    rho_hidden_dims: Sequence[int] = (128, 64) 
    output_dim: int = 1
    pooling: str = 'mean'
    activation: str = 'relu'
    dropout_rate: float = 0.0
    use_batch_norm: bool = False
    
    def setup(self):
        """Initialize phi and rho networks."""
        # Phi network layers - créer directement avec setattr, pas de listes !
        for i, dim in enumerate(self.phi_hidden_dims):
            setattr(self, f'phi_dense_{i}', nn.Dense(dim))
            
            if self.use_batch_norm:
                setattr(self, f'phi_bn_{i}', nn.BatchNorm())
            
            if self.dropout_rate > 0:
                setattr(self, f'phi_dropout_{i}', nn.Dropout(rate=self.dropout_rate))
        
        # Rho network layers - créer directement avec setattr, pas de listes !
        for i, dim in enumerate(self.rho_hidden_dims):
            setattr(self, f'rho_dense_{i}', nn.Dense(dim))
            
            if self.use_batch_norm:
                setattr(self, f'rho_bn_{i}', nn.BatchNorm())
            
            if self.dropout_rate > 0:
                setattr(self, f'rho_dropout_{i}', nn.Dropout(rate=self.dropout_rate))
        
        # Output layer
        self.output_layer = nn.Dense(self.output_dim)
    
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass of DeepSet network.
        
        Args:
            x: Input data of shape (batch_size, n_elements, element_dim)
               or (batch_size, feature_dim) for flattened input
            training: Whether in training mode
            
        Returns:
            Output of shape (batch_size, output_dim)
        """
        # Handle both 2D and 3D inputs
        if x.ndim == 2:
            # Assume flattened input, treat each feature as an element
            x = x[:, :, None]  # (batch_size, n_features, 1)
        elif x.ndim != 3:
            raise ValueError(f"Input must be 2D or 3D, got shape {x.shape}")
        
        batch_size, n_elements, element_dim = x.shape
        
        # Apply phi network to each element
        # Reshape to (batch_size * n_elements, element_dim)
        x_flat = x.reshape(-1, element_dim)
        
        # Pass through phi network - récupérer dynamiquement avec getattr
        phi_out = x_flat
        activation_fn = self._get_activation_fn()
        
        for i, dim in enumerate(self.phi_hidden_dims):
            # Dense layer
            dense_layer = getattr(self, f'phi_dense_{i}')
            phi_out = dense_layer(phi_out)
            
            # Apply batch normalization if enabled
            if self.use_batch_norm:
                bn_layer = getattr(self, f'phi_bn_{i}')
                phi_out = bn_layer(phi_out, use_running_average=not training)
            
            # Apply activation
            phi_out = activation_fn(phi_out)
            
            # Apply dropout if enabled
            if self.dropout_rate > 0:
                dropout_layer = getattr(self, f'phi_dropout_{i}')
                phi_out = dropout_layer(phi_out, deterministic=not training)
        
        # Reshape back to (batch_size, n_elements, phi_output_dim)
        phi_output_dim = phi_out.shape[-1]
        phi_out = phi_out.reshape(batch_size, n_elements, phi_output_dim)
        
        # Apply pooling operation
        if self.pooling == 'mean':
            pooled = jnp.mean(phi_out, axis=1)
        elif self.pooling == 'max':
            pooled = jnp.max(phi_out, axis=1)
        elif self.pooling == 'sum':
            pooled = jnp.sum(phi_out, axis=1)
        else:
            raise ValueError(f"Unknown pooling operation: {self.pooling}")
        
        # Apply rho network - récupérer dynamiquement avec getattr
        rho_out = pooled
        
        for i, dim in enumerate(self.rho_hidden_dims):
            # Dense layer
            dense_layer = getattr(self, f'rho_dense_{i}')
            rho_out = dense_layer(rho_out)
            
            # Apply batch normalization if enabled
            if self.use_batch_norm:
                bn_layer = getattr(self, f'rho_bn_{i}')
                rho_out = bn_layer(rho_out, use_running_average=not training)
            
            # Apply activation
            rho_out = activation_fn(rho_out)
            
            # Apply dropout if enabled
            if self.dropout_rate > 0:
                dropout_layer = getattr(self, f'rho_dropout_{i}')
                rho_out = dropout_layer(rho_out, deterministic=not training)
        
        # Final output layer
        rho_out = self.output_layer(rho_out)
        
        return rho_out
    
    def _get_activation_fn(self):
        """Get activation function."""
        if self.activation == 'relu':
            return nn.relu
        elif self.activation == 'tanh':
            return nn.tanh
        elif self.activation == 'gelu':
            return nn.gelu
        elif self.activation == 'swish':
            return nn.swish
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get network configuration."""
        return {
            'phi_hidden_dims': list(self.phi_hidden_dims),
            'rho_hidden_dims': list(self.rho_hidden_dims),
            'output_dim': self.output_dim,
            'pooling': self.pooling,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm
        }
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


class CompactDeepSetNetwork(NetworkBase):
    """
    Compact DeepSet for simple ABC inference.
    
    Simplified version with fewer parameters for quick experimentation.
    """
    
    hidden_dim: int = 64
    output_dim: int = 1
    pooling: str = 'mean'
    
    def setup(self):
        """Initialize compact phi and rho networks."""
        self.phi = nn.Dense(self.hidden_dim)
        self.rho_dense1 = nn.Dense(self.hidden_dim)
        self.rho_dense2 = nn.Dense(self.output_dim)
    
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass."""
        if x.ndim == 2:
            x = x[:, :, None]
        
        batch_size, n_elements, element_dim = x.shape
        
        # Apply phi to each element
        x_flat = x.reshape(-1, element_dim)
        phi_out = nn.relu(self.phi(x_flat))
        phi_out = phi_out.reshape(batch_size, n_elements, self.hidden_dim)
        
        # Pool and apply rho
        if self.pooling == 'mean':
            pooled = jnp.mean(phi_out, axis=1)
        elif self.pooling == 'max':
            pooled = jnp.max(phi_out, axis=1)
        else:
            pooled = jnp.sum(phi_out, axis=1)
        
        # Apply rho network
        rho_out = nn.relu(self.rho_dense1(pooled))
        rho_out = self.rho_dense2(rho_out)
        
        return rho_out
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        return {
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'pooling': self.pooling
        }


# Export classes
__all__ = ["DeepSetNetwork", "CompactDeepSetNetwork"]