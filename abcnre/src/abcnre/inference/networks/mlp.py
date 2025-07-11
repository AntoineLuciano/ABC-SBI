"""
Multi-Layer Perceptron (MLP) networks for NRE.

Standard fully-connected networks for neural ratio estimation
when the input has a fixed structure.
"""

from typing import Sequence, Dict, Any
import flax.linen as nn
import jax.numpy as jnp

from .base import NetworkBase


class MLPNetwork(NetworkBase):
    """
    Standard Multi-Layer Perceptron for neural ratio estimation.
    
    A simple but effective architecture for NRE when dealing with
    fixed-size summary statistics or feature vectors.
    
    Args:
        hidden_dims: Sequence of hidden layer dimensions
        output_dim: Output dimension (default: 1 for binary classification)
        activation: Activation function ('relu', 'tanh', 'gelu', 'swish')
        dropout_rate: Dropout rate (0.0 = no dropout)
        use_batch_norm: Whether to use batch normalization
        final_activation: Activation for final layer (None for linear output)
        
    Example:
        # Standard MLP for ABC
        network = MLPNetwork(
            hidden_dims=[128, 64, 32],
            dropout_rate=0.1,
            use_batch_norm=True
        )
        
        # For fixed-size features
        x = jnp.ones((32, 50))  # 32 batches, 50 features
        output = network(x)     # Shape: (32, 1)
    """
    
    hidden_dims: Sequence[int] = (128, 64, 32)
    output_dim: int = 1
    activation: str = 'relu'
    dropout_rate: float = 0.0
    use_batch_norm: bool = False
    final_activation: str = None
    
    def setup(self):
        """Initialize MLP layers."""
        # Créer les couches directement avec setattr - pas de listes !
        for i, dim in enumerate(self.hidden_dims):
            setattr(self, f'dense_{i}', nn.Dense(dim))
            
            if self.use_batch_norm:
                setattr(self, f'bn_{i}', nn.BatchNorm())
            
            if self.dropout_rate > 0:
                setattr(self, f'dropout_{i}', nn.Dropout(rate=self.dropout_rate))
        
        # Output layer
        self.output_layer = nn.Dense(self.output_dim)
    
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass of MLP."""
        if x.ndim != 2:
            raise ValueError(f"MLP expects 2D input, got shape {x.shape}")
        
        out = x
        
        # Process hidden layers - récupérer dynamiquement avec getattr
        for i, dim in enumerate(self.hidden_dims):
            # Dense layer
            dense_layer = getattr(self, f'dense_{i}')
            out = dense_layer(out)
            
            # Batch norm
            if self.use_batch_norm:
                bn_layer = getattr(self, f'bn_{i}')
                out = bn_layer(out, use_running_average=not training)
            
            # Activation
            out = self._get_activation_fn()(out)
            
            # Dropout
            if self.dropout_rate > 0:
                dropout_layer = getattr(self, f'dropout_{i}')
                out = dropout_layer(out, deterministic=not training)
        
        # Output layer
        out = self.output_layer(out)
        
        # Final activation if specified
        if self.final_activation is not None:
            out = self._get_activation_fn(self.final_activation)(out)
        
        return out
    
    def _get_activation_fn(self, activation_name: str = None):
        """Get activation function."""
        if activation_name is None:
            activation_name = self.activation
            
        if activation_name == 'relu':
            return nn.relu
        elif activation_name == 'tanh':
            return nn.tanh
        elif activation_name == 'gelu':
            return nn.gelu
        elif activation_name == 'swish':
            return nn.swish
        elif activation_name == 'sigmoid':
            return nn.sigmoid
        else:
            raise ValueError(f"Unknown activation: {activation_name}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get network configuration."""
        return {
            'hidden_dims': list(self.hidden_dims),
            'output_dim': self.output_dim,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'final_activation': self.final_activation
        }
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


class SimpleMLP(NetworkBase):
    """
    Simplified MLP for quick experimentation.
    
    Minimal configuration MLP with sensible defaults.
    """
    
    hidden_dim: int = 64
    n_layers: int = 3
    output_dim: int = 1
    
    def setup(self):
        """Initialize simple MLP."""
        # Créer les couches directement avec setattr - pas de listes !
        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                setattr(self, f'dense_{i}', nn.Dense(self.hidden_dim))
            else:
                setattr(self, f'dense_{i}', nn.Dense(self.output_dim))
    
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass."""
        out = x
        for i in range(self.n_layers):
            layer = getattr(self, f'dense_{i}')
            out = layer(out)
            # Apply ReLU to all layers except the last
            if i < self.n_layers - 1:
                out = nn.relu(out)
        return out
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        return {
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'output_dim': self.output_dim
        }


class ResidualMLP(NetworkBase):
    """
    MLP with residual connections for deeper networks.
    
    Adds skip connections to allow training of deeper MLPs.
    """
    
    hidden_dims: Sequence[int] = (128, 128, 128)
    output_dim: int = 1
    dropout_rate: float = 0.1
    
    def setup(self):
        """Initialize residual MLP."""
        self.input_proj = nn.Dense(self.hidden_dims[0])
        
        # Residual blocks - utiliser setattr
        for i, dim in enumerate(self.hidden_dims):
            setattr(self, f'residual_block_{i}', ResidualBlock(dim, self.dropout_rate))
        
        self.output_layer = nn.Dense(self.output_dim)
    
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass with residual connections."""
        # Project to hidden dimension
        out = nn.relu(self.input_proj(x))
        
        # Apply residual blocks
        for i in range(len(self.hidden_dims)):
            block = getattr(self, f'residual_block_{i}')
            out = block(out, training=training)
        
        # Output layer
        return self.output_layer(out)
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        return {
            'hidden_dims': list(self.hidden_dims),
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate
        }


class ResidualBlock(nn.Module):
    """Residual block for deeper MLPs."""
    
    hidden_dim: int
    dropout_rate: float = 0.0
    
    def setup(self):
        """Initialize residual block."""
        self.dense1 = nn.Dense(self.hidden_dim)
        self.dense2 = nn.Dense(self.hidden_dim)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass with skip connection."""
        residual = x
        
        out = nn.relu(self.dense1(x))
        out = self.dropout(out, deterministic=not training)
        out = self.dense2(out)
        
        # Skip connection
        out = out + residual
        return nn.relu(out)


# Export classes
__all__ = ["MLPNetwork", "SimpleMLP", "ResidualMLP"]