"""
Base class for neural network architectures in NRE.

This module defines the abstract interface that all neural networks
must implement for neural ratio estimation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import flax.linen as nn
import jax.numpy as jnp
import jax
from jax import random


class NetworkBase(nn.Module, ABC):
    """
    Abstract base class for all neural networks used in NRE.
    
    This class defines the interface that neural network architectures
    must implement to be compatible with the NeuralRatioEstimator.
    
    All networks must implement:
    - __call__(): Forward pass for inference
    - get_config(): Return configuration for serialization
    
    Optional methods:
    - setup(): Initialize network components
    - get_feature_dim(): Return feature dimension
    """
    
    @abstractmethod
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass of the network.
        
        Args:
            x: Input features of shape (batch_size, feature_dim)
            training: Whether in training mode (for dropout, batch norm, etc.)
            
        Returns:
            Network output of shape (batch_size, output_dim)
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get network configuration for serialization.
        
        Returns:
            Dictionary containing all parameters needed to recreate the network
        """
        pass
    
    def setup(self):
        """
        Initialize network components.
        
        Override this method to define layers and other components.
        Called automatically by Flax.
        """
        pass
    
    def get_feature_dim(self) -> Optional[int]:
        """
        Get expected input feature dimension.
        
        Returns:
            Expected input dimension, or None if variable
        """
        return None
    
    def get_output_dim(self) -> int:
        """
        Get output dimension.
        
        Returns:
            Output dimension (typically 1 for binary classification)
        """
        return 1
    
    def init_params(self, key: random.PRNGKey, input_shape: Tuple[int, ...]) -> Any:
        """
        Initialize network parameters.
        
        Args:
            key: JAX random key
            input_shape: Shape of input data (batch_size, feature_dim)
            
        Returns:
            Initialized parameters (for backward compatibility)
        """
        dummy_input = jnp.ones(input_shape)
        variables = self.init(key, dummy_input, training=False)
        # Return only params for backward compatibility
        return variables['params']
    
    def init_variables(self, key: random.PRNGKey, input_shape: Tuple[int, ...]) -> Any:
        """
        Initialize all network variables including batch_stats.
        
        Args:
            key: JAX random key
            input_shape: Shape of input data (batch_size, feature_dim)
            
        Returns:
            Dictionary with 'params' and potentially 'batch_stats'
        """
        dummy_input = jnp.ones(input_shape)
        variables = self.init(key, dummy_input, training=False)
        return variables
    
    def count_parameters(self, params: Any) -> int:
        """
        Count total number of parameters in the network.
        
        Args:
            params: Network parameters
            
        Returns:
            Total number of parameters
        """
        return sum(x.size for x in jax.tree_util.tree_leaves(params))
    
    def __repr__(self) -> str:
        """String representation of the network."""
        config = self.get_config()
        config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
        return f"{self.__class__.__name__}({config_str})"
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NetworkBase':
        """
        Factory method to create a network from a configuration dictionary.

        This method intelligently handles configuration dictionaries that may
        contain extra metadata (like 'network_type') by removing such keys
        before initializing the class.

        Args:
            config: The configuration dictionary for the network.

        Returns:
            An instance of the network class (e.g., MLPNetwork).
        """
        # Create a copy to avoid modifying the original dictionary in place.
        init_params = config.copy()

        # Remove keys that are for metadata/selection, not for initialization.
        init_params.pop('network_type', None)

        # 'cls' refers to the specific subclass this is called on (e.g., MLPNetwork).
        # This initializes the network with only the valid parameters.
        return cls(**init_params)


from .mlp import MLPNetwork, SimpleMLP, ResidualMLP
from .deepset import DeepSetNetwork, CompactDeepSetNetwork

# A registry to map string names to network classes
NETWORK_REGISTRY = {
    'MLPNetwork': MLPNetwork,
    'SimpleMLP': SimpleMLP,
    'ResidualMLP': ResidualMLP,
    'DeepSetNetwork': DeepSetNetwork,
    'CompactDeepSetNetwork': CompactDeepSetNetwork,
    # Add any new network class here
}

def create_network_from_config(network_config: Dict[str, Any]) -> NetworkBase:
    """
    Factory function to create a neural network from a configuration dictionary.

    This function reads the 'network_type' from the config, finds the correct
    network class, and instantiates it with the arguments found in the nested
    'network_args' dictionary.

    Args:
        network_config: A dictionary containing 'network_type' and a nested
                        'network_args' dictionary.

    Returns:
        An instantiated neural network object that inherits from NetworkBase.
    """
    # 1. Get the network type string
    network_type = network_config.get('network_type')
    if network_type is None:
        raise ValueError("Configuration dictionary must contain a 'network_type' key.")

    # 2. Get the nested dictionary of arguments for the network's constructor
    network_args = network_config.get('network_args', {})
    
    # 3. Look up the network class in the registry
    NetworkClass = NETWORK_REGISTRY.get(network_type)
    if NetworkClass is None:
        raise ValueError(f"Unknown network_type '{network_type}'. "
                         f"Available types are: {list(NETWORK_REGISTRY.keys())}")

    # 4. Instantiate the class by unpacking the **nested** network_args dictionary
    return NetworkClass(**network_args)


# Update the __all__ list to export the new function
__all__ = ["NetworkBase", "create_network_from_config"]
