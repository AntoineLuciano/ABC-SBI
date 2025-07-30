"""
Network registry for unified training architecture.

This module provides a centralized registry for creating networks
based on NNConfig specifications, supporting both summary learners
and classifiers with automatic dimension inference.
"""

from typing import Dict, Type, Any, Optional
import jax.numpy as jnp
import logging

from .config import NetworkConfig, NNConfig

logger = logging.getLogger(__name__)

# Network registry will be populated dynamically
NETWORK_REGISTRY = {}


def register_networks():
    """Register classifier networks from training module."""
    try:
        from .networks.mlp import MLP
        from .networks.deepset import DeepSet
        from .networks.conditioned_deepset import ConditionedDeepSet

        NETWORK_REGISTRY.update(
            {
                "MLP": MLP,
                "DeepSet": DeepSet,
                "ConditionedDeepSet": ConditionedDeepSet,
            }
        )
        logger.info("Registered networks: " + ", ".join(NETWORK_REGISTRY.keys()))
    except ImportError as e:
        logger.warning(f"Could not register networks: {e}")


def create_network_from_config(network_config: NetworkConfig, task_type: str) -> Any:
    """
    Create a network instance from NetworkConfig.

    Args:
        network_config: NetworkConfig with network_type and network_args
        task_type: "classifier" or "summary_learner"

    Returns:
        Instantiated network

    Raises:
        ValueError: If network_type is unknown or required args are missing
    """

    register_networks()
    registry = NETWORK_REGISTRY
    network_type = network_config.network_type

    # Handle network type mapping and aliases (case-insensitive)
    # Build a mapping from lower-case names to actual keys
    lower_to_key = {k.lower(): k for k in registry.keys()}
    network_type_key = lower_to_key.get(network_type.lower())

    if network_type_key is None:
        available = list(registry.keys())
        raise ValueError(
            f"Unknown network type '{network_type}' for task '{task_type}'. "
            f"Available: {available}"
        )

    network_class = registry[network_type_key]

    # Get network arguments and ensure output_dim is set
    network_args = network_config.network_args.copy()

    # Always ensure output_dim=1 for all networks (classifier and summary_learner)
    if "output_dim" not in network_args:
        network_args["output_dim"] = 1
        logger.info(f"Set output_dim=1 for {network_type_key} network")

    try:
        # Create network instance
        network = network_class(**network_args)

        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Created {network_type_key} network for {task_type}")
            logger.info(f"Network args: {network_args}")

        return network

    except TypeError as e:
        # Provide helpful error message with expected parameters
        import inspect

        sig = inspect.signature(network_class.__init__)
        params = list(sig.parameters.keys())[1:]  # Skip 'self'

        raise TypeError(
            f"Failed to create {network_type_key} network. "
            f"Error: {e}. "
            f"Expected parameters: {params}. "
            f"Provided: {list(network_args.keys())}"
        ) from e


def create_network_from_nn_config(nn_config: NNConfig) -> Any:
    """
    Create a network from a complete NNConfig.

    Args:
        nn_config: NNConfig instance
    Returns:
        Instantiated network
    """
    return create_network_from_config(nn_config.network, nn_config.task_type)


# Convenience functions for backward compatibility
def create_summary_network(network_type: str, **kwargs) -> Any:
    """Create a summary learning network with legacy interface."""
    from .config import NetworkConfig

    network_config = NetworkConfig(network_type=network_type, network_args=kwargs)

    return create_network_from_config(network_config, "summary_learner")


def create_classifier_network(network_type: str, **kwargs) -> Any:
    """Create a classifier network with legacy interface."""
    from .config import NetworkConfig

    network_config = NetworkConfig(network_type=network_type, network_args=kwargs)

    return create_network_from_config(network_config, "classifier")
