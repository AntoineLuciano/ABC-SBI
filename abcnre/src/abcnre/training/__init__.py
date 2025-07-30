"""
Unified training module for neural networks.

This module provides a unified interface for training both classifier
and summary learner neural networks with consistent configuration and
stopping rules.
"""

from .config import (
    NNConfig,
    NetworkConfig,
    TrainingConfig,
    LRSchedulerConfig,
    StoppingRulesConfig,
    get_nn_config,
    get_quick_nn_config,
)

from .train import (
    train_nn,
    train_classifier,
    train_summary_learner,
    TrainingResult,
  
)

from .registry import (
    create_network_from_config,
    create_network_from_nn_config,
    create_summary_network,
    create_classifier_network,
)

__all__ = [
    # Configuration classes
    "NNConfig",
    "NetworkConfig",
    "TrainingConfig",
    "LRSchedulerConfig",
    "StoppingRulesConfig",
    # Configuration functions
    "get_nn_config",
    "get_quick_nn_config",
    # Training functions
    "train_nn",
    "train_classifier",
    "train_summary_learner",
    # Result classes
    "TrainingResult",
    # Network creation functions
    "create_network_from_config",
    "create_network_from_nn_config",
    "create_summary_network",
    "create_classifier_network",
]
