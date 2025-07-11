"""
Neural Ratio Estimation (NRE) module for ABC inference.

This module provides neural network architectures and training utilities
for estimating likelihood ratios in ABC inference.
"""

# Main estimator
from .estimator import NeuralRatioEstimator, ABCClassifier

# Network architectures
from .networks.base import NetworkBase
from .networks.mlp import MLPNetwork, SimpleMLP, ResidualMLP
from .networks.deepset import DeepSetNetwork, CompactDeepSetNetwork

# Training utilities
from .trainer import (
    TrainingState, 
    train_step, 
    evaluate_step, 
    binary_cross_entropy_loss,
    compute_accuracy,
    EarlyStopping,
    train_with_validation
)

# Configuration
from .config import (
    NetworkConfig,
    TrainingConfig,
    ExperimentConfig,
    get_experiment_config
)

# Utilities
from .utils import (
    save_model_config,
    load_model_config,
    compute_metrics,
    plot_training_history,
    plot_posterior_comparison,
    compute_posterior_statistics,
    effective_sample_size,
    importance_sampling_diagnostics
)

# Validation
from .validation import (
    validate_classifier_performance,
    validate_posterior_quality,
    compute_calibration_metrics,
    plot_calibration_curve,
    plot_roc_curve,
    run_comprehensive_validation
)

# Diagnostics Integration
from .diagnostics_integration import (
    InferenceResultsPackage,
    create_sbc_experiment,
    prepare_posterior_analysis,
    export_for_diagnostics
)

# Persistence
from .persistence import (
    ModelRegistry,
    EstimatorCheckpoint,
    ExperimentManager,
    export_model_for_deployment,
    load_model_for_deployment
)

__all__ = [
    # Main classes
    "NeuralRatioEstimator",
    "ABCClassifier",
    
    # Networks
    "NetworkBase",
    "MLPNetwork",
    "SimpleMLP", 
    "ResidualMLP",
    "DeepSetNetwork",
    "CompactDeepSetNetwork",
    
    # Training
    "TrainingState",
    "train_step",
    "evaluate_step",
    "binary_cross_entropy_loss",
    "compute_accuracy",
    "EarlyStopping",
    "train_with_validation",
    
    # Configuration
    "NetworkConfig",
    "TrainingConfig",
    "InferenceConfig",
    "ExperimentConfig",
    "create_mlp_experiment",
    "create_deepset_experiment",
    
    # Utilities
    "save_model_config",
    "load_model_config",
    "compute_metrics",
    "plot_training_history",
    "plot_posterior_comparison",
    "compute_posterior_statistics",
    "effective_sample_size",
    "importance_sampling_diagnostics",
    
    # Validation
    "validate_classifier_performance",
    "validate_posterior_quality",
    "compute_calibration_metrics",
    "plot_calibration_curve",
    "plot_roc_curve",
    "run_comprehensive_validation",
    
    # Diagnostics Integration
    "InferenceResultsPackage",
    "create_sbc_experiment",
    "prepare_posterior_analysis",
    "export_for_diagnostics",
    
    # Persistence
    "ModelRegistry",
    "EstimatorCheckpoint",
    "ExperimentManager",
    "export_model_for_deployment",
    "load_model_for_deployment"
]