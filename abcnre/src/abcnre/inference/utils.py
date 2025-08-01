"""
OUTDATED
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def save_model_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save model configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save configuration
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    config_with_timestamp = {
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    with open(filepath, 'w') as f:
        yaml.dump(config_with_timestamp, f, default_flow_style=False)


def load_model_config(filepath: str) -> Dict[str, Any]:
    """
    Load model configuration from YAML file.
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    return data['config']


def compute_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for binary classification.
    
    Args:
        logits: Network outputs
        labels: True labels
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy for easier computation
    logits = np.array(logits)
    labels = np.array(labels)
    
    # Compute predictions
    probabilities = 1 / (1 + np.exp(-logits))
    predictions = probabilities > 0.5
    
    # Basic metrics
    accuracy = np.mean(predictions == labels)
    
    # Compute precision, recall, F1
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Compute AUC-ROC
    from sklearn.metrics import roc_auc_score
    auc_roc = roc_auc_score(labels, probabilities)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'auc_roc': float(auc_roc)
    }


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['train_accuracy'], label='Train Accuracy')
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_posterior_comparison(
    abc_samples: jnp.ndarray,
    nre_samples: jnp.ndarray,
    true_value: Optional[float] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison between ABC and NRE posterior estimates.
    
    Args:
        abc_samples: ABC posterior samples
        nre_samples: NRE posterior samples
        true_value: True parameter value (if known)
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(abc_samples, bins=50, alpha=0.7, label='ABC Posterior', density=True)
    plt.hist(nre_samples, bins=50, alpha=0.7, label='NRE Posterior', density=True)
    
    # Plot true value if provided
    if true_value is not None:
        plt.axvline(true_value, color='red', linestyle='--', label='True Value')
    
    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
    plt.title('Posterior Comparison: ABC vs NRE')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_log_ratio_distribution(
    log_ratios: jnp.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of estimated log ratios.
    
    Args:
        log_ratios: Estimated log likelihood ratios
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(log_ratios, bins=50, alpha=0.7, density=True)
    
    # Add statistics
    mean_val = np.mean(log_ratios)
    std_val = np.std(log_ratios)
    plt.axvline(mean_val, color='red', linestyle='-', label=f'Mean: {mean_val:.3f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'Mean ± Std')
    plt.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
    
    plt.xlabel('Log Likelihood Ratio')
    plt.ylabel('Density')
    plt.title('Distribution of Log Likelihood Ratios')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compute_posterior_statistics(
    samples: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None
) -> Dict[str, float]:
    """
    Compute statistics for posterior samples.
    
    Args:
        samples: Posterior samples
        weights: Optional weights for samples
        
    Returns:
        Dictionary of posterior statistics
    """
    samples = np.array(samples)
    
    if weights is not None:
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Weighted statistics
        mean = np.average(samples, weights=weights)
        variance = np.average((samples - mean) ** 2, weights=weights)
        std = np.sqrt(variance)
        
        # Weighted quantiles
        sorted_indices = np.argsort(samples)
        sorted_samples = samples[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumulative_weights = np.cumsum(sorted_weights)
        
        q25_idx = np.searchsorted(cumulative_weights, 0.25)
        q50_idx = np.searchsorted(cumulative_weights, 0.5)
        q75_idx = np.searchsorted(cumulative_weights, 0.75)
        
        q25 = sorted_samples[q25_idx]
        q50 = sorted_samples[q50_idx]
        q75 = sorted_samples[q75_idx]
        
    else:
        # Unweighted statistics
        mean = np.mean(samples)
        std = np.std(samples)
        q25 = np.percentile(samples, 25)
        q50 = np.percentile(samples, 50)
        q75 = np.percentile(samples, 75)
    
    return {
        'mean': float(mean),
        'std': float(std),
        'q25': float(q25),
        'median': float(q50),
        'q75': float(q75),
        'min': float(np.min(samples)),
        'max': float(np.max(samples))
    }


def effective_sample_size(weights: jnp.ndarray) -> float:
    """
    Compute effective sample size for weighted samples.
    
    Args:
        weights: Sample weights
        
    Returns:
        Effective sample size
    """
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize
    
    # Effective sample size
    ess = 1.0 / np.sum(weights ** 2)
    
    return float(ess)


def importance_sampling_diagnostics(
    log_ratios: jnp.ndarray,
    weights: jnp.ndarray
) -> Dict[str, float]:
    """
    Compute diagnostics for importance sampling.
    
    Args:
        log_ratios: Log likelihood ratios
        weights: Importance weights
        
    Returns:
        Dictionary of diagnostics
    """
    log_ratios = np.array(log_ratios)
    weights = np.array(weights)
    
    # Basic statistics
    ess = effective_sample_size(weights)
    rel_ess = ess / len(weights)
    
    # Weight statistics
    weight_cv = np.std(weights) / np.mean(weights)  # Coefficient of variation
    max_weight = np.max(weights)
    weight_ratio = max_weight / np.mean(weights)
    
    # Log ratio statistics
    log_ratio_mean = np.mean(log_ratios)
    log_ratio_std = np.std(log_ratios)
    
    return {
        'effective_sample_size': float(ess),
        'relative_ess': float(rel_ess),
        'weight_cv': float(weight_cv),
        'max_weight_ratio': float(weight_ratio),
        'log_ratio_mean': float(log_ratio_mean),
        'log_ratio_std': float(log_ratio_std)
    }


def save_results(
    results: Dict[str, Any],
    filepath: str,
    format: str = 'json'
) -> None:
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        filepath: Path to save results
        format: File format ('json' or 'yaml')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    results_with_timestamp = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results_with_timestamp, f, indent=2, default=str)
    elif format == 'yaml':
        with open(filepath, 'w') as f:
            yaml.dump(results_with_timestamp, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Results dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif filepath.suffix in ['.yaml', '.yml']:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    return data['results']


def create_experiment_directory(
    base_path: str,
    experiment_name: str
) -> Path:
    """
    Create directory structure for experiment.
    
    Args:
        base_path: Base path for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_path) / f"{experiment_name}_{timestamp}"
    
    # Create subdirectories
    (exp_dir / "configs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "models").mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ['network_config', 'network_class', 'learning_rate']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate network configuration
    if not isinstance(config['network_config'], dict):
        raise ValueError("network_config must be a dictionary")
    
    # Validate learning rate
    if not isinstance(config['learning_rate'], (int, float)) or config['learning_rate'] <= 0:
        raise ValueError("learning_rate must be a positive number")
    
    return True