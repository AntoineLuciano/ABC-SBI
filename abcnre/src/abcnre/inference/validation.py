"""
Validation and advanced metrics for neural ratio estimation.

This module provides comprehensive validation tools and advanced metrics
for evaluating the quality of neural ratio estimators in ABC inference.
"""

from typing import Dict, Any, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve




def validate_classifier_performance(
    estimator,
    test_data_generator,
    num_test_batches: int = 10,
    batch_size: int = 256
) -> Dict[str, Any]:
    """
    Comprehensive validation of classifier performance.
    
    Args:
        estimator: Trained NeuralRatioEstimator
        test_data_generator: Function to generate test data
        num_test_batches: Number of test batches to evaluate
        batch_size: Size of each test batch
        
    Returns:
        Dictionary of validation metrics
    """
    if not estimator.is_trained:
        raise ValueError("Estimator must be trained before validation")
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    key = random.PRNGKey(42)
    
    # Collect predictions over multiple batches
    for i in range(num_test_batches):
        key, batch_key = random.split(key)
        test_batch = test_data_generator(batch_key, batch_size)
        
        # Predict probabilities
        predictions = estimator.predict(test_batch.features)
        
        # Build full variable dict for apply_fn so params/batch_stats are present
        variables = {'params': estimator.state.params}
        if hasattr(estimator.state, 'batch_stats'):
            variables['batch_stats'] = estimator.state.batch_stats
        
        # Compute raw logits
        logits = estimator.state.apply_fn(
            variables,
            test_batch.features,
            training=False
        )
        
        all_predictions.append(predictions)
        all_labels.append(test_batch.labels)
        all_logits.append(logits)
    
    # Concatenate all results
    all_predictions = jnp.concatenate(all_predictions, axis=0)
    all_labels      = jnp.concatenate(all_labels,      axis=0)
    all_logits      = jnp.concatenate(all_logits,      axis=0)
    
    # Convert to numpy for sklearn metrics
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_curve, auc, precision_recall_curve
    )
    
    preds_np  = np.array(all_predictions).flatten()
    labels_np = np.array(all_labels)
    logits_np = np.array(all_logits).flatten()
    
    # Binary predictions
    binary_preds = (preds_np > 0.5).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(labels_np, binary_preds)
    precision = precision_score(labels_np, binary_preds, zero_division=0)
    recall = recall_score(labels_np, binary_preds, zero_division=0)
    f1 = f1_score(labels_np, binary_preds, zero_division=0)
    
    fpr, tpr, _ = roc_curve(labels_np, preds_np)
    roc_auc = auc(fpr, tpr)
    
    prec_curve, rec_curve, _ = precision_recall_curve(labels_np, preds_np)
    pr_auc = auc(rec_curve, prec_curve)
    
    calibration_metrics = compute_calibration_metrics(preds_np, labels_np)
    log_likelihood = compute_log_likelihood(logits_np, labels_np)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'log_likelihood': float(log_likelihood),
        'num_samples': len(labels_np),
        **calibration_metrics
    }




def compute_calibration_metrics(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics for binary classifier.
    
    Args:
        predicted_probs: Predicted probabilities
        true_labels: True binary labels
        num_bins: Number of bins for calibration
        
    Returns:
        Dictionary of calibration metrics
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0  # Expected Calibration Error
    mce = 0.0  # Maximum Calibration Error
    reliability = 0.0  # Reliability (weighted average calibration error)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = true_labels[in_bin].mean()
            avg_confidence_in_bin = predicted_probs[in_bin].mean()
            
            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * calibration_error
            mce = max(mce, calibration_error)
            reliability += prop_in_bin * calibration_error
    
    # Brier score
    brier_score = np.mean((predicted_probs - true_labels) ** 2)
    
    return {
        'expected_calibration_error': float(ece),
        'maximum_calibration_error': float(mce),
        'reliability': float(reliability),
        'brier_score': float(brier_score)
    }


def compute_log_likelihood(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute average log-likelihood of predictions.
    
    Args:
        logits: Raw network outputs
        labels: True binary labels
        
    Returns:
        Average log-likelihood
    """
    # Convert to probabilities
    probs = 1 / (1 + np.exp(-logits))
    
    # Compute log-likelihood
    log_probs = labels * np.log(probs + 1e-15) + (1 - labels) * np.log(1 - probs + 1e-15)
    
    return np.mean(log_probs)


def validate_posterior_quality(
    abc_samples: jnp.ndarray,
    nre_samples: jnp.ndarray,
    true_value: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate quality of NRE posterior against ABC posterior.
    
    Args:
        abc_samples: ABC posterior samples
        nre_samples: NRE posterior samples
        true_value: True parameter value (if known)
        
    Returns:
        Dictionary of posterior quality metrics
    """
    abc_samples = np.array(abc_samples)
    nre_samples = np.array(nre_samples)
    
    # Basic statistics comparison
    abc_mean = np.mean(abc_samples)
    nre_mean = np.mean(nre_samples)
    abc_std = np.std(abc_samples)
    nre_std = np.std(nre_samples)
    
    # Statistical tests
    # Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = stats.ks_2samp(abc_samples, nre_samples)
    
    # Wasserstein distance
    wasserstein_distance = stats.wasserstein_distance(abc_samples, nre_samples)
    
    # Jensen-Shannon divergence (approximate using histograms)
    js_divergence = compute_js_divergence(abc_samples, nre_samples)
    
    # Coverage probability (if true value is known)
    coverage_metrics = {}
    if true_value is not None:
        coverage_metrics.update(compute_coverage_metrics(
            abc_samples, nre_samples, true_value
        ))
    
    return {
        'abc_mean': float(abc_mean),
        'nre_mean': float(nre_mean),
        'abc_std': float(abc_std),
        'nre_std': float(nre_std),
        'mean_difference': float(abs(abc_mean - nre_mean)),
        'std_ratio': float(nre_std / abc_std) if abc_std > 0 else np.inf,
        'ks_statistic': float(ks_statistic),
        'ks_pvalue': float(ks_pvalue),
        'wasserstein_distance': float(wasserstein_distance),
        'js_divergence': float(js_divergence),
        **coverage_metrics
    }


def compute_js_divergence(samples1: np.ndarray, samples2: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two sample distributions.
    
    Args:
        samples1: First set of samples
        samples2: Second set of samples
        
    Returns:
        Jensen-Shannon divergence
    """
    # Create histograms
    min_val = min(np.min(samples1), np.min(samples2))
    max_val = max(np.max(samples1), np.max(samples2))
    bins = np.linspace(min_val, max_val, 50)
    
    hist1, _ = np.histogram(samples1, bins=bins, density=True)
    hist2, _ = np.histogram(samples2, bins=bins, density=True)
    
    # Normalize to probabilities
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-15
    hist1 = hist1 + eps
    hist2 = hist2 + eps
    
    # Compute Jensen-Shannon divergence
    m = 0.5 * (hist1 + hist2)
    js_div = 0.5 * np.sum(hist1 * np.log(hist1 / m)) + 0.5 * np.sum(hist2 * np.log(hist2 / m))
    
    return js_div


def compute_coverage_metrics(
    abc_samples: np.ndarray,
    nre_samples: np.ndarray,
    true_value: float
) -> Dict[str, float]:
    """
    Compute coverage metrics for posterior estimates.
    
    Args:
        abc_samples: ABC posterior samples
        nre_samples: NRE posterior samples
        true_value: True parameter value
        
    Returns:
        Dictionary of coverage metrics
    """
    coverage_levels = [0.5, 0.68, 0.95, 0.99]
    metrics = {}
    
    for level in coverage_levels:
        alpha = 1 - level
        
        # ABC coverage
        abc_lower = np.percentile(abc_samples, 100 * alpha / 2)
        abc_upper = np.percentile(abc_samples, 100 * (1 - alpha / 2))
        abc_coverage = (abc_lower <= true_value <= abc_upper)
        
        # NRE coverage
        nre_lower = np.percentile(nre_samples, 100 * alpha / 2)
        nre_upper = np.percentile(nre_samples, 100 * (1 - alpha / 2))
        nre_coverage = (nre_lower <= true_value <= nre_upper)
        
        # Interval widths
        abc_width = abc_upper - abc_lower
        nre_width = nre_upper - nre_lower
        
        metrics[f'abc_coverage_{int(level*100)}'] = float(abc_coverage)
        metrics[f'nre_coverage_{int(level*100)}'] = float(nre_coverage)
        metrics[f'abc_width_{int(level*100)}'] = float(abc_width)
        metrics[f'nre_width_{int(level*100)}'] = float(nre_width)
        metrics[f'width_ratio_{int(level*100)}'] = float(nre_width / abc_width) if abc_width > 0 else np.inf
    
    return metrics


def plot_calibration_curve(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    num_bins: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Plot calibration curve for binary classifier.
    
    Args:
        predicted_probs: Predicted probabilities
        true_labels: True binary labels
        num_bins: Number of bins for calibration
        save_path: Path to save the plot
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = true_labels[in_bin].mean()
            avg_confidence_in_bin = predicted_probs[in_bin].mean()
            
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(np.sum(in_bin))
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Plot calibration curve
    plt.plot(bin_confidences, bin_accuracies, 'o-', label='Calibration curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Add histogram
    plt.hist(predicted_probs, bins=num_bins, alpha=0.3, density=True, 
             label='Prediction distribution')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve for binary classifier.
    
    Args:
        predicted_probs: Predicted probabilities
        true_labels: True binary labels
        save_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def run_comprehensive_validation(
    estimator,
    abc_simulator,
    num_test_batches: int = 10,
    batch_size: int = 256,
    num_posterior_samples: int = 1000,
    true_value: float = None,
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Run full validation: classifier + posterior quality + summary.

    Returns a dict with keys:
      'classifier_performance', 'posterior_quality', 'validation_summary'
    """
    # 1. Classifier performance
    print("1. Validating classifier performance...")
    classifier_metrics = validate_classifier_performance(
        estimator,
        abc_simulator.generate_training_samples,
        num_test_batches=num_test_batches,
        batch_size=batch_size
    )

    # 2. Posterior sampling
    print("2. Generating posterior samples...")
    key = random.PRNGKey(0)
    key, samp_key = random.split(key)

    abc_batch = abc_simulator.generate_samples(samp_key, num_posterior_samples)
    theta_samples = abc_batch.theta_samples

    obs_mean = jnp.mean(abc_simulator.observed_data)
    features = jnp.column_stack([
        jnp.repeat(obs_mean, num_posterior_samples),
        theta_samples
    ])

    # Calcul des poids et échantillonnage par importance-resampling
    weights = estimator.posterior_weights(features)
    normalized_weights = weights / jnp.sum(weights)

    key, res_key = random.split(key)
    indices = random.choice(
        res_key,
        num_posterior_samples,
        shape=(num_posterior_samples,),
        p=normalized_weights
    )
    nre_samples = theta_samples[indices]

    # 3. Posterior quality
    print("3. Validating posterior quality...")
    posterior_metrics = validate_posterior_quality(
        abc_batch.theta_samples,
        nre_samples,
        true_value=true_value
    )

    # 4. Récapitulatif
    summary = {
        'classifier_accuracy': classifier_metrics['accuracy'],
        'classifier_auc': classifier_metrics['roc_auc'],
        'calibration_error': classifier_metrics.get('expected_calibration_error'),
        'posterior_ks_pvalue': posterior_metrics['ks_pvalue'],
        'posterior_wasserstein': posterior_metrics['wasserstein_distance'],
        'posterior_mean_diff': posterior_metrics['mean_difference']
    }

    return {
        'classifier_performance': classifier_metrics,
        'posterior_quality': posterior_metrics,
        'validation_summary': summary
    }
