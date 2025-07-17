# src/abcnre/diagnostics/classifier.py

from typing import Dict, Any, Callable
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve
)

# To avoid circular imports, we use TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..inference.estimator import NeuralRatioEstimator
    from ..simulation.base import ABCTrainingResult


def validate_classifier_performance(
    estimator: 'NeuralRatioEstimator',
    test_data_generator: Callable[['jax.random.PRNGKey', int], 'ABCTrainingResult'],
    num_test_samples: int = 10000,
    batch_size: int = 256
) -> Dict[str, Any]:
    """
    Performs a comprehensive validation of the classifier's performance.

    Args:
        estimator: The trained NeuralRatioEstimator.
        test_data_generator: A function to generate batches of test data.
        num_test_samples: The total number of samples to evaluate on.
        batch_size: The size of each batch for processing.

    Returns:
        A dictionary of comprehensive validation metrics.
    """
    if not estimator.is_trained:
        raise ValueError("Estimator must be trained before validation.")
    
    all_probs = []
    all_labels = []
    num_batches = (num_test_samples + batch_size - 1) // batch_size
    
    key = estimator.key

    for _ in range(num_batches):
        key, batch_key = jax.random.split(key)
        test_batch = test_data_generator(batch_key, batch_size)
        
        # Predict probabilities for the current batch
        probs = estimator.predict(test_batch.features)
        
        all_probs.append(np.array(probs))
        all_labels.append(np.array(test_batch.labels))
    
    # Concatenate all results
    predicted_probs = np.concatenate(all_probs, axis=0)[:num_test_samples].flatten()
    true_labels = np.concatenate(all_labels, axis=0)[:num_test_samples]
    
    # --- Compute all relevant metrics ---
    binary_preds = (predicted_probs > 0.5).astype(int)
    
    accuracy = accuracy_score(true_labels, binary_preds)
    precision = precision_score(true_labels, binary_preds, zero_division=0)
    recall = recall_score(true_labels, binary_preds, zero_division=0)
    f1 = f1_score(true_labels, binary_preds, zero_division=0)
    
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    
    prec_curve, rec_curve, _ = precision_recall_curve(true_labels, predicted_probs)
    pr_auc = auc(rec_curve, prec_curve)
    
    calibration_metrics = compute_calibration_metrics(predicted_probs, true_labels)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        **calibration_metrics
    }


def compute_calibration_metrics(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    num_bins: int = 15
) -> Dict[str, float]:
    """
    Computes calibration metrics for a binary classifier.

    Args:
        predicted_probs: An array of predicted probabilities (between 0 and 1).
        true_labels: An array of true binary labels (0 or 1).
        num_bins: The number of bins to use for the calibration calculation.

    Returns:
        A dictionary of calibration metrics (ECE, MCE, Brier score).
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0  # Expected Calibration Error
    mce = 0.0  # Maximum Calibration Error
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(true_labels[in_bin])
            avg_confidence_in_bin = np.mean(predicted_probs[in_bin])
            
            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * calibration_error
            mce = max(mce, calibration_error)
            
    brier_score = np.mean((predicted_probs - true_labels) ** 2)
    
    return {
        'expected_calibration_error': float(ece),
        'maximum_calibration_error': float(mce),
        'brier_score': float(brier_score)
    }