# src/abcnre/diagnostics/metrics.py

import numpy as np
import torch
from sbibm.metrics import c2st as c2st_sbibm
from scipy.stats import wasserstein_distance
from typing import Dict, Union, np, torch

def compute_c2st(
    samples_1: Union[np.ndarray, torch.Tensor],
    samples_2: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes the Classifier 2-Sample Test (C2ST) score.

    This metric trains a classifier to distinguish between two sets of samples.
    An accuracy of ~0.5 indicates the distributions are indistinguishable.
    An accuracy of ~1.0 indicates they are easily separable.

    Args:
        samples_1: Samples from the first distribution.
        samples_2: Samples from the second distribution.

    Returns:
        The C2ST accuracy score.
    """
    # Ensure data is in the correct format (2D tensor) for c2st
    x = torch.as_tensor(samples_1, dtype=torch.float32)
    y = torch.as_tensor(samples_2, dtype=torch.float32)
    
    if x.ndim == 1: x = x.unsqueeze(1)
    if y.ndim == 1: y = y.unsqueeze(1)

    c2st_result = c2st_sbibm(x, y, n_folds=5)
    return float(c2st_result.mean())


def compute_wasserstein(
    samples_1: np.ndarray,
    samples_2: np.ndarray
) -> float:
    """
    Computes the 1D Wasserstein distance between two sets of samples.

    Also known as the Earth Mover's Distance, it measures the "work"
    required to transform one distribution into the other. A smaller
    distance means the distributions are more similar.

    Args:
        samples_1: Samples from the first distribution.
        samples_2: Samples from the second distribution.

    Returns:
        The 1D Wasserstein distance.
    """
    # Scipy's function requires 1D arrays
    return wasserstein_distance(np.asarray(samples_1).flatten(), 
                              np.asarray(samples_2).flatten())


def compute_mse_mean_std(
    samples_1: np.ndarray,
    samples_2: np.ndarray
) -> Dict[str, float]:
    """
    Computes the Mean Squared Error (MSE) between the mean and std dev.

    Args:
        samples_1: Samples from the first (reference) distribution.
        samples_2: Samples from the second (approximated) distribution.

    Returns:
        A dictionary containing the MSE of the mean and std dev.
    """
    mse_mean = (np.mean(samples_1) - np.mean(samples_2)) ** 2
    mse_std = (np.std(samples_1) - np.std(samples_2)) ** 2
    return {
        'mse_mean': float(mse_mean),
        'mse_std': float(mse_std)
    }


def run_metrics_suite(
    true_samples: np.ndarray,
    approx_samples: np.ndarray
) -> Dict[str, float]:
    """
    Runs a full suite of metrics to compare two sets of posterior samples.

    Args:
        true_samples: Samples from the reference or true posterior.
        approx_samples: Samples from the approximated posterior (e.g., NRE).

    Returns:
        A dictionary containing the results of all computed metrics.
    """
    print("Running metrics suite...")
    if true_samples is None or approx_samples is None:
        print("Warning: Cannot compute metrics, one of the sample sets is None.")
        return {}
        
    results = {}
    
    # Run moment-based metrics
    results.update(compute_mse_mean_std(true_samples, approx_samples))

    # Run distribution-based metrics
    results['wasserstein_distance'] = compute_wasserstein(true_samples, approx_samples)
    results['c2st'] = compute_c2st(true_samples, approx_samples)

    print("Metrics suite complete.")
    return results