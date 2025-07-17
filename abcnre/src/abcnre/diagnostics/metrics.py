# src/abcnre/diagnostics/metrics.py
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sbibm.metrics import c2st as c2st_sbibm
from scipy import stats
from typing import Dict, Union, np, torch, Callable, List, Any, Optional
import jax


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
    return stats.wasserstein_distance(np.asarray(samples_1).flatten(), 
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


def compute_js_divergence(samples1: np.ndarray, samples2: np.ndarray) -> float:
    """Computes Jensen-Shannon divergence between two sample distributions."""
    # (Cette fonction vient de votre fichier validation.py)
    min_val, max_val = min(np.min(samples1), np.min(samples2)), max(np.max(samples1), np.max(samples2))
    bins = np.linspace(min_val, max_val, 50)
    hist1, _ = np.histogram(samples1, bins=bins, density=True)
    hist2, _ = np.histogram(samples2, bins=bins, density=True)
    hist1 = hist1 / np.sum(hist1) + 1e-15
    hist2 = hist2 / np.sum(hist2) + 1e-15
    m = 0.5 * (hist1 + hist2)
    return 0.5 * np.sum(stats.entropy(hist1, m)) + 0.5 * np.sum(stats.entropy(hist2, m))

def compute_coverage_metrics(approx_samples: np.ndarray, true_value: float) -> Dict[str, float]:
    """Computes coverage metrics for posterior estimates."""
    # (Cette fonction vient de votre fichier validation.py)
    coverage_levels = [0.5, 0.68, 0.95, 0.99]
    metrics = {}
    for level in coverage_levels:
        alpha = 1 - level
        lower = np.percentile(approx_samples, 100 * alpha / 2)
        upper = np.percentile(approx_samples, 100 * (1 - alpha / 2))
        coverage = (lower <= true_value <= upper)
        metrics[f'coverage_{int(level*100)}'] = float(coverage)
    return metrics


def run_metrics_suite(
    true_samples: np.ndarray,
    approx_samples_dict: Dict[str, np.ndarray],
    metrics_to_run: List[str] = ['c2st', 'wasserstein', 'mse']
) -> Dict[str, Dict[str, float]]:
    """
    Runs a full suite of metrics to compare a true distribution against
    one or more approximated distributions.

    Args:
        true_samples: Samples from the reference or true posterior.
        approx_samples_dict: A dictionary where keys are approximation names
                             (e.g., 'NRE') and values are the corresponding samples.
        metrics_to_run: A list of metric names to compute.

    Returns:
        A nested dictionary with results for each approximation method.
    """
    print("Running metrics suite...")
    if true_samples is None:
        print("Warning: Cannot compute metrics, true_samples is None.")
        return {}
        
    all_results = {}
    
    for approx_name, approx_samples in approx_samples_dict.items():
        print(f"  Computing metrics for '{approx_name}'...")
        if approx_samples is None:
            print(f"  -> Warning: Skipping '{approx_name}', samples are None.")
            continue

        results = {}
        if 'mse' in metrics_to_run:
            results.update(compute_mse_mean_std(true_samples, approx_samples))
        if 'wasserstein' in metrics_to_run:
            results['wasserstein_distance'] = compute_wasserstein(true_samples, approx_samples)
        if 'c2st' in metrics_to_run:
            results['c2st'] = compute_c2st(true_samples, approx_samples)
        if 'js_divergence' in metrics_to_run:
            results['js_divergence'] = compute_js_divergence(true_samples, approx_samples)
            
        
        all_results[approx_name] = results

    print("Metrics suite complete.")
    return all_results


def generate_and_evaluate_metrics(
    key: 'jax.random.PRNGKey',
    true_sampler: Callable[[int], np.ndarray],
    approx_samplers_dict: Dict[str, Callable[[int], np.ndarray]],
    n_samples: int = 10000,
    metrics_to_run: List[str] = ['c2st', 'wasserstein', 'mse']
) -> Dict[str, Dict[str, float]]:
    """
    Generates samples and runs the metrics suite for multiple approximations.
    """
    print(f"Generating {n_samples} samples for metrics evaluation...")
    
    # 1. Generate true samples once
    key, sample_key = jax.random.split(key)
    true_samples = true_sampler(sample_key, n_samples)
    
    # 2. Generate samples for each approximation method
    approx_samples_dict = {}
    for approx_name, approx_sampler in approx_samplers_dict.items():
        print(f"  -> Generating samples for '{approx_name}'...")
        key, sample_key = jax.random.split(key)
        approx_samples_dict[approx_name] = approx_sampler(sample_key, n_samples)
    
    # 3. Call the low-level function to compute all metrics
    return run_metrics_suite(true_samples, approx_samples_dict, metrics_to_run)


def save_metrics_to_csv(metrics_results: Dict[str, Dict[str, float]], filepath: Path):
    """
    Saves a nested dictionary of metrics to a CSV file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # from_dict with orient='index' creates a DataFrame where each
    # dictionary key becomes a row index.
    df = pd.DataFrame.from_dict(metrics_results, orient='index')
    
    df.to_csv(filepath, index_label='method') # Save the index with a proper name
    print(f"✅ Metrics for all methods saved to {filepath}")