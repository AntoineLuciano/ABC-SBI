# src/abcnre/diagnostics/viz.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc
from typing import Dict, Tuple, Optional, Any

def plot_posterior_comparison(
    distributions: Dict[str, Any],
    prior_pdf: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    true_value: Optional[float] = None,
    title: str = "Comparison of Posterior Distributions",
    save_path: Optional[str] = None
):
    """
    Plots a comprehensive comparison of multiple posterior distributions.
    
    This function can plot distributions from samples (using KDE) and from
    pre-computed PDFs (lines).

    Args:
        distributions: A dictionary where keys are labels (e.g., 'NRE')
                       and values are a 1D array of samples or a (grid, pdf) tuple.
        prior_pdf: An optional tuple (grid, pdf_values) for the prior.
        true_value: An optional true parameter value to plot as a vertical line.
        title: The title of the plot.
        save_path: The path to save the figure to.
    """
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(distributions)))

    for i, (label, data) in enumerate(distributions.items()):
        if isinstance(data, (list, np.ndarray)) and np.array(data).ndim == 1:
            sns.kdeplot(data, label=label, linewidth=2.5, alpha=0.8, color=colors[i])
        elif isinstance(data, tuple) and len(data) == 2:
            grid, pdf_values = data
            plt.plot(grid, pdf_values, lw=2.5, alpha=0.8, label=label, color=colors[i])
        else:
            print(f"Warning: Could not plot '{label}', data format is not recognized.")

    if prior_pdf:
        grid, pdf = prior_pdf
        plt.plot(grid, pdf, 'k--', lw=2, alpha=0.6, label='Prior')

    if true_value is not None:
        plt.axvline(true_value, color='r', linestyle=':', lw=3, label=f'True Value = {true_value:.2f}')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Parameter value (phi)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_sbc_ranks(
    ranks: np.ndarray,
    num_posterior_samples: int,
    title: str = "SBC Rank Distribution",
    save_path: Optional[str] = None
):
    """
    Plots the histogram of ranks from a Simulation-Based Calibration (SBC).
    """
    plt.figure(figsize=(10, 6))
    n_sbc_rounds = len(ranks)
    num_bins = max(num_posterior_samples // 2**3 + 1, 17)
    
    alpha = 0.05
    lower_bound = stats.binom.ppf(alpha / 2, n_sbc_rounds, 1 / num_bins)
    upper_bound = stats.binom.ppf(1 - alpha / 2, n_sbc_rounds, 1 / num_bins)

    plt.hist(ranks, bins=num_bins, density=False, alpha=0.8, label='Actual Ranks', edgecolor='k')
    
    expected_count = n_sbc_rounds / num_bins
    plt.axhline(expected_count, color='r', linestyle='--', label='Expected Uniform Count')
    plt.fill_between([0, num_posterior_samples], lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')

    plt.title(title, fontsize=16)
    plt.xlabel('Rank Statistic')
    plt.ylabel('Frequency (Count)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_roc_curve(
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
    title: str = 'ROC Curve',
    save_path: Optional[str] = None
):
    """Plots the ROC curve for a binary classifier."""
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_calibration_curve(
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
    num_bins: int = 15,
    title: str = 'Calibration Curve',
    save_path: Optional[str] = None
):
    """Plots a calibration curve for a binary classifier."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    fraction_of_positives = np.zeros(num_bins)
    mean_predicted_value = np.zeros(num_bins)
    
    for i, bin_lower in enumerate(bin_lowers):
        bin_upper = bin_boundaries[i + 1]
        in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
        if np.sum(in_bin) > 0:
            fraction_of_positives[i] = np.mean(true_labels[in_bin])
            mean_predicted_value[i] = np.mean(predicted_probs[in_bin])
            
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")

    plt.xlabel("Mean Predicted Probability (Confidence)")
    plt.ylabel("Fraction of Positives (Accuracy)")
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()