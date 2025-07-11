# src/abcnre/diagnostics/viz.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Tuple, Optional, Any
import seaborn as sns


def plot_posterior_comparison(
    distributions: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
    prior_pdf: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    true_value: Optional[float] = None,
    title: str = "Comparison of Posterior Distributions",
    save_path: Optional[str] = None
):
    """
    Plots a comprehensive comparison of multiple posterior distributions.

    This function can plot distributions from samples (histograms) and from
    pre-computed PDFs (lines).

    Args:
        distributions: A dictionary where keys are labels (e.g., 'NRE', 'ABC')
                       and values are either a 1D array of samples, or a
                       tuple (grid, pdf_values).
        prior_pdf: An optional tuple (grid, pdf_values) for the prior.
        true_value: An optional true parameter value to plot as a vertical line.
        title: The title of the plot.
        save_path: The path to save the figure to.
    """
    plt.figure(figsize=(12, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # # Get the standard matplotlib color cycle
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # if len(colors) < len(distributions):
    #     # Extend colors if not enough
    #     cmap = plt.get_cmap('tab10')
    #     colors = [cmap(i % cmap.N) for i in range(len(distributions))]

    for i, (label, data) in enumerate(distributions.items()):
        # Check if data is samples (1D array) or a pre-computed PDF (tuple)
        if isinstance(data, (list, np.ndarray)) or (isinstance(data, np.ndarray) and np.array(data).ndim == 1):
            sns.kdeplot(data.flatten(), label=label, linewidth=2.5, alpha=0.8, color=colors[i])
        elif isinstance(data, tuple) and len(data) == 2:
            grid, pdf_values = data
            plt.plot(grid, pdf_values, lw=2.5, alpha=0.8, label=label, color=colors[i])
        
        else:
            print(f"Warning: Could not plot '{label}', data format is not recognized.")

    # Plot the prior if provided
    if prior_pdf:
        grid, pdf = prior_pdf
        plt.plot(grid, pdf,  lw=1, alpha=0.7, label='Prior', color='gray', linestyle='--')

    # Plot the true value if provided
    if true_value is not None:
        plt.axvline(true_value, color='k', linestyle=':', lw=3, label=f'True Value = {true_value:.2f}')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Parameter value (phi)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Diagnostic plot saved to {save_path}")
    
    plt.show()


def plot_sbc_ranks(
    ranks: np.ndarray,
    num_posterior_samples: int,
    title: str = "SBC Rank Distribution",
    save_path: Optional[str] = None
):
    """
    Plots the histogram of ranks from a Simulation-Based Calibration (SBC).

    Also plots the expected uniform distribution and confidence intervals.
    A uniform rank histogram indicates a well-calibrated inference procedure.

    Args:
        ranks: An array of integer ranks from the SBC run.
        num_posterior_samples: The number of posterior samples used to compute each rank.
        title: The title of the plot.
        save_path: The path to save the figure to.
    """
    plt.figure(figsize=(10, 6))
    
    n_sbc_rounds = len(ranks)
    # The number of bins is L+1, where L is the number of posterior samples
    num_bins = max(num_posterior_samples//2**3 + 1, 17)
    print(f"Number of bins for histogram: {num_bins}")

    # Calculate confidence intervals using a binomial distribution
    # This shows the range where counts are expected to fall by chance.
    alpha = 0.05 # 95% confidence interval
    lower_bound = stats.binom.ppf(alpha / 2, n_sbc_rounds, 1 / num_bins)
    upper_bound = stats.binom.ppf(1 - alpha / 2, n_sbc_rounds, 1 / num_bins)

    plt.hist(ranks, bins=num_bins, density=False, alpha=0.8, label='Actual Ranks', color = "red", edgecolor='black')
    
    # Plot expected uniform distribution and confidence bands
    expected_count = n_sbc_rounds / num_bins
    plt.axhline(expected_count, color='k', linestyle='--', label='Expected Uniform Count')
    plt.fill_between([0, num_posterior_samples], lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')

    plt.title(title, fontsize=16)
    plt.xlabel('Rank Statistic')
    plt.ylabel('Frequency (Count)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"SBC plot saved to {save_path}")

    plt.show()