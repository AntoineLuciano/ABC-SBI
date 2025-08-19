# src/abcnre/diagnostics/viz.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc
from typing import Dict, Tuple, Optional, Any, Union, TYPE_CHECKING
import jax.numpy as jnp

if TYPE_CHECKING:
    from .mcmc import MCMCResults


def plot_posterior_comparison(
    distributions: Dict[str, Any],
    prior_pdf: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    true_value: Optional[float] = None,
    title: str = "Comparison of Posterior Distributions",
    save_path: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
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
        plt.plot(grid, pdf, "k--", lw=2, alpha=0.6, label="Prior")

    if true_value is not None:
        plt.axvline(
            true_value,
            color="r",
            linestyle=":",
            lw=3,
            label=f"True Value = {true_value:.2f}",
        )

    plt.title(title, fontsize=16)
    plt.xlabel("Parameter value (phi)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    if xlim:
        plt.xlim(xlim)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_sbc_ranks(
    sbc_results: Dict[str, Any],
    title: str = "SBC Rank Distribution",
    save_path: Optional[str] = None,
):
    """
    Plots the histogram of ranks from a Simulation-Based Calibration (SBC).
    """
    ranks = sbc_results["ranks"]
    num_posterior_samples = (sbc_results["posterior_phis"][0]).shape[0]

    plt.figure(figsize=(10, 6))
    n_sbc_rounds = len(ranks)

    num_bins = max(num_posterior_samples // 2**3 + 1, 17)

    alpha = 0.05
    lower_bound = stats.binom.ppf(alpha / 2, n_sbc_rounds, 1 / num_bins)
    upper_bound = stats.binom.ppf(1 - alpha / 2, n_sbc_rounds, 1 / num_bins)

    plt.hist(
        ranks,
        bins=num_bins,
        density=False,
        alpha=0.8,
        label="Actual Ranks",
        edgecolor="k",
    )

    expected_count = n_sbc_rounds / num_bins
    plt.axhline(
        expected_count, color="r", linestyle="--", label="Expected Uniform Count"
    )
    plt.fill_between(
        [0, num_posterior_samples],
        lower_bound,
        upper_bound,
        color="gray",
        alpha=0.3,
        label="95% Confidence Interval",
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Rank Statistic")
    plt.ylabel("Frequency (Count)")
    plt.legend()
    # plt.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_roc_curve(
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
):
    """Plots the ROC curve for a binary classifier."""
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_mcmc_output(
    mcmc_results: Union["MCMCResults", Dict[str, Any]],
    chains: Optional[jnp.ndarray] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    true_values: Optional[jnp.ndarray] = None,
    parameter_names: Optional[list] = None,
    title: str = "MCMC Diagnostics",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    autocorrelation: bool = True,
    log_probability_trace: bool = True,
    probability_trace: bool = False,
    show: bool = True,
):
    """
    Simplified MCMC visualization with organized layout:
    - Row 1: Trace plots (max 5 columns)
    - Row 2: Marginal distributions (max 5 columns)
    - Row 3: Autocorrelations (if autocorrelation=True)
    - Row 4: Log probability trace (if log_probability_trace=True and log_probs available)
    - Row 4: Probability trace (if probability_trace=True and log_probs available)

    Args:
        mcmc_results: MCMCResults object or dictionary with 'samples', 'log_probs', etc.
        chains: Optional multiple chains array (n_chains, n_samples, n_dims)
        diagnostics: Optional diagnostics dictionary
        true_values: Optional true parameter values to overlay
        parameter_names: Optional parameter names for labeling
        title: Main title for the figure
        save_path: Optional path to save the figure
        figsize: Figure size (width, height). If None, auto-determined
        autocorrelation: Whether to plot autocorrelation functions
        log_probability_trace: Whether to plot log probability trace
        probability_trace: Whether to plot probability trace (exp of log_probs)
    """
    # Extract samples from results
    if hasattr(mcmc_results, "samples"):
        samples = np.array(mcmc_results.samples)
        log_probs = (
            np.array(mcmc_results.log_probs)
            if hasattr(mcmc_results, "log_probs") and mcmc_results.log_probs is not None
            else None
        )
        acceptance_rate = getattr(mcmc_results, "acceptance_rate", None)
    else:
        samples = np.array(mcmc_results["samples"])
        log_probs = mcmc_results.get("log_probs", None)
        if log_probs is not None:
            log_probs = np.array(log_probs)
        acceptance_rate = mcmc_results.get("acceptance_rate", None)

    n_samples, n_dims = samples.shape

    # Default parameter names
    if parameter_names is None:
        parameter_names = [f"θ_{i+1}" for i in range(n_dims)]

    # Determine layout
    max_cols = min(5, n_dims)  # Maximum 5 columns
    n_rows = 2  # Traces + Marginals by default

    if autocorrelation:
        n_rows += 1
    if (log_probability_trace or probability_trace) and log_probs is not None:
        n_rows += 1

    # Auto-determine figure size if not provided
    if figsize is None:
        width = max(15, max_cols * 3)
        height = n_rows * 4
        figsize = (width, height)

    fig, axes = plt.subplots(n_rows, max_cols, figsize=figsize)

    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif max_cols == 1:
        axes = axes.reshape(-1, 1)

    row_idx = 0

    # Row 1: Trace plots
    for dim in range(n_dims):
        col_idx = dim % max_cols
        if dim >= max_cols:
            # For parameters beyond 5, we wrap to next "logical" row within trace plots
            continue

        ax = axes[row_idx, col_idx]

        # Plot multiple chains if available
        if chains is not None:
            n_chains = chains.shape[0]
            colors = plt.cm.tab10(np.linspace(0, 1, n_chains))
            for chain_idx in range(n_chains):
                ax.plot(
                    chains[chain_idx, :, dim],
                    alpha=0.7,
                    color=colors[chain_idx],
                    label=f"Chain {chain_idx+1}" if dim == 0 else "",
                )
        else:
            ax.plot(range(len(samples)), samples[:, dim], alpha=0.8, color="blue")

        if true_values is not None and dim < len(true_values):
            ax.axhline(
                true_values[dim],
                color="red",
                linestyle="--",
                label="True value" if dim == 0 else "",
                alpha=0.8,
            )

        ax.set_title(f"Trace - {parameter_names[dim]}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(parameter_names[dim])
        if dim == 0 and (chains is not None or true_values is not None):
            ax.legend(fontsize=8)

    # Hide unused trace plot subplots
    for col_idx in range(n_dims, max_cols):
        axes[row_idx, col_idx].axis("off")

    row_idx += 1

    # Row 2: Marginal distributions
    for dim in range(n_dims):
        col_idx = dim % max_cols
        if dim >= max_cols:
            continue

        ax = axes[row_idx, col_idx]

        # Plot marginal posterior
        if chains is not None:
            # Combine all chains for marginal
            all_samples = chains.reshape(-1, n_dims)
            sns.histplot(all_samples[:, dim], kde=True, alpha=0.6, ax=ax)
        else:
            sns.histplot(samples[:, dim], kde=True, alpha=0.6, ax=ax)

        if true_values is not None and dim < len(true_values):
            ax.axvline(
                true_values[dim],
                color="red",
                linestyle="--",
                label="True value",
                alpha=0.8,
            )
            ax.legend()

        ax.set_title(f"Marginal - {parameter_names[dim]}")
        ax.set_xlabel(parameter_names[dim])
        ax.set_ylabel("Density")

    # Hide unused marginal plot subplots
    for col_idx in range(n_dims, max_cols):
        axes[row_idx, col_idx].axis("off")

    row_idx += 1

    # Row 3: Autocorrelations (if requested)
    if autocorrelation:
        for dim in range(min(n_dims, max_cols)):
            ax = axes[row_idx, dim]

            if n_samples > 50:
                # Compute autocorrelation
                x = samples[:, dim] - np.mean(samples[:, dim])
                autocorr = np.correlate(x, x, mode="full")
                autocorr = autocorr[autocorr.size // 2 :]
                autocorr = autocorr / autocorr[0]

                max_lag = min(200, len(autocorr) // 4)
                lags = np.arange(max_lag)

                ax.plot(lags, autocorr[:max_lag])
                ax.axhline(0, color="black", linestyle="-", alpha=0.3)
                ax.axhline(
                    0.1, color="red", linestyle="--", alpha=0.7, label="10% threshold"
                )
                ax.set_title(f"Autocorr - {parameter_names[dim]}")
                ax.set_xlabel("Lag")
                ax.set_ylabel("Autocorrelation")
                if dim == 0:
                    ax.legend()
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Need more\nsamples",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"Autocorr - {parameter_names[dim]}")

        # Hide unused autocorrelation subplots
        for col_idx in range(min(n_dims, max_cols), max_cols):
            axes[row_idx, col_idx].axis("off")

        row_idx += 1

    # Row 4: Probability traces (if requested and available)
    if log_probs is not None and (log_probability_trace or probability_trace):
        # Use first subplot for probability trace
        ax = axes[row_idx, 0]

        if log_probability_trace:
            ax.plot(range(len(log_probs)), log_probs, alpha=0.8, color="green")
            ax.set_title("Log Probability Trace")
            ax.set_ylabel("Log Probability")
        elif probability_trace:
            probs = np.exp(log_probs - np.max(log_probs))  # Normalize to avoid overflow
            ax.plot(range(len(probs)), probs, alpha=0.8, color="purple")
            ax.set_title("Probability Trace (normalized)")
            ax.set_ylabel("Probability")

        ax.set_xlabel("Iteration")

        # Hide other subplots in probability row
        for col_idx in range(1, max_cols):
            axes[row_idx, col_idx].axis("off")

    # Overall title and layout
    title_text = title
    if acceptance_rate is not None:
        title_text += f" (Acceptance rate: {acceptance_rate:.3f})"

    fig.suptitle(title_text, fontsize=16, y=0.95)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()


def plot_1D_marginal_comparison(
    samples_dict: Dict[str, np.ndarray],
    parameter_names: Optional[list] = None,
    true_values: Optional[np.ndarray] = None,
    title: str = "1D Marginal Distributions Comparison",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    xlims: Optional[list[float]] = None,
    show: bool = True,
):
    """
    Compare 1D marginal distributions across different sampling methods.

    Args:
        samples_dict: Dictionary with sample_name as keys and samples arrays as values
                     Each array should have shape (n_samples, n_dims) or (n_samples,) for 1D
        parameter_names: Optional list of parameter names for labeling
        true_values: Optional true parameter values to overlay as vertical lines
        title: Main title for the figure
        save_path: Optional path to save the figure
        figsize: Figure size (width, height). If None, auto-determined from n_dims
    """
    # Normalize all samples to 2D arrays and get dimensions
    normalized_samples = {}
    n_dims = None

    for sample_name, samples in samples_dict.items():
        samples_array = np.array(samples)
        if samples_array.ndim == 1:
            # Convert 1D to 2D with single column
            samples_array = samples_array.reshape(-1, 1)
        elif samples_array.ndim != 2:
            raise ValueError(f"Samples for '{sample_name}' must be 1D or 2D array")

        normalized_samples[sample_name] = samples_array

        # Determine number of dimensions
        if n_dims is None:
            n_dims = samples_array.shape[1]
        elif n_dims != samples_array.shape[1]:
            raise ValueError(
                f"All samples must have same number of dimensions. "
                f"Got {n_dims} and {samples_array.shape[1]}"
            )

    # Default parameter names
    if parameter_names is None:
        parameter_names = [f"θ_{i+1}" for i in range(n_dims)]

    # Determine figure size and layout
    if figsize is None:
        if n_dims <= 4:
            figsize = (15, 4 * ((n_dims + 1) // 2))
        else:
            figsize = (20, 5 * ((n_dims + 2) // 3))

    # Subplot layout
    if n_dims == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]
    elif n_dims <= 4:
        n_cols = 2
        n_rows = (n_dims + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes[0], axes[1]]
    else:
        n_cols = 3
        n_rows = (n_dims + 2) // 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

    # Colors for different methods
    colors = plt.cm.Set1(np.linspace(0, 1, len(normalized_samples)))

    # Plot marginals for each dimension
    for dim in range(n_dims):
        ax = axes[dim]

        # Plot KDE for each sampling method
        for (sample_name, samples), color in zip(normalized_samples.items(), colors):
            sns.kdeplot(
                samples[:, dim],
                ax=ax,
                label=sample_name,
                color=color,
                linewidth=2,
                alpha=0.8,
            )

        # Add true value if provided
        if true_values is not None and dim < len(true_values):
            ax.axvline(
                true_values[dim],
                color="red",
                linestyle="--",
                linewidth=2,
                label="True value" if dim == 0 else "",
                alpha=0.8,
            )

        ax.set_title(f"Marginal: {parameter_names[dim]}", fontsize=12)
        ax.set_xlabel(parameter_names[dim])
        ax.set_ylabel("Density")

        # Add legend only to first subplot
        if dim == 0:
            ax.legend()

    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=16, y=0.98)
    # plt.tight_layout()
    if xlims:
        for ax, xlim in zip(axes, xlims):
            ax.set_xlim(xlim)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()


def plot_2D_marginal_comparison(
    samples_dict: Dict[str, np.ndarray],
    parameter_names: Optional[list] = None,
    true_values: Optional[np.ndarray] = None,
    title: str = "2D Marginal Distributions Comparison",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    alpha: float = 0.6,
    show: bool = True,
):
    """
    Compare 2D marginal distributions across different sampling methods.

    Args:
        samples_dict: Dictionary with sample_name as keys and samples arrays as values
                     Each array should have shape (n_samples, n_dims) or (n_samples,) for 1D
        parameter_names: Optional list of parameter names for labeling
        true_values: Optional true parameter values to overlay as points
        title: Main title for the figure
        save_path: Optional path to save the figure
        figsize: Figure size (width, height). If None, auto-determined from n_dims
        alpha: Transparency for scatter plots
    """
    # Normalize all samples to 2D arrays and get dimensions
    normalized_samples = {}
    n_dims = None

    for sample_name, samples in samples_dict.items():
        samples_array = np.array(samples)
        if samples_array.ndim == 1:
            # Convert 1D to 2D with single column
            samples_array = samples_array.reshape(-1, 1)
        elif samples_array.ndim != 2:
            raise ValueError(f"Samples for '{sample_name}' must be 1D or 2D array")

        normalized_samples[sample_name] = samples_array

        # Determine number of dimensions
        if n_dims is None:
            n_dims = samples_array.shape[1]
        elif n_dims != samples_array.shape[1]:
            raise ValueError(
                f"All samples must have same number of dimensions. "
                f"Got {n_dims} and {samples_array.shape[1]}"
            )

    if n_dims < 2:
        print("Warning: Need at least 2 dimensions for 2D marginal plots")
        return

    # Default parameter names
    if parameter_names is None:
        parameter_names = [f"θ_{i+1}" for i in range(n_dims)]

    # Calculate number of 2D combinations
    n_pairs = n_dims * (n_dims - 1) // 2

    # Determine figure size and layout
    if figsize is None:
        if n_pairs <= 3:
            figsize = (15, 5)
            n_cols = n_pairs
            n_rows = 1
        elif n_pairs <= 6:
            figsize = (15, 10)
            n_cols = 3
            n_rows = 2
        else:
            figsize = (20, 15)
            n_cols = 4
            n_rows = (n_pairs + 3) // 4
    else:
        n_cols = min(4, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_pairs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if hasattr(axes, "__len__") else [axes]
    else:
        axes = axes.flatten()

    # Colors for different methods
    colors = plt.cm.Set1(np.linspace(0, 1, len(normalized_samples)))

    # Plot all pairwise 2D marginals
    pair_idx = 0
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            if pair_idx < len(axes):
                ax = axes[pair_idx]

                # Plot samples for each method
                for (sample_name, samples), color in zip(
                    normalized_samples.items(), colors
                ):
                    ax.scatter(
                        samples[:, i],
                        samples[:, j],
                        label=sample_name,
                        color=color,
                        alpha=alpha,
                        s=1,
                    )

                # Add true value if provided
                if (
                    true_values is not None
                    and i < len(true_values)
                    and j < len(true_values)
                ):
                    ax.scatter(
                        true_values[i],
                        true_values[j],
                        color="red",
                        marker="x",
                        s=100,
                        linewidth=3,
                        label="True value" if pair_idx == 0 else "",
                        zorder=10,
                    )

                ax.set_xlabel(parameter_names[i])
                ax.set_ylabel(parameter_names[j])
                ax.set_title(f"{parameter_names[i]} vs {parameter_names[j]}")

                # Add legend only to first subplot
                if pair_idx == 0:
                    ax.legend(markerscale=10)  # Make legend markers bigger

                pair_idx += 1

    # Hide unused subplots
    for i in range(pair_idx, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()


def plot_calibration_curve(
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
    num_bins: int = 15,
    title: str = "Calibration Curve",
    save_path: Optional[str] = None,
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
