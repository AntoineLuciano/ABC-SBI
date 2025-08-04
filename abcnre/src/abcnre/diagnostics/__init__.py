# src/abcnre/diagnostics/__init__.py

"""
Diagnostics module for ABC-NRE.

This module provides tools for evaluating the quality of neural ratio estimators,
including simulation-based calibration (SBC) and comparison metrics.
"""

# Original diagnostics functions
from .calibration import run_abc_sbc, save_sbc_results_to_csv
from .metrics import (
    compute_c2st,
    compute_wasserstein,
    compute_mse_mean_std,
    compute_js_divergence,
    compute_coverage_metrics,
    run_metrics_suite,
    generate_and_evaluate_metrics,
    save_metrics_to_csv,
)
from .posterior import (
    get_unnormalized_nre_pdf,
    get_unnormalized_corrected_nre_pdf,
    get_normalized_pdf,
    sample_from_pdf,
    get_sampler_from_pdf,
)


from .sbc_not_ready import (
    SBCConfig,
    run_abc_sbc_with_config,
    run_sbc_from_yaml,
    compute_sbc_ranks,
    validate_config,
    create_example_config,
    )

from .viz import (
    plot_sbc_ranks,
    plot_posterior_comparison,
    plot_roc_curve,
    plot_calibration_curve,
    )

__all__ = [
    # SBC Configuration and Execution
    "SBCConfig",
    "run_abc_sbc_with_config",
    "run_sbc_from_yaml",
    "compute_sbc_ranks",
    "validate_config",
    "create_example_config",
    # Original SBC functions
    "run_abc_sbc",
    "save_sbc_results_to_csv",
    # Metrics functions
    "compute_c2st",
    "compute_wasserstein",
    "compute_mse_mean_std",
    "compute_js_divergence",
    "compute_coverage_metrics",
    "run_metrics_suite",
    "generate_and_evaluate_metrics",
    "save_metrics_to_csv",
    # Posterior functions
    "get_unnormalized_nre_pdf",
    "get_unnormalized_corrected_nre_pdf",
    "get_normalized_pdf",
    "sample_from_pdf",
    "get_sampler_from_pdf",
    # Visualization (if available)
    "plot_sbc_ranks",
    "plot_posterior_comparison",
    "plot_roc_curve",
    "plot_calibration_curve",
]

