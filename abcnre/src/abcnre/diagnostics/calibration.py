import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..inference.estimator import NeuralRatioEstimator
from ..simulation.samplers import ABCSimulator

from .posterior import (
    get_unnormalized_nre_pdf,
    get_normalized_pdf,
    sample_from_pdf,
    get_unnormalized_corrected_nre_pdf,
)


def run_abc_sbc(
    key: jax.random.PRNGKey,
    estimator: "NeuralRatioEstimator",
    num_sbc_rounds: int = 500,
    num_posterior_samples: int = 1000,
    correction: Optional[bool] = False,
    initial_bounds: Optional[Tuple[float, float]] = (0.0, 10.0),
) -> Dict[str, Any]:
    """
    Performs ABC-based Simulation-Based Calibration (ABC-SBC).

    If `true_phis_for_sbc` is provided, it will be used as the set of
    ground truth parameters. Otherwise, `num_sbc_rounds` parameters will be
    randomly drawn from `abc_phi_samples`.

    Args:
        true_phis_for_sbc: An optional array of pre-defined "true" phi values to use for calibration.
    """

    simulator = estimator.simulator
    sampler = simulator.sampler

    key, key_sample = jax.random.split(key)
    abc_results = sampler.sample(key_sample, n_samples=num_sbc_rounds)
    print(
        f"DEBUG: ABC-SBC sampled {num_sbc_rounds} data points. Type abc_results={type(abc_results)}"
    )
    datas = abc_results.data
    phi_samples = abc_results.phi

    if estimator.stored_phis is not None:
        abc_phi_samples = estimator.stored_phis
    else:
        abc_phi_samples = phi_samples

    ranks = []
    posterior_phis = []

    for i in tqdm(range(num_sbc_rounds), desc="ABC-SBC Progress"):

        data = datas[i]
        phi_true = phi_samples[i]

        if correction:
            unnormalized_pdf_func = get_unnormalized_nre_pdf(estimator, x=data)
        else:
            unnormalized_pdf_func = get_unnormalized_corrected_nre_pdf(
                estimator, x=data, phi_samples=abc_phi_samples
            )

        grid, normalized_pdf = get_normalized_pdf(unnormalized_pdf_func, initial_bounds)
        key, sample_key = jax.random.split(key)
        # 3. Draw samples from this NRE posterior
        posterior_samples = sample_from_pdf(
            grid, normalized_pdf, num_posterior_samples, sample_key
        )

        # 4. Compute the rank of phi_true
        rank = jnp.sum(posterior_samples < phi_true)
        ranks.append(int(rank))
        posterior_phis.append(posterior_samples)

    print("ABC-SBC complete.")
    return {
        "ranks": np.array(ranks),
        "phis": np.array(phi_samples),
        "posterior_phis": np.array(posterior_phis),
        "datas": np.array(datas),
    }


def save_sbc_results_to_csv(sbc_results: Dict[str, np.ndarray], filepath: Path):
    """
    Saves the results of a Simulation-Based Calibration run to a CSV file.

    Args:
        sbc_results: The dictionary returned by run_abc_sbc.
        filepath: The path to the output CSV file.
    """
    # Ensure the output directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # The dictionary keys ('ranks', 'true_phis') become the columns
    df = pd.DataFrame(sbc_results)

    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"SBC results saved to {filepath}")
