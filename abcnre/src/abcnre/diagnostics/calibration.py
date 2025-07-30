import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..inference.estimator import NeuralRatioEstimator
from ..simulation.simulator import ABCSimulator

from .posterior import get_unnormalized_nre_pdf, get_normalized_pdf, sample_from_pdf, get_unormalized_corrected_nre_pdf


def run_abc_sbc(
    key: jax.random.PRNGKey,
    estimator: 'NeuralRatioEstimator',
    simulator: 'ABCSimulator',
    abc_phi_samples: np.ndarray,
    num_sbc_rounds: int = 500,
    num_posterior_samples: int = 1000,
    true_phis_for_sbc: Optional[np.ndarray] = None 

) -> Dict[str, Any]:
    """
    Performs ABC-based Simulation-Based Calibration (ABC-SBC).
    
    If `true_phis_for_sbc` is provided, it will be used as the set of
    ground truth parameters. Otherwise, `num_sbc_rounds` parameters will be
    randomly drawn from `abc_phi_samples`.
    
    Args:
        true_phis_for_sbc: An optional array of pre-defined "true" phi values to use for calibration.
    """
    if true_phis_for_sbc is not None:
        print(f"Running ABC-SBC using {len(true_phis_for_sbc)} provided true phi values...")
        phis_to_iterate = true_phis_for_sbc
    else:
        print(f"Running ABC-SBC by drawing {num_sbc_rounds} phi values from the ABC posterior...")
        key, choice_key = jax.random.split(key)
        phis_to_iterate = jax.random.choice(choice_key, abc_phi_samples, shape=(num_sbc_rounds,))

    ranks = []
    true_phis_used = []
    initial_bounds = (np.min(abc_phi_samples), np.max(abc_phi_samples))

    for phi_true in tqdm(phis_to_iterate, desc="ABC-SBC Progress"):
        key, sim_key, sample_key = jax.random.split(key, 3)

       
        n_obs = simulator.observed_data.shape[0]
        x_sim = simulator.model.simulate(sim_key, phi_true)
        
        temp_simulator = ABCSimulator(model=simulator.model, observed_data=x_sim)

        unnormalized_pdf_func = get_unormalized_corrected_nre_pdf(estimator, temp_simulator, 
                                                                  phi_samples=abc_phi_samples)
        grid, normalized_pdf = get_normalized_pdf(unnormalized_pdf_func, initial_bounds)

        # 3. Draw samples from this NRE posterior
        posterior_samples = sample_from_pdf(
            grid, normalized_pdf, num_posterior_samples, sample_key
        )

        # 4. Compute the rank of phi_true
        rank = jnp.sum(posterior_samples < phi_true)
        ranks.append(int(rank))
        true_phis_used.append(float(phi_true.item()))

    print("ABC-SBC complete.")
    return {
        'ranks': np.array(ranks),
        'true_phis': np.array(true_phis_used)
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
    print(f"✅ SBC results saved to {filepath}")