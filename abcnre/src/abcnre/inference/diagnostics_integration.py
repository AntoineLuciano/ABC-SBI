"""
Integration utilities for connecting inference module with diagnostics.

This module provides utilities to prepare and format inference results
for use with the diagnostics module, including simulation-based calibration
and posterior analysis.
"""

from typing import Dict, Any, List, Tuple, Optional, Callable
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from pathlib import Path
import pickle

from .estimator import NeuralRatioEstimator
from ..simulation.simulator import ABCSimulator


class InferenceResultsPackage:
    """
    Package containing all inference results for diagnostic analysis.
    
    This class encapsulates all the results from neural ratio estimation
    that are needed for downstream diagnostic analysis.
    """
    
    def __init__(
        self,
        estimator: NeuralRatioEstimator,
        abc_simulator: ABCSimulator,
        training_results: Dict[str, Any],
        validation_results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.estimator = estimator
        self.abc_simulator = abc_simulator
        self.training_results = training_results
        self.validation_results = validation_results
        self.metadata = metadata or {}
        
        # Extract key information
        self.model_info = estimator.get_model_info()
        self.abc_config = abc_simulator.get_model_info()
        
    def generate_posterior_samples(
        self,
        key: Any,
        num_samples: int = 10000,
        batch_size: Optional[int] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Generate posterior samples using the trained estimator.

        Args:
            key: RNG key
            num_samples: number of posterior samples to draw
            batch_size: size of minibatches for log‐ratio computation;
                        if None, process all samples in one batch

        Returns:
            A dict with:
            - theta_samples: all prior draws (shape [num_samples, ...])
            - posterior_samples: resampled posterior draws
            - log_ratios: log p(θ|x) − log p(θ)
            - weights: unnormalized importance weights
            - normalized_weights: weights normalized to sum to 1
            - effective_sample_size: ESS of the weighted sample
        """
        # If no batch size is given, process all samples at once
        if batch_size is None or batch_size <= 0:
            batch_size = num_samples

        # 1) Draw num_samples from the prior by splitting the key
        key, prior_key = random.split(key)
        prior_keys = random.split(prior_key, num_samples)
        theta_samples = jnp.stack(
            [self.abc_simulator.model.prior_sample(k) for k in prior_keys],
            axis=0
        )

        # 2) Compute log‐ratios and weights in minibatches
        all_log_ratios = []
        all_weights = []
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_theta = theta_samples[start:end]
            feats = self._create_features(batch_theta)
            log_ratios = self.estimator.log_ratio(feats)
            weights = jnp.exp(log_ratios)
            all_log_ratios.append(log_ratios)
            all_weights.append(weights)

        log_ratios = jnp.concatenate(all_log_ratios, axis=0)
        weights    = jnp.concatenate(all_weights,    axis=0)

        # 3) Normalize weights and resample
        normalized_weights = weights / jnp.sum(weights)
        key, resample_key = random.split(key)
        indices = random.choice(
            resample_key,
            num_samples,
            shape=(num_samples,),
            p=normalized_weights
        )
        posterior_samples = theta_samples[indices]

        return {
            'theta_samples': theta_samples,
            'posterior_samples': posterior_samples,
            'log_ratios': log_ratios,
            'weights': weights,
            'normalized_weights': normalized_weights,
            'effective_sample_size': self._compute_ess(normalized_weights)
        }


    def _create_features(self, theta_samples: jnp.ndarray) -> jnp.ndarray:
        """
        Create features for the neural network.
        
        This is a simplified version - in practice, this would depend
        on your specific model and feature engineering.
        """
        # For models with summary statistics
        if hasattr(self.abc_simulator.model, 'summary_stat_fn'):
            obs_summary = self.abc_simulator.model.summary_stat_fn(
                self.abc_simulator.observed_data
            )
            
            # If theta is multidimensional, flatten it
            if theta_samples.ndim > 1:
                theta_flat = theta_samples.reshape(theta_samples.shape[0], -1)
            else:
                theta_flat = theta_samples[:, None]
            
            # Repeat observed summary for each theta
            if obs_summary.ndim == 0:
                obs_summary_repeated = jnp.repeat(obs_summary, len(theta_samples))[:, None]
            else:
                obs_summary_repeated = jnp.repeat(
                    obs_summary[None, :], len(theta_samples), axis=0
                )
            
            features = jnp.concatenate([obs_summary_repeated, theta_flat], axis=1)
        else:
            # Fallback: use raw data mean
            obs_mean = jnp.mean(self.abc_simulator.observed_data)
            features = jnp.column_stack([
                jnp.repeat(obs_mean, len(theta_samples)),
                theta_samples
            ])
        
        return features
    
    def _compute_ess(self, weights: jnp.ndarray) -> float:
        """Compute effective sample size."""
        return float(1.0 / jnp.sum(weights ** 2))
    
    def prepare_for_sbc(
        self,
        key: Any,
        num_simulations: int,
        num_posterior_samples: int
    ) -> Dict[str, Any]:
        """
        Prepare data for Simulation-Based Calibration (SBC).

        Returns a dict with keys:
        'true_parameters', 'posterior_samples', 'ranks'
        """
        true_parameters = []
        posterior_samples = []
        ranks = []

        for _ in range(num_simulations):
            # Split key for prior draw and simulation
            key, param_key, sim_key = random.split(key, 3)

            # 1) Draw one true parameter from the prior
            theta_true = self.abc_simulator.model.prior_sample(param_key)
            true_parameters.append(theta_true)

            # 2) Simulate data under this true parameter,
            #    using the length of the observed_data array
            n_obs = self.abc_simulator.observed_data.shape[0]
            data = self.abc_simulator.model.simulate(
                sim_key,
                theta_true,
                n_obs
            )

            # 3) Create a temporary simulator with the new observed data
            temp_sim = ABCSimulator(
                model=self.abc_simulator.model,
                observed_data=data,
                epsilon=self.abc_simulator.epsilon
            )

            # 4) Generate posterior samples via this package
            post = self.generate_posterior_samples(
                key,
                num_samples=num_posterior_samples,
                batch_size=None
            )
            posterior_samples.append(post['theta_samples'])

            # 5) Compute the rank of the true parameter among posterior samples
            ranks.append(int(jnp.sum(post['theta_samples'] <= theta_true)))

        return {
            'true_parameters': true_parameters,
            'posterior_samples': posterior_samples,
            'ranks': ranks
        }


    
    def save_package(self, filepath: str) -> None:
        """
        Save the complete inference results package.
        
        Args:
            filepath: Path to save the package
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save estimator separately
        estimator_path = filepath.parent / f"{filepath.stem}_estimator.npz"
        self.estimator.save_model(str(estimator_path))
        
        # Save ABC simulator configuration
        abc_config_path = filepath.parent / f"{filepath.stem}_abc_config.yaml"
        self.abc_simulator.save_configuration(str(abc_config_path))
        
        # Create package data
        package_data = {
            'estimator_path': str(estimator_path),
            'abc_config_path': str(abc_config_path),
            'training_results': self.training_results,
            'validation_results': self.validation_results,
            'metadata': self.metadata,
            'model_info': self.model_info,
            'abc_config': self.abc_config
        }
        
        # Save package
        with open(filepath, 'wb') as f:
            pickle.dump(package_data, f)
        
        print(f"Inference results package saved to {filepath}")
    
    @classmethod
    def load_package(cls, filepath: str) -> 'InferenceResultsPackage':
        """
        Load inference results package from file.
        
        Args:
            filepath: Path to load the package from
            
        Returns:
            Loaded inference results package
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Package file not found: {filepath}")
        
        # Load package data
        with open(filepath, 'rb') as f:
            package_data = pickle.load(f)
        
        # Load estimator
        # Note: This would need to be implemented based on your estimator loading logic
        # For now, this is a placeholder
        estimator = None  # Would load from package_data['estimator_path']
        
        # Load ABC simulator
        # Note: This would need to be implemented based on your simulator loading logic
        abc_simulator = None  # Would load from package_data['abc_config_path']
        
        return cls(
            estimator=estimator,
            abc_simulator=abc_simulator,
            training_results=package_data['training_results'],
            validation_results=package_data['validation_results'],
            metadata=package_data['metadata']
        )


def create_sbc_experiment(
    estimator: NeuralRatioEstimator,
    abc_simulator: ABCSimulator,
    num_simulations: int = 1000,
    num_posterior_samples: int = 1000,
    save_results: bool = True,
    results_dir: str = './sbc_results'
) -> Dict[str, Any]:
    """
    Create a complete SBC experiment setup.
    
    Args:
        estimator: Trained neural ratio estimator
        abc_simulator: ABC simulator
        num_simulations: Number of SBC simulations
        num_posterior_samples: Number of posterior samples per simulation
        save_results: Whether to save results
        results_dir: Directory to save results
        
    Returns:
        SBC experiment results
    """
    # Create results package
    results_package = InferenceResultsPackage(
        estimator=estimator,
        abc_simulator=abc_simulator,
        training_results=estimator.training_history,
        metadata={
            'sbc_config': {
                'num_simulations': num_simulations,
                'num_posterior_samples': num_posterior_samples,
                'results_dir': results_dir
            }
        }
    )
    
    # Prepare SBC data
    key = random.PRNGKey(42)
    sbc_data = results_package.prepare_for_sbc(
        key, num_simulations, num_posterior_samples
    )
    
    # Save results if requested
    if save_results:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete package
        package_path = results_dir / 'inference_results_package.pkl'
        results_package.save_package(str(package_path))
        
        # Save SBC data separately
        sbc_path = results_dir / 'sbc_data.pkl'
        with open(sbc_path, 'wb') as f:
            pickle.dump(sbc_data, f)
        
        print(f"SBC experiment results saved to {results_dir}")
    
    return {
        'sbc_data': sbc_data,
        'results_package': results_package,
        'experiment_info': {
            'num_simulations': num_simulations,
            'num_posterior_samples': num_posterior_samples,
            'model_type': type(estimator.network).__name__,
            'abc_model': type(abc_simulator.model).__name__
        }
    }


def prepare_posterior_analysis(
    estimator: NeuralRatioEstimator,
    abc_simulator: ABCSimulator,
    num_samples: int = 10000,
    compare_with_abc: bool = True,
    abc_samples_count: int = 10000
) -> Dict[str, Any]:
    """
    Prepare comprehensive posterior analysis.
    
    Args:
        estimator: Trained neural ratio estimator
        abc_simulator: ABC simulator
        num_samples: Number of NRE posterior samples
        compare_with_abc: Whether to generate ABC samples for comparison
        abc_samples_count: Number of ABC samples for comparison
        
    Returns:
        Posterior analysis data
    """
    key = random.PRNGKey(42)
    
    # Create results package
    results_package = InferenceResultsPackage(
        estimator=estimator,
        abc_simulator=abc_simulator,
        training_results=estimator.training_history
    )
    
    # Generate NRE posterior samples
    key, nre_key = random.split(key)
    nre_results = results_package.generate_posterior_samples(nre_key, num_samples)
    
    analysis_data = {
        'nre_posterior': nre_results,
        'model_info': results_package.model_info,
        'abc_config': results_package.abc_config
    }
    
    # Generate ABC samples for comparison
    if compare_with_abc:
        key, abc_key = random.split(key)
        abc_samples = abc_simulator.generate_samples(abc_key, abc_samples_count)
        analysis_data['abc_posterior'] = {
            'samples': abc_samples.theta_samples,
            'distances': abc_samples.distances,
            'summary_stats': abc_samples.summary_stats
        }
    
    return analysis_data


def export_for_diagnostics(
    analysis_data: Dict[str, Any],
    export_path: str
) -> None:
    """
    Export analysis data in format suitable for diagnostics module.
    
    Args:
        analysis_data: Posterior analysis data
        export_path: Path to export data
    """
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create export data structure
    export_data = {
        'format_version': '1.0',
        'data_type': 'posterior_analysis',
        'nre_posterior_samples': np.array(analysis_data['nre_posterior']['posterior_samples']),
        'nre_log_ratios': np.array(analysis_data['nre_posterior']['log_ratios']),
        'nre_weights': np.array(analysis_data['nre_posterior']['weights']),
        'effective_sample_size': analysis_data['nre_posterior']['effective_sample_size'],
        'model_info': analysis_data['model_info'],
        'abc_config': analysis_data['abc_config']
    }
    
    # Add ABC data if available
    if 'abc_posterior' in analysis_data:
        export_data['abc_posterior_samples'] = np.array(analysis_data['abc_posterior']['samples'])
        export_data['abc_distances'] = np.array(analysis_data['abc_posterior']['distances'])
        if analysis_data['abc_posterior']['summary_stats'] is not None:
            export_data['abc_summary_stats'] = np.array(analysis_data['abc_posterior']['summary_stats'])
    
    # Save export data
    with open(export_path, 'wb') as f:
        pickle.dump(export_data, f)
    
    print(f"Data exported for diagnostics to {export_path}")


def create_diagnostic_summary(
    estimator: NeuralRatioEstimator,
    abc_simulator: ABCSimulator,
    validation_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create diagnostic summary for the inference results.
    
    Args:
        estimator: Trained neural ratio estimator
        abc_simulator: ABC simulator
        validation_results: Optional validation results
        
    Returns:
        Diagnostic summary
    """
    summary = {
        'model_summary': {
            'network_type': type(estimator.network).__name__,
            'network_config': estimator.network.get_config(),
            'is_trained': estimator.is_trained,
            'num_parameters': estimator.network.count_parameters(estimator.state.params) if estimator.is_trained else 0,
            'training_epochs': len(estimator.training_history['train_loss'])
        },
        'abc_summary': {
            'model_type': type(abc_simulator.model).__name__,
            'epsilon': abc_simulator.epsilon,
            'observed_data_shape': abc_simulator.observed_data.shape,
            'has_summary_stats': hasattr(abc_simulator.model, 'summary_stat_fn')
        },
        'performance_summary': {}
    }
    
    if estimator.is_trained:
        summary['performance_summary'] = {
            'final_train_loss': estimator.training_history['train_loss'][-1],
            'final_val_loss': estimator.training_history['val_loss'][-1],
            'final_train_accuracy': estimator.training_history['train_accuracy'][-1],
            'final_val_accuracy': estimator.training_history['val_accuracy'][-1]
        }
    
    if validation_results:
        summary['validation_summary'] = validation_results.get('validation_summary', {})
    
    return summary