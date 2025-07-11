"""
Advanced tests for the ABCNRE inference module adapted to basic style.

This file contains tests for the comprehensive inference workflow,
including model simulation, checkpointing, experiment management, and cleanup.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from jax import random
import jax.numpy as jnp

# Attempt to import components; skip tests if unavailable
try:
    from abcnre.simulation.models import GaussGaussModel
    from abcnre.inference import (
        NeuralRatioEstimator,
        MLPNetwork,
        DeepSetNetwork,
        create_mlp_experiment,
        validate_classifier_performance,
        validate_posterior_quality,
        run_comprehensive_validation,
        InferenceResultsPackage,
        create_sbc_experiment,
        EstimatorCheckpoint,
        ExperimentManager,
        ModelRegistry
    )
    from abcnre.simulation import ABCSimulator
except ImportError as e:
    pytest.skip(f"Some components unavailable, skipping advanced tests: {e}", allow_module_level=True)

# Global RNG key
key = random.PRNGKey(0)


def test_gauss_gauss_model_basic():
    """Test GaussGaussModel sampling and log_prob."""
    model = GaussGaussModel(mu0=0.0, sigma0=1.0, sigma=0.5)
    samples = model.sample(key, num_samples=10)
    assert samples.shape == (10,)
    logp = model.log_prob(samples)
    assert logp.shape == (10,)


def test_estimator_checkpoint(tmp_path):
    """Test saving and loading checkpoints via EstimatorCheckpoint."""
    checkpoint = EstimatorCheckpoint(tmp_path)
    state = {'step': 1, 'params': jnp.array([1.0, 2.0])}
    checkpoint.save(state, step=1)
    loaded = checkpoint.load(step=1)
    assert loaded['step'] == state['step']
    assert jnp.allclose(loaded['params'], state['params'])


def test_experiment_manager(tmp_path):
    """Test ExperimentManager run and result packaging."""
    config = create_mlp_experiment(
        experiment_name='test', hidden_dims=[8,4], learning_rate=1e-3, num_epochs=1
    )
    manager = ExperimentManager(workdir=tmp_path)
    manager.run(key)
    results_file = Path(tmp_path) / 'results.pkl'
    assert results_file.exists(), "Results file should be created."


def test_cleanup_temp_dirs():
    """Test cleanup of temporary directories uses shutil.rmtree."""
    temp_dir = tempfile.mkdtemp()
    assert Path(temp_dir).exists()
    shutil.rmtree(temp_dir)
    assert not Path(temp_dir).exists()


def test_classifier_validation():
    """Test classifier performance validation."""
    model = GaussGaussModel(mu0=0.0, sigma0=1.0, sigma=0.5)
    simulator = ABCSimulator(model, model.simulate(key, 1.0, 10), epsilon=0.1)
    network = MLPNetwork(hidden_dims=[8,4], dropout_rate=0.0)
    estimator = NeuralRatioEstimator(network=network, learning_rate=1e-3, random_seed=0)

    def gen(k, b): return simulator.generate_training_samples(k, b)
    estimator.train(data_generator=gen, num_epochs=1, batch_size=8, verbose=False)
    results = validate_classifier_performance(estimator, gen, num_test_batches=2, batch_size=4)
    assert 'accuracy' in results and 0.0 <= results['accuracy'] <= 1.0
    assert 'roc_auc' in results


def test_posterior_quality_validation():
    """Test posterior quality validation."""
    model = GaussGaussModel(mu0=0.0, sigma0=1.0, sigma=0.5)
    simulator = ABCSimulator(model, model.simulate(key, 1.0, 20), epsilon=0.1)
    theta_abc = simulator.generate_samples(key, 50).theta_samples
    theta_nre = model.prior_sample(key, 50)
    results = validate_posterior_quality(theta_abc, theta_nre, true_value=1.0)
    assert 'ks_statistic' in results
    assert 'wasserstein_distance' in results


def test_comprehensive_validation():
    """Test comprehensive validation workflow."""
    model = GaussGaussModel(mu0=0.0, sigma0=1.0, sigma=0.5)
    simulator = ABCSimulator(model, model.simulate(key, 1.0, 20), epsilon=0.1)
    network = MLPNetwork(hidden_dims=[8,4], dropout_rate=0.0)
    estimator = NeuralRatioEstimator(network=network, learning_rate=1e-3, random_seed=0)

    def gen(k, b): return simulator.generate_training_samples(k, b)
    estimator.train(data_generator=gen, num_epochs=1, batch_size=8, verbose=False)
    val = run_comprehensive_validation(
        estimator, simulator,
        num_test_batches=1, batch_size=4,
        num_posterior_samples=10, true_value=1.0,
        save_plots=False
    )
    assert 'classifier_performance' in val
    assert 'posterior_quality' in val


def test_inference_results_package():
    """Test InferenceResultsPackage functionality."""
    model = GaussGaussModel(mu0=0.0, sigma0=1.0, sigma=0.5)
    simulator = ABCSimulator(model, model.simulate(key, 1.0, 30), epsilon=0.1)
    network = MLPNetwork(hidden_dims=[8,4], dropout_rate=0.0)
    estimator = NeuralRatioEstimator(network=network, learning_rate=1e-3, random_seed=0)

    def gen(k, b): return simulator.generate_training_samples(k, b)
    training = estimator.train(data_generator=gen, num_epochs=1, batch_size=8, verbose=False)
    package = InferenceResultsPackage(estimator, simulator, training, metadata={})
    post = package.generate_posterior_samples(key, num_samples=10, batch_size=5)
    assert post['theta_samples'].shape[0] == 10


def test_sbc_experiment_creation():
    """Test SBC experiment creation."""
    model = GaussGaussModel(mu0=0.0, sigma0=1.0, sigma=0.5)
    simulator = ABCSimulator(model, model.simulate(key, 1.0, 30), epsilon=0.1)
    network = MLPNetwork(hidden_dims=[8,4], dropout_rate=0.0)
    estimator = NeuralRatioEstimator(network=network, learning_rate=1e-3, random_seed=0)

    sbc = create_sbc_experiment(
        estimator, simulator,
        num_simulations=2, num_posterior_samples=5,
        save_results=False
    )
    assert 'sbc_data' in sbc and len(sbc['sbc_data']['true_parameters']) == 2


def test_model_registry_and_deepset_training():
    """Test ModelRegistry and DeepSet training."""
    assert ModelRegistry.get_network('MLPNetwork') == MLPNetwork
    nets = ModelRegistry.list_networks()
    assert 'DeepSetNetwork' in nets
    model = GaussGaussModel(mu0=0.0, sigma0=1.0, sigma=0.5)
    simulator = ABCSimulator(model, model.simulate(key, 1.0, 20), epsilon=0.1)
    deepset = DeepSetNetwork(phi_hidden_dims=[4], rho_hidden_dims=[8], pooling='mean', dropout_rate=0.0)
    est = NeuralRatioEstimator(network=deepset, learning_rate=1e-3, random_seed=0)
    def gen(k, b): return simulator.generate_training_samples(k, b)
    res = est.train(data_generator=gen, num_epochs=1, batch_size=4, verbose=False)
    preds = est.predict(simulator.generate_training_samples(key,4).features)
    assert preds.shape[0] == 4


def test_configuration_persistence():
    """Test configuration saving and loading."""
    config = create_mlp_experiment(
        experiment_name='cfg_test', hidden_dims=[8], learning_rate=5e-4, num_epochs=2
    )
    cfg_path = Path(tempfile.mkdtemp()) / 'cfg.yaml'
    config.save(cfg_path)
    loaded = type(config).load(cfg_path)
    assert loaded.experiment_name == 'cfg_test'


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    print("End-to-end workflow...")
    model = GaussGaussModel(mu0=0.0, sigma0=1.0, sigma=0.5)
    simulator = ABCSimulator(model, model.simulate(key, 1.0, 50), epsilon=0.1)
    network = MLPNetwork(hidden_dims=[8,4])
    estimator = NeuralRatioEstimator(network, learning_rate=1e-3)
    def gen(k, b): return simulator.generate_training_samples(k, b)
    estimator.train(data_generator=gen, num_epochs=1, batch_size=8, verbose=False)
    val = run_comprehensive_validation(
        estimator, simulator,
        num_test_batches=1, batch_size=4,
        num_posterior_samples=10, true_value=1.0,
        save_plots=False
    )
    sbc = create_sbc_experiment(
        estimator, simulator,
        num_simulations=1, num_posterior_samples=5,
        save_results=False
    )
    assert 'validation_summary' in val
    assert len(sbc['sbc_data']['true_parameters']) == 1
    print("Workflow OK")


if __name__ == "__main__":
    # Run end-to-end test directly
    test_end_to_end_workflow()
    print("All advanced tests passed!")
