## Overview of the `abcnre.inference` Package

The `abcnre.inference` subpackage implements Approximate Bayesian Computation with Neural Ratio Estimation (ABC-NRE). It is organized into several modules, each responsible for one part of the inference pipeline:

abcnre/inference/
├── base.py
├── config.py
├── mlp.py
├── deepset.py
├── utils.py
├── trainer.py
├── estimator.py
├── persistence.py
├── validation.py
└── diagnostics_integration.py


- **base.py**  
  Defines shared type aliases and low-level abstractions (e.g. RNG handling, array shapes).

- **config.py**  
  Houses experiment configurations (network architecture, optimizer settings, training hyperparameters) via dataclasses with `save()` / `load()` methods.

- **mlp.py**  
  Implements `MLPNetwork`, a Flax-based multilayer perceptron with configurable hidden layers, dropout, batch-norm, and activation.

- **deepset.py**  
  Implements `DeepSetNetwork`, a permutation-invariant network using separate φ and ρ subnetworks.

- **utils.py**  
  Utility functions for data preprocessing (e.g. feature concatenation), RNG splitting, and generic helpers.

- **trainer.py**  
  Orchestrates the training loop: batching, loss computation, optimizer steps, early stopping.

- **estimator.py**  
  Contains `NeuralRatioEstimator`, which wraps the Flax model, training state, `train()`, `predict()`, and `posterior_weights()` methods.

- **persistence.py**  
  Checkpoint saving/loading logic via `EstimatorCheckpoint`, and experiment metadata serialization via `ExperimentManager`.

- **validation.py**  
  Implements validation workflows:  
  - `validate_classifier_performance(...)`  
  - `validate_posterior_quality(...)`  
  - `run_comprehensive_validation(...)`

- **diagnostics_integration.py**  
  High-level wrappers (`InferenceResultsPackage`, `create_sbc_experiment`) combining simulation, inference, SBC prep, and packaging of results.

---

## Example Usage

Below is an example script demonstrating the inference workflow using the `abcnre.inference` package:

```python
import jax
import jax.numpy as jnp
from jax import random

from abcnre.simulation import ABCSimulator
from abcnre.simulation.models import GaussGaussModel
from abcnre.inference import (
    MLPNetwork,
    NeuralRatioEstimator,
    run_comprehensive_validation
)

# 1. Initialize random key and model
key = random.PRNGKey(0)
model = GaussGaussModel(mu0=0.0, sigma0=1.0, sigma=0.5)

# 2. Simulate observed data
key, obs_key = random.split(key)
observed = model.simulate(obs_key, theta=1.0, n_obs=100)

# 3. Set up ABC simulator
sim = ABCSimulator(model=model, observed_data=observed, epsilon=0.1)

# 4. Define network and estimator
network = MLPNetwork(hidden_dims=[64, 32], dropout_rate=0.1)
estimator = NeuralRatioEstimator(network=network, learning_rate=1e-3, random_seed=42)

# 5. Train briefly
def gen_data(k, bs): 
    return sim.generate_training_samples(k, bs)

estimator.train(
    data_generator=gen_data,
    num_epochs=10,
    batch_size=128
)

# 6. Validate performance
results = run_comprehensive_validation(
    estimator,
    sim,
    num_test_batches=5,
    batch_size=64,
    num_posterior_samples=500,
    true_value=1.0,
    save_plots=False
)

print(results['validation_summary'])
