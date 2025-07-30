# **Approximate Bayesian Computation with Neural Ratio Estimation**

A modern Python package for approximate Bayesian inference using neural networks, with JAX/Flax support for high-performance computing.

## 🚀 Overview

ABCNRE provides a comprehensive and modular approach to ABC with neural ratio estimation, organized into four main modules:

- **🔧 Preprocessing** - Direct parameter prediction with adaptive architectures
- **🎲 Simulation** - Simulation tools and data management
- **🧠 Inference** - Neural ratio estimation and advanced ABC methods  
- **📊 Diagnostics** - Validation, calibration and results visualization

## ⚡ Installation



```bash
# Installation depuis le répertoire source
git clone https://github.com/AntoineLuciano/ABC-SBI.git
cd ABC-SBI/abcnre
pip install -e .

# Dépendances requises
pip install jax jaxlib flax optax numpy matplotlib
```



## 🎯 Quick Start

### Direct Parameter Prediction

```python
from abcnre.preprocessing import DirectPreprocessingConfig, DirectAdaptivePreprocessor
import jax
import jax.numpy as jnp

# Example data
data = jax.random.normal(jax.random.PRNGKey(42), (1000, 20, 3))  # i.i.d.
params = jax.random.normal(jax.random.PRNGKey(123), (1000, 2))

# Automatic configuration
config = DirectPreprocessingConfig.auto_detect(data, params)
preprocessor = DirectAdaptivePreprocessor(config)
preprocessor.initialize(jax.random.PRNGKey(42))

# Direct training x → φ
preprocessor.train(data, params, n_steps=1000)

# Prediction
predicted_params = preprocessor.predict_parameters(new_data)
```

### Complete ABC Workflow

```python
from abcnre.simulation import Simulator
from abcnre.inference import NRE, ABC
from abcnre.diagnostics import calibration_check, posterior_plots

# 1. Simulator configuration
simulator = Simulator(model_func=your_model, prior=your_prior)

# 2. NRE training
nre = NRE(architecture="deep_set")
nre.train(simulator, n_simulations=10000)

# 3. ABC inference
abc = ABC(nre=nre, simulator=simulator)
posterior = abc.sample(observed_data, n_samples=5000)

# 4. Diagnostics
calibration_check(posterior, true_params)
posterior_plots(posterior, observed_data)
```

## 📦 Package Architecture

### 🔧 Preprocessing (`abcnre.preprocessing`)

**Direct parameter prediction with automatic architecture selection**

```python
# Direct approach x → φ_pred (no more summary statistics!)
from abcnre.preprocessing import DirectAdaptivePreprocessor

# Auto-detection: DirectMLP for correlated data, DirectDeepSet for i.i.d.
config = DirectPreprocessingConfig.auto_detect(your_data, your_params)
preprocessor = DirectAdaptivePreprocessor(config)

# Training and prediction
preprocessor.train(data, params, n_steps=1000)
predictions = preprocessor.predict_parameters(new_data)
```

**Features:**
- ✅ Direct prediction `L = ||φ - φ_pred(x)||²`
- ✅ Automatic MLP/DeepSet selection
- ✅ JAX/Flax compatible with GPU optimizations
- ✅ Invariance testing for DeepSet validation
- ✅ Production export

### 🎲 Simulation (`abcnre.simulation`)

**Robust simulation tools and data management**

```python
from abcnre.simulation import Simulator, Prior, DataManager

# Simulator configuration
prior = Prior.uniform(low=[-2, -1], high=[2, 1])
simulator = Simulator(
    model=your_generative_model,
    prior=prior,
    batch_size=1000
)

# Batch simulation
simulated_data, params = simulator.simulate(n_samples=10000)

# Data management
data_manager = DataManager()
data_manager.save_simulation(simulated_data, params, "simulation_001")
```

**Features:**
- ✅ High-performance vectorized simulation
- ✅ Flexible priors (uniform, gaussian, custom)
- ✅ Memory management for large simulations
- ✅ Automatic caching and persistence
- ✅ Multiple format support (HDF5, pickle, numpy)

### 🧠 Inference (`abcnre.inference`)

**Neural ratio estimation and advanced ABC methods**

```python
from abcnre.inference import NeuralRatioEstimator, ABCSampler

# NRE configuration
nre = NeuralRatioEstimator(
    architecture="deep_set",  # or "mlp", "transformer"
    hidden_dims=[128, 64],
    learning_rate=1e-3
)

# Training
nre.train(
    simulator=simulator,
    n_simulations=50000,
    validation_split=0.2
)

# ABC sampling
sampler = ABCSampler(nre=nre)
posterior = sampler.sample(
    observed_data=obs_data,
    n_samples=10000,
    method="mcmc"  # or "smc", "rejection"
)
```

**Features:**
- ✅ Multiple NRE architectures (MLP, DeepSet, Transformer)
- ✅ ABC methods: MCMC, SMC, Rejection
- ✅ Sequential and adaptive training
- ✅ Integrated cross-validation
- ✅ Epistemic uncertainty management

### 📊 Diagnostics (`abcnre.diagnostics`)

**Complete validation, calibration and visualization**

```python
from abcnre.diagnostics import (
    calibration_check, 
    coverage_test,
    posterior_plots,
    sbc_analysis
)

# Calibration tests
calibration_results = calibration_check(
    posterior_samples=posterior,
    true_parameters=true_params,
    alpha_levels=[0.1, 0.05, 0.01]
)

# Simulation-Based Calibration
sbc_results = sbc_analysis(
    nre_model=nre,
    simulator=simulator,
    n_sbc_runs=1000
)

# Visualizations
posterior_plots(
    posterior=posterior,
    observed_data=obs_data,
    true_params=true_params,
    save_path="diagnostics.png"
)
```

**Features:**
- ✅ Coverage and calibration tests
- ✅ Simulation-Based Calibration (SBC)
- ✅ Performance metrics (MSE, coverage, rank statistics)
- ✅ Interactive visualizations
- ✅ Automatic HTML/PDF reports

## 🔬 Complete Examples

### Example 1: Direct Parameter Prediction

```python
# Generate example data
import jax.numpy as jnp
import jax

# i.i.d. data (independent observations)
n_samples, n_obs, obs_dim = 1000, 20, 3
data = jax.random.normal(jax.random.PRNGKey(42), (n_samples, n_obs, obs_dim))
params = jax.random.normal(jax.random.PRNGKey(123), (n_samples, 2))

from abcnre.preprocessing import DirectPreprocessingConfig, DirectAdaptivePreprocessor

# Automatic configuration (detects DirectDeepSet for i.i.d. data)
config = DirectPreprocessingConfig.auto_detect(data, params)
print(f"Detected architecture: {'DirectDeepSet' if config.sample_is_iid else 'DirectMLP'}")

# Training
preprocessor = DirectAdaptivePreprocessor(config)
preprocessor.initialize(jax.random.PRNGKey(42))
preprocessor.train(data, params, n_steps=1000)

# Invariance test (for DirectDeepSet)
if config.sample_is_iid:
    invariance_test = preprocessor.test_invariance(data[:5], jax.random.PRNGKey(999))
    print(f"Invariance respected: {invariance_test['is_invariant']}")

# Prediction on new data
new_data = jax.random.normal(jax.random.PRNGKey(456), (100, n_obs, obs_dim))
predictions = preprocessor.predict_parameters(new_data)
print(f"Predictions shape: {predictions.shape}")
```

### Example 2: Bayesian ABC Workflow

```python
from abcnre.simulation import Simulator, GaussianPrior
from abcnre.inference import NeuralRatioEstimator, MCMCSampler
from abcnre.diagnostics import posterior_summary, coverage_test

# 1. Define generative model
def gaussian_model(params):
    mu, sigma = params[0], jnp.exp(params[1])  # sigma > 0
    return jax.random.normal(jax.random.PRNGKey(0), (20,)) * sigma + mu

# 2. Prior and simulator  
prior = GaussianPrior(mean=[0, 0], cov=[[1, 0], [0, 1]])
simulator = Simulator(model=gaussian_model, prior=prior)

# 3. NRE training
nre = NeuralRatioEstimator(
    architecture="deep_set",
    hidden_dims=[128, 64, 32],
    learning_rate=1e-3
)

training_data = simulator.simulate(n_samples=10000)
nre.train(training_data, validation_split=0.2, epochs=100)

# 4. Observation and inference
observed_data = jnp.array([1.2, 1.1, 0.9, ...])  # your observed data
sampler = MCMCSampler(nre=nre, prior=prior)
posterior = sampler.sample(observed_data, n_samples=5000)

# 5. Diagnostics
summary = posterior_summary(posterior)
print(f"Posterior mean: {summary['mean']}")
print(f"95% CI: {summary['ci_95']}")

# Coverage test
coverage = coverage_test(nre, simulator, alpha_levels=[0.05, 0.1])
print(f"Coverage 95%: {coverage['0.05']:.3f}")
```

### Example 3: Benchmarking and Optimization

```python
from abcnre.preprocessing import benchmark_direct_architectures

# Test different architectures
data, params = generate_your_data()  # your data

results = benchmark_direct_architectures(
    sample_data=data,
    param_data=params,
    test_hidden_dims=[
        [64],           # Simple
        [128, 64],      # Standard  
        [256, 128, 64], # Complex
        [512, 256, 128] # Very complex
    ],
    n_steps=200
)

# Display results
for arch, metrics in results.items():
    print(f"{arch}: MSE={metrics['final_mse']:.4f}, Time={metrics['training_time']:.1f}s")

# Automatic selection of best architecture
best_arch = min(results.items(), key=lambda x: x[1]['final_mse'])
print(f"Best architecture: {best_arch[0]}")
```

## 🛠️ Installation and Configuration

### Prerequisites

```bash
# Python 3.8+
python --version

# JAX installation (CPU)
pip install jax jaxlib

# JAX installation (GPU - optional)
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

### Package Installation

```bash
# From GitHub
git clone https://github.com/AntoineLuciano/ABC-SBI.git
cd ABC-SBI/abcnre

# Development installation
pip install -e .

# Verification
python -c "import abcnre; print('✅ Installation successful!')"
```

### Advanced Configuration

```python
# JAX configuration for ARM Mac
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Force CPU on ARM Mac

# Memory configuration
import jax
jax.config.update('jax_enable_x64', True)  # Double precision
```

## 🧪 Tests and Validation

```bash
# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# Performance tests
pytest tests/benchmarks/

# Code coverage
pytest --cov=abcnre --cov-report=html tests/
```

## 📁 Project Structure

```
abcnre/
├── src/abcnre/              # Main source code
│   ├── preprocessing/       # 🔧 Direct parameter prediction
│   │   ├── adaptive_networks.py      # DirectMLP/DirectDeepSet architectures
│   │   ├── adaptive_factory_direct.py # High-level interface
│   │   └── examples.py               # Usage examples
│   ├── simulation/          # 🎲 Simulation and data management
│   │   ├── simulators.py             # Simulation classes
│   │   ├── priors.py                 # Prior distributions
│   │   └── data_management.py        # Storage and cache
│   ├── inference/           # 🧠 NRE and ABC methods
│   │   ├── nre.py                    # Neural Ratio Estimation
│   │   ├── abc_samplers.py           # MCMC, SMC, Rejection
│   │   └── training.py               # Training and validation
│   └── diagnostics/         # 📊 Validation and visualization
│       ├── calibration.py            # Calibration tests
│       ├── metrics.py                # Performance metrics
│       ├── posterior.py              # Posterior analysis
│       └── viz.py                    # Visualizations
├── tests/                   # Automated tests
│   ├── unit/               # Unit tests per module
│   ├── integration/        # Integration tests
│   └── benchmarks/         # Performance tests
├── examples/               # Examples and tutorials
│   ├── notebooks/          # Jupyter notebooks
│   ├── scripts/            # Standalone scripts
│   └── data/              # Example data
├── docs/                   # Documentation
│   ├── source/            # Sphinx sources
│   ├── tutorials/         # Advanced tutorials
│   └── api/              # API reference
└── scripts/               # Development scripts
    ├── install_dev.sh     # Development setup
    ├── run_tests.sh       # Automated testing
```

## 🚀 Advanced Features

### Performance and Optimization

- **JAX/Flax**: JIT compilation and automatic vectorization
- **GPU Support**: Hardware acceleration for intensive training
- **Batch Processing**: Optimized simulation and inference by batches
- **Memory Management**: Intelligent memory handling for large datasets
- **Caching**: Automatic caching of expensive simulations

### Neural Architectures

- **DirectMLP**: Dense networks for correlated data (time series, images)
- **DirectDeepSet**: Invariant networks for i.i.d. data (observation sets)
- **Transformer**: Attention for complex sequential data
- **Custom Architectures**: Flexible API for custom architectures

### Inference Methods

- **MCMC Samplers**: Hamiltonian, Metropolis, NUTS
- **SMC**: Sequential Monte Carlo with adaptation
- **Rejection Sampling**: Exact methods for small problems
- **Variational Inference**: Fast approximations for exploration

### Comprehensive Diagnostics

- **Simulation-Based Calibration**: Rigorous model validation
- **Coverage Tests**: Confidence interval verification
- **Posterior Predictive Checks**: Predictive validation
- **Convergence Diagnostics**: R-hat, ESS, trace plots

## 📚 Documentation and Resources

### Useful Links

- 📖 **Complete documentation**: [https://abcnre.readthedocs.io](https://abcnre.readthedocs.io)
- 🎓 **Tutorials**: `examples/notebooks/`
- 🔬 **Reference paper**: [arXiv:2024.xxxxx](https://arxiv.org)
- 💬 **Discussions**: GitHub Discussions
- 🐛 **Issues**: GitHub Issues

### Recommended Tutorials

1. **Getting Started**: `examples/notebooks/01_introduction.ipynb`
2. **Direct Parameter Prediction**: `examples/notebooks/02_preprocessing.ipynb`
3. **Neural Ratio Estimation**: `examples/notebooks/03_inference.ipynb`
4. **Advanced Diagnostics**: `examples/notebooks/04_diagnostics.ipynb`
5. **Custom Models**: `examples/notebooks/05_custom_models.ipynb`



### Development Architecture

```bash
# Continuous testing
pytest-watch tests/

# Benchmarks
python scripts/benchmark.py --module preprocessing --iterations 100

# Documentation
cd docs/
make html
```

## 📄 License and Support

### License

MIT License - see [LICENSE](LICENSE) for complete details.

### Support and Community

- 💡 **Questions**: GitHub Discussions
- 🐛 **Bugs**: GitHub Issues  
- 📧 **Contact**: antoine.luciano@research.gouv.fr
- 🐦 **News**: [@ABCNREPython](https://twitter.com/ABCNREPython)

### Changelog

**v1.0.0** (2024-07-22)
- ✅ Preprocessing module with direct prediction
- ✅ DirectMLP and DirectDeepSet architectures
- ✅ Complete JAX/Flax support
- ✅ Automatic invariance testing
- ✅ Production export

**v0.9.0** (2024-07-15)
- ✅ Complete modular architecture
- ✅ Simulation/inference/diagnostics integration
- ✅ Documentation and examples

---

**⭐ Feel free to star the project if you find it useful!**

[![Stars](https://img.shields.io/github/stars/AntoineLuciano/ABC-SBI?style=social)](https://github.com/AntoineLuciano/ABC-SBI)
[![Forks](https://img.shields.io/github/forks/AntoineLuciano/ABC-SBI?style=social)](https://github.com/AntoineLuciano/ABC-SBI)
[![Issues](https://img.shields.io/github/issues/AntoineLuciano/ABC-SBI)](https://github.com/AntoineLuciano/ABC-SBI/issues)
[![License](https://img.shields.io/github/license/AntoineLuciano/ABC-SBI)](https://github.com/AntoineLuciano/ABC-SBI/blob/main/LICENSE)
