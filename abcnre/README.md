# ABC-SBI: Approximate Bayesian Computation with Neural Ratio Estimation

A Python package for approximate Bayesian inference using neural networks and JAX/Flax.

## Project Overview

This package implements Approximate Bayesian Computation (ABC) with Neural Ratio Estimation (NRE) for efficient Bayesian inference in high-dimensional parameter spaces. The main innovation is using neural networks to learn likelihood-to-evidence ratios and automatically discover optimal summary statistics.

## Code Architecture

The codebase is organized into three main sections:

### 1. Simulation (`src/abcnre/simulation/`)

Defines the core simulation framework:

- **Model classes**: Define statistical models with prior distributions and likelihood functions
- **ABCSimulator class**: Handles ABC simulations and can optionally learn optimal summary statistics s(x) to approximate the parameter of interest phi
- **Currently supports scalar summary statistics** for approximating phi

Key components:
- `models.py`: Statistical model definitions
- `simulator.py`: Main ABCSimulator class
- `utils.py`: Simulation utilities

### 2. Inference (`src/abcnre/inference/`)

Implements neural ratio estimation:

- **NeuralRatioEstimator class**: Takes an ABCSimulator as input and learns the classification task
- **Learns likelihood-to-evidence ratios**: Estimates p(x|theta)/p(x) to approximate the posterior
- **Classification approach**: Transforms the ratio estimation into a binary classification problem

Key components:
- `estimator.py`: Main NeuralRatioEstimator class
- `validation.py`: Model validation utilities
- `io.py`: Save/load functionality

### 3. Training (`src/abcnre/training/`)

Unified training framework used by both simulation and inference sections:

- **Summary learner**: Trains networks to learn optimal summary statistics s(x)
- **Classifier**: Trains networks for likelihood ratio estimation and posterior approximation
- **NNConfig class**: Unified configuration system containing:
  - Network architecture parameters
  - Training hyperparameters  
  - Learning rate schedules
  - Stopping rules

Key components:
- `config.py`: NNConfig unified configuration system
- `optimization.py`: Loss functions and optimizers
- `networks/`: Neural network architectures (MLP, DeepSet, etc.)
- `components/`: Training loop, schedulers, metrics

## Configuration System

All configuration files (both model configs and NNConfig) are stored in:
```
abcnre/examples/configs/
├── models/           # Statistical model configurations
├── networks/         # Network architecture configs  
├── training/         # Training parameter configs
├── lr_schedulers/    # Learning rate schedule configs
└── stopping_rules/   # Early stopping configurations
```

## Working Examples

Currently three fully functional examples:

1. **gauss_1D**: 1-dimensional Gaussian model
2. **gauss_2D**: 2-dimensional Gaussian model  
3. **gauss_100D**: 100-dimensional Gaussian model (demonstrates scalability)

Each example includes:
- Model configuration files
- Training notebooks with full pipeline
- Results and diagnostics

Example locations:
```
abcnre/examples/
├── gauss_1D/
│   ├── notebooks/    # Training and evaluation notebooks
│   └── results/      # Saved models and outputs
├── gauss_2D/
└── gauss_100D/
```

## Command Line Interface (CLI)

Created a functional CLI system for running experiments:
```bash
python -m cli.main --help
```

Current CLI features:
- Model configuration and creation
- Interactive workflow (partial implementation)

**Note**: CLI is functional but not yet fully operational. See TODO.txt for remaining work.

## Installation and Setup

```bash
# Clone repository
git clone https://github.com/AntoineLuciano/ABC-SBI.git
cd ABC-SBI/abcnre

# Install package
pip install -e .

# Install dependencies
pip install jax jaxlib flax optax numpy matplotlib
```

## Quick Start Example

```python
from abcnre.simulation import ABCSimulator
from abcnre.simulation.models import get_example_model_configs, create_model_from_dict
from abcnre.inference import NeuralRatioEstimator
from abcnre.training import get_nn_config
import jax

# 1. Load and create a model
model_config = get_example_model_configs("gauss_gauss_2d_default")
model = create_model_from_dict(model_config)

# 2. Create simulator
simulator = ABCSimulator(model=model)

# 3. Set up observed data and epsilon
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
true_theta = model.get_prior_sample(subkey)
key, subkey = jax.random.split(key)
x_obs = model.simulate_data(subkey, true_theta)
simulator.update_observed_data(x_obs)

# 4. Configure neural network
nn_config = get_nn_config(
    network_name="mlp",
    training_size="default",
    task_type="classifier"
)

# 5. Create and train estimator
estimator = NeuralRatioEstimator(nn_config=nn_config, simulator=simulator)
key, subkey = jax.random.split(key)
results = estimator.train(subkey)
```

## Current Status and TODO

Please check `TODO.txt`


## Key Features

- **JAX/Flax backend**: High-performance automatic differentiation
- **Modular design**: Clean separation between simulation, inference, and training
- **Unified configuration**: Single NNConfig system for all neural network training
- **Scalable**: Successfully tested on 100-dimensional problems
- **Extensible**: Easy to add new models, networks, and training strategies


## File Structure Overview

```
abcnre/
├── src/abcnre/
│   ├── simulation/          # Model definitions and ABC simulation
│   ├── inference/           # Neural ratio estimation
│   ├── training/            # Unified NN training framework
│   ├── diagnostics/         # Validation and visualization
│   └── utils/              # Utilities and CLI tools
├── examples/
│   ├── configs/            # All configuration files
│   ├── gauss_1D/          # 1D Gaussian example
│   ├── gauss_2D/          # 2D Gaussian example  
│   └── gauss_100D/        # 100D Gaussian example
├── cli/                   # Command line interface
└── tests/                 # Test suite
```
