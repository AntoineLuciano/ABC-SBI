# ABCNRE Module File Documentation

This document provides a comprehensive overview of the contents and purpose of each file in the ABCNRE package.

## 📁 `src/abcnre/simulation/`

### `__init__.py`
**Purpose**: Package initialization and public API definition for the simulation module

**Exports**:
- `ABCSimulator` - Main ABC simulation class
- `RejectionSampler`, `BaseSampler` - Sampling strategy classes
- `ABCSampleResult`, `ABCTrainingResult`, `ABCSingleResult` - Result data structures
- `save_generator_config`, `load_generator_config`, `import_class_from_string` - Utility functions

**Key Features**:
- Clean public API for ABC simulation
- Result structures for different sampling operations
- Configuration management utilities

---

### `base.py`
**Purpose**: Base classes and result structures for ABC simulation framework

**Classes**:
- `ABCSampleResult(NamedTuple)` - Container for multiple ABC samples
  - `sim_data`: Simulated data arrays
  - `theta_samples`: Parameter samples
  - `distances`: ABC distances
  - `summary_stats`: Optional summary statistics
  - `key`: JAX random key
  
- `ABCTrainingResult(NamedTuple)` - Container for neural ratio estimation training data
  - `features`: Input features for NRE training
  - `labels`: Binary labels (0=joint, 1=marginal)
  - `distances`: ABC distances
  - `summary_stats`: Optional summary statistics
  - `key`: JAX random key
  
- `ABCSingleResult(NamedTuple)` - Container for single ABC sample
  - `sim_data`: Single simulated dataset
  - `theta`: Single parameter sample
  - `distance`: ABC distance
  - `summary_stat`: Optional summary statistic

- `BaseSampler(ABC)` - Abstract base class for ABC samplers
  - `sample()` - Generate multiple samples
  - `sample_single()` - Generate single sample

**Key Features**:
- Type-safe result structures using NamedTuple
- Abstract interface for different sampling strategies
- JAX-compatible data structures

---

### `sampler.py`
**Purpose**: Implementation of ABC rejection sampling with JIT compilation

**Classes**:
- `RejectionSampler(BaseSampler)` - Main ABC rejection sampling implementation
  - `__init__()` - Initialize with model functions and ABC parameters
  - `sample_single()` - Generate single ABC sample using rejection
  - `sample()` - Generate multiple samples using vectorization
  - `generate_training_samples()` - Create training data for NRE
  - `get_epsilon_quantile()` - Compute epsilon from distance quantiles
  - `update_epsilon()` - Update tolerance threshold
  - `update_observed_data()` - Update observed dataset

**Key Functions**:
- `get_abc_sample()` - JIT-compiled core sampling function with 9 parameters
  - JAX-compatible with `static_argnums=(1, 2, 3, 4, 8)` for functions and n_obs
  - Handles both raw data and summary statistics comparison
  - Uses `lax.while_loop` for efficient rejection sampling
- `get_abc_samples_vectorized()` - Backward compatibility wrapper
- `get_training_samples()` - Backward compatibility wrapper

**JAX Optimizations**:
- JIT compilation for performance with proper static argument handling
- Vectorized sampling using JAX vmap
- Numerical stability with proper array handling
- Support for summary statistics without Python control flow

**Algorithm Details**:
- Uses JAX `lax.while_loop` for rejection sampling
- Creates balanced datasets for NRE training (y=0: joint, y=1: marginal)
- Handles epsilon quantile computation with temporary infinite tolerance

---

### `simulator.py`
**Purpose**: Main ABC simulator class - high-level interface for ABC data generation

**Classes**:
- `ABCSimulator` - Main ABC simulation orchestrator
  - `__init__()` - Initialize with model, observed data, and configuration
  - `generate_samples()` - Generate multiple ABC samples
  - `generate_single_sample()` - Generate single ABC sample
  - `generate_training_samples()` - Create training data for NRE
  - `get_epsilon_quantile()` - Compute epsilon from distance distribution
  - `set_epsilon_from_quantile()` - Set epsilon based on quantile
  - `save_configuration()` - Save complete configuration to YAML
  - `load_configuration()` - Load configuration from YAML
  - `update_epsilon()` - Update tolerance threshold
  - `update_observed_data()` - Update observed dataset
  - `set_model()` - Set statistical model
  - `get_model_info()` - Get model information
  - `get_summary_stats_info()` - Get summary statistics information
  - `_initialize_sampler()` - Private method to create RejectionSampler instance

**Backward Compatibility**:
- `ABCDataGenerator = ABCSimulator` - Alias for old class name

**Key Features**:
- High-level interface abstracting sampling details
- Configuration management with YAML serialization including model reconstruction
- Automatic sampler initialization and management with proper JAX function wrapping
- Support for quantile-based epsilon selection with lazy computation
- Comprehensive model and configuration introspection
- Lazy initialization of internal sampler to avoid unnecessary computation

**Configuration Structure**:
- `quantile_n_samples`: Number of samples for epsilon quantile computation
- `max_rejections`: Maximum rejection attempts
- `verbose`: Enable/disable logging
- `summary_stats_enabled`: Whether summary statistics are used

**YAML Persistence**:
- Complete configuration serialization including model class paths
- Observed data information storage
- Metadata support for experimental tracking

---

### `utils.py`
**Purpose**: Utility functions for configuration management and class importing

**Functions**:
- `import_class_from_string()` - Dynamically import class from string path
- `save_generator_config()` - Save configuration dictionary to YAML
- `load_generator_config()` - Load configuration from YAML
- `save_generator()` - Save complete simulator to file
- `load_generator()` - Load simulator from file
- `validate_generator_config()` - Validate configuration structure
- `get_generator_info()` - Get information about saved configuration
- `create_generator_template()` - Create template configuration files
- `list_generator_files()` - Find all generator files in directory
- `compare_generator_configs()` - Compare two configurations

**Template Support**:
- Pre-defined templates for "g_and_k" and "gauss_gauss" models
- Standardized configuration structure
- Validation and error handling

**Key Features**:
- Dynamic class importing for model deserialization
- YAML-based configuration persistence with validation
- Template generation for common model types
- Configuration comparison and validation utilities
- Directory scanning for generator files with automatic detection

---

## 📁 `src/abcnre/simulation/models/`

### `__init__.py`
**Purpose**: Package initialization for statistical models module

**Exports**:
- `StatisticalModel` - Base class for all statistical models
- `GaussGaussModel`, `GaussGaussMultiDimModel` - Gaussian models
- `GAndKModel`, `generate_g_and_k_samples`, `create_synthetic_g_and_k_data` - G-and-K models

**Key Features**:
- Clean API for statistical models
- Utility functions for synthetic data generation
- Backward compatibility with old generator names

---

### `base.py`
**Purpose**: Abstract base class defining the interface for all statistical models

**Classes**:
- `StatisticalModel(ABC)` - Abstract base class for statistical models
  - `prior_sample()` - Abstract: Sample from prior distribution
  - `simulate()` - Abstract: Simulate data given parameters and number of observations
  - `discrepancy_fn()` - Abstract: Compute distance between datasets
  - `get_model_args()` - Abstract: Get serialization arguments
  - `summary_stat_fn()` - Optional: Compute summary statistics
  - `transform_phi()` - Optional: Transform parameters to target parameter
  - `has_summary_stats()` - Check if summary statistics are implemented
  - `validate_parameters()` - Optional: Validate parameter constraints
  - `get_model_info()` - Get model metadata

**Key Features**:
- Standardized interface for all statistical models compatible with JAX JIT compilation
- Clear separation between required and optional methods
- Built-in parameter validation framework
- Support for parameter transformations (phi) for marginal inference
- Automatic detection of summary statistics capability
- JAX-compatible method signatures for efficient computation

**Interface Requirements**:
- All abstract methods must be JAX-compatible (no Python control flow)
- `simulate()` must accept (key, theta, n_obs) for vectorization
- `prior_sample()` must return JAX arrays
- `discrepancy_fn()` must return scalar distances

---

### `gauss_gauss.py`
**Purpose**: Gaussian-Gaussian models with known variance

**Classes**:
- `GaussGaussModel(StatisticalModel)` - 1D Gaussian-Gaussian model
  - `__init__()` - Initialize with prior/model parameters (mu0, sigma0, sigma)
  - `prior_sample()` - Sample from Gaussian prior N(mu0, sigma0²)
  - `simulate()` - Sample from Gaussian likelihood N(theta, sigma²) with JAX-compatible theta handling
  - `summary_stat_fn()` - Return sample mean (sufficient statistic)
  - `get_analytical_posterior_stats()` - Compute exact posterior parameters
  - `get_posterior_distribution()` - Return scipy posterior distribution
  - `update_model_params()` - Update model parameters
  - `transform_phi()` - Identity transformation (theta is parameter of interest)
  - `validate_parameters()` - Always returns True (no constraints for Gaussian)

- `GaussGaussMultiDimModel(StatisticalModel)` - Multidimensional version
  - `__init__()` - Initialize with multivariate parameters and Cholesky decompositions
  - `prior_sample()` - Sample from multivariate Gaussian prior using Cholesky
  - `simulate()` - Sample from multivariate Gaussian likelihood
  - `summary_stat_fn()` - Return sample mean vector
  - `transform_phi()` - Return first component as scalar by default

**Backward Compatibility**:
- `GaussGaussWithKnownStdGenerator = GaussGaussModel`
- `GaussGaussWithKnownStdMultiDimGenerator = GaussGaussMultiDimModel`

**Key Features**:
- Conjugate Gaussian model with analytical posterior for validation
- Support for both 1D and multidimensional cases
- Efficient Cholesky decomposition for multivariate sampling
- Complete analytical validation capabilities
- JAX-compatible parameter handling (no `.item()` calls in JIT context)

**Mathematical Model**:
- Prior: θ ~ N(μ₀, Σ₀)
- Likelihood: X|θ ~ N(θ, Σ)
- Posterior: θ|X ~ N(μₚₒₛₜ, Σₚₒₛₜ) (analytically tractable)

**JAX Compatibility Notes**:
- Uses `theta.flatten()[0]` instead of `theta.item()` for scalar extraction
- Handles both scalar and array theta inputs safely
- All operations vectorizable with vmap

---

### `g_and_k.py`
**Purpose**: G-and-K distribution models following Fearnhead & Prangle (2011)

**Classes**:
- `GAndKModel(StatisticalModel)` - G-and-K distribution model
  - `__init__()` - Initialize with parameter bounds (uniform prior on [0,10]⁴)
  - `prior_sample()` - Sample from uniform prior with proper constraints
  - `simulate()` - Generate G-and-K samples using quantile transformation
  - `summary_stat_fn()` - Return order statistics (sorted data)
  - `get_evenly_spaced_order_stats()` - Select subset of order statistics
  - `compute_theoretical_quantiles()` - Compute theoretical quantiles
  - `validate_parameters()` - Check constraints (B > 0, k > -0.5)
  - `transform_phi()` - Return A parameter (location) by default
  - `get_parameter_names()` - Return ['A', 'B', 'g', 'k']
  - `get_param_bounds()` - Return parameter bounds dictionary

**Key Functions**:
- `generate_g_and_k_samples()` - JIT-compiled G-and-K sampling with numerical stability
- `compute_g_and_k_quantiles()` - Compute theoretical quantiles for validation
- `create_synthetic_g_and_k_data()` - Generate test datasets with optional noise
- `get_fearnhead_prangle_setup()` - Reference experimental setup from paper
- `create_order_statistics_subset()` - Dimension reduction for summary stats
- `create_g_and_k_benchmark_study()` - Generate benchmark datasets for validation

**Backward Compatibility**:
- `GAndKGenerator = GAndKModel`

**Key Features**:
- Implementation following Fearnhead & Prangle (2011) exactly
- Numerically stable G-and-K transformation with overflow/underflow protection
- Order statistics as summary statistics for dimension reduction
- Parameter constraint validation (B > 0, k > -0.5)
- Comprehensive benchmarking utilities for reproducible research

**Mathematical Model**:
- Parameters: A (location), B (scale), g (skewness), k (kurtosis)
- Quantile function: Q(u) = A + B(1 + c·h(z))(1 + z²)ᵏz
- Where: z = Φ⁻¹(u), h(z) = (1-exp(-gz))/(1+exp(-gz)), c = 0.8
- Constraints: B > 0, k > -0.5

**Numerical Stability**:
- Handles extreme parameter values with `jnp.where` for NaN/Inf protection
- Stable computation of skewness term for small |g| values
- Robust quantile function implementation

---

## 📁 `tests/unit/test_simulation/`

### `test_gauss_gauss.py`
**Purpose**: Comprehensive test suite for Gaussian-Gaussian model and ABC simulator integration

**Test Classes**:
- `TestGaussGaussModel` - Unit tests for GaussGaussModel
  - `test_model_initialization()` - Verify model parameter setup and validation
  - `test_model_validation()` - Test parameter constraint checking
  - `test_prior_sampling()` - Validate prior sampling distribution
  - `test_data_simulation()` - Test data generation with correct statistics
  - `test_summary_statistics()` - Verify summary statistic computation
  - `test_discrepancy_function()` - Test distance calculation
  - `test_analytical_posterior()` - Compare with analytical solution

- `TestABCSimulatorWithGaussGauss` - Integration tests
  - `test_simulator_initialization()` - Test simulator setup with model
  - `test_single_sample_generation()` - Validate single ABC sample
  - `test_multiple_samples_generation()` - Test batch sampling
  - `test_training_samples_generation()` - Verify NRE training data format
  - `test_epsilon_quantile_computation()` - Test automatic epsilon calculation
  - `test_simulator_with_quantile_initialization()` - Test quantile-based setup

**Enhanced End-to-End Test**:
- `test_end_to_end_workflow()` - Complete workflow validation with visualization
  - Tests three different epsilon values (50%, 10%, 1% quantiles)
  - Generates comparative KDE plots using seaborn
  - Creates 4-panel visualization: KDE overlays, distance distributions, epsilon vs posterior mean, summary table
  - Validates statistical consistency across different tolerance levels
  - Saves high-resolution plots as 'abc_posterior_analysis.png'

**Key Features**:
- Complete test coverage for both unit and integration scenarios
- Statistical validation of ABC properties
- Visual verification of posterior behavior across epsilon values
- Analytical posterior comparison for Gaussian model
- Training data format validation for neural ratio estimation
- Comprehensive error handling and edge case testing

**Visualization Components**:
- Seaborn KDE plots for posterior distributions
- Distance distribution analysis
- Epsilon sensitivity analysis
- Summary statistics table
- True parameter value overlay for validation

---

## 📁 Root Test Files

### `test_yaml_persistence.py`
**Purpose**: Complete validation of YAML persistence cycle for simulator configurations

**Main Function**:
- `test_yaml_persistence_cycle()` - End-to-end persistence validation
  - **Creation Phase**: Generate synthetic data and create original simulator
  - **Simulation Phase**: Generate 300 ABC samples from original simulator
  - **Saving Phase**: Serialize complete configuration to YAML with metadata
  - **Loading Phase**: Reconstruct simulator from YAML configuration
  - **Verification Phase**: Compare configurations with numerical precision checks
  - **Comparative Simulation**: Generate samples from loaded simulator with identical seed
  - **Statistical Comparison**: Kolmogorov-Smirnov test for distribution equality
  - **Visualization**: Overlayed KDE plots and comparative histograms

**Validation Components**:
- Configuration consistency (epsilon preservation to 1e-6 precision)
- Model parameter preservation
- Observed data integrity
- Statistical distribution equivalence
- Professional visualization with publication-quality plots

**Key Features**:
- Temporary directory management for test isolation
- Complete YAML round-trip validation
- Statistical hypothesis testing for distribution equality
- Professional output suitable for research documentation
- Comprehensive error handling and assertion testing
- High-resolution plot generation and export

**Technical Validation**:
- Epsilon preservation: < 1e-6 difference
- Data preservation: < 1e-6 maximum difference
- Statistical equivalence: KS test p-value > 0.05
- Configuration integrity: Exact model parameter matching

---

## 🔄 Data Flow Overview

```
Statistical Model → ABCSimulator → RejectionSampler → Results
                ↓                    ↓                  ↓
            Model Interface     JAX/JIT Sampling    Structured
            (base.py)           Implementation       Outputs
                ↓                    ↓                  ↓
            Concrete Models     Configuration        Training Data
            (gauss_gauss.py,    Management           for NRE
             g_and_k.py)        (utils.py)               ↓
                ↓                    ↓              Visualization
            YAML Persistence    Test Validation    & Analysis
            (simulator.py)      (test files)       (KDE plots)
```

## 🎯 Key Design Patterns

### **Model Interface Pattern**:
- `StatisticalModel`: Abstract interface with JAX-compatible signatures
- Concrete implementations: `GaussGaussModel`, `GAndKModel`
- Clear separation of model logic from simulation logic

### **Separation of Concerns**:
- `ABCSimulator`: High-level interface and configuration management
- `RejectionSampler`: Low-level sampling implementation with JIT optimization
- `StatisticalModel`: Model-specific logic with parameter validation
- `utils.py`: Configuration serialization and class importing

### **JAX Integration**:
- All sampling functions are JIT-compiled with proper static argument handling
- Uses JAX random keys throughout for reproducibility
- Vectorized operations with vmap for efficiency
- Compatible with JAX data structures and control flow restrictions

### **Configuration-Driven**:
- YAML-based serialization for complete reproducibility
- Template support for common patterns
- Dynamic class importing for model reconstruction
- Metadata support for experimental tracking

### **Result Structures**:
- Type-safe NamedTuple containers for all outputs
- Consistent interface across sampling operations
- Support for both single and batch operations
- Training data formatted for binary classification (joint vs marginal)

### **Testing Strategy**:
- Unit tests for individual components
- Integration tests for complete workflows
- Statistical validation with analytical solutions
- Visual verification with publication-quality plots
- YAML persistence round-trip testing

## 🧪 Testing Strategy

Each module has comprehensive testing:
- `test_simulator.py` - ABCSimulator class functionality
- `test_sampler.py` - RejectionSampler implementation
- `test_utils.py` - Configuration management utilities
- `test_models_base.py` - StatisticalModel interface
- `test_gauss_gauss.py` - Gaussian models with analytical validation
- `test_g_and_k.py` - G-and-K models with reference implementation
- `test_yaml_persistence.py` - Complete persistence cycle validation
- `test_integration.py` - End-to-end workflow testing

## 🚀 Usage Examples

### Basic Model Usage:
```python
from abcnre.simulation import ABCSimulator
from abcnre.simulation.models import GaussGaussModel

# Create model
model = GaussGaussModel(mu0=0.0, sigma0=2.0, sigma=0.5)

# Create simulator
simulator = ABCSimulator(model, observed_data, epsilon=0.1)

# Generate samples
samples = simulator.generate_samples(key, 1000)
```

### Advanced G-and-K Usage:
```python
from abcnre.simulation.models import GAndKModel, create_synthetic_g_and_k_data

# Generate synthetic data
observed_data, true_params = create_synthetic_g_and_k_data(key)

# Create model and simulator
model = GAndKModel(prior_bounds=(0.0, 10.0))
simulator = ABCSimulator(model, observed_data, quantile_distance=0.01)

# Training data for NRE
training_data = simulator.generate_training_samples(key, 2000)
```

### Configuration Management:
```python
# Save complete setup
simulator.save_configuration("experiments/config.yml")

# Load and reproduce
new_simulator = ABCSimulator()
new_simulator.load_configuration("experiments/config.yml")
```

### Statistical Validation:
```python
# Compare with analytical solution (Gaussian case)
posterior_stats = model.get_analytical_posterior_stats(observed_data)
abc_mean = jnp.mean(abc_samples.theta_samples)
analytical_mean = posterior_stats['posterior_mean']
``` - Compute theoretical quantiles
  - `validate_parameters()` - Check constraints (B > 0, k > -0.5)

**Key Functions**:
- `generate_g_and_k_samples()` - JIT-compiled G-and-K sampling
- `compute_g_and_k_quantiles()` - Compute theoretical quantiles
- `create_synthetic_g_and_k_data()` - Generate test datasets
- `get_fearnhead_prangle_setup()` - Reference experimental setup
- `create_order_statistics_subset()` - Dimension reduction for summary stats
- `create_g_and_k_benchmark_study()` - Generate benchmark datasets

**Backward Compatibility**:
- `GAndKGenerator = GAndKModel`

**Key Features**:
- Implementation following Fearnhead & Prangle (2011) exactly
- Numerically stable G-and-K transformation
- Order statistics as summary statistics
- Parameter constraint validation
- Comprehensive benchmarking utilities

**Mathematical Model**:
- Parameters: A (location), B (scale), g (skewness), k (kurtosis)
- Quantile function: Q(u) = A + B(1 + c·h(z))(1 + z²)ᵏz
- Where: z = Φ⁻¹(u), h(z) = (1-exp(-gz))/(1+exp(-gz))
- Constraints: B > 0, k > -0.5

---

## 🔄 Data Flow Overview

```
Statistical Model → ABCSimulator → RejectionSampler → Results
                ↓                    ↓                  ↓
            Model Interface     JAX/JIT Sampling    Structured
            (base.py)           Implementation       Outputs
                ↓                    ↓                  ↓
            Concrete Models     Configuration        Training Data
            (gauss_gauss.py,    Management           for NRE
             g_and_k.py)        (utils.py)
```

## 🎯 Key Design Patterns

### **Model Interface Pattern**:
- `StatisticalModel`: Abstract interface
- Concrete implementations: `GaussGaussModel`, `GAndKModel`
- Clear separation of model logic from simulation logic

### **Separation of Concerns**:
- `ABCSimulator`: High-level interface and configuration
- `RejectionSampler`: Low-level sampling implementation
- `StatisticalModel`: Model-specific logic
- `utils.py`: Configuration and serialization

### **JAX Integration**:
- All sampling functions are JIT-compiled
- Uses JAX random keys throughout
- Vectorized operations with vmap
- Compatible with JAX data structures

### **Configuration-Driven**:
- YAML-based serialization for complete reproducibility
- Template support for common patterns
- Validation and error handling
- Model arguments stored for reconstruction

### **Result Structures**:
- Type-safe NamedTuple containers
- Consistent interface across sampling operations
- Support for both single and batch operations
- Training data formatted for binary classification

## 🧪 Testing Strategy

Each file should have corresponding tests:
- `test_simulator.py` - Test ABCSimulator class
- `test_sampler.py` - Test RejectionSampler implementation
- `test_utils.py` - Test configuration management
- `test_models_base.py` - Test StatisticalModel interface
- `test_gauss_gauss.py` - Test Gaussian models
- `test_g_and_k.py` - Test G-and-K models
- `test_integration.py` - Test complete workflow

## 🚀 Usage Examples

### Basic Model Usage:
```python
from abcnre.simulation import ABCSimulator
from abcnre.simulation.models import GaussGaussModel

# Create model
model = GaussGaussModel(mu0=0.0, sigma0=2.0, sigma=0.5)

# Create simulator
simulator = ABCSimulator(model, observed_data, epsilon=0.1)

# Generate samples
samples = simulator.generate_samples(key, 1000)
```

### Advanced G-and-K Usage:
```python
from abcnre.simulation.models import GAndKModel, create_synthetic_g_and_k_data

# Generate synthetic data
observed_data, true_params = create_synthetic_g_and_k_data(key)

# Create model and simulator
model = GAndKModel(prior_bounds=(0.0, 10.0))
simulator = ABCSimulator(model, observed_data, quantile_distance=0.01)

# Training data for NRE
training_data = simulator.generate_training_samples(key, 2000)
```

### Configuration Management:
```python
# Save complete setup
simulator.save_configuration("experiments/config.yml")

# Load and reproduce
new_simulator = ABCSimulator()
new_simulator.load_configuration("experiments/config.yml")
```
