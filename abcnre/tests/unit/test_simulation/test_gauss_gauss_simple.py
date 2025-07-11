"""
Simple test for Gaussian-Gaussian model and simulator.

This test validates the basic functionality of the GaussGaussModel
with the ABCSimulator to ensure the Phase 1 migration works correctly.
"""

import pytest
import jax.numpy as jnp
from jax import random
import numpy as np

# Imports from our new architecture
from abcnre.simulation import ABCSimulator
from abcnre.simulation.models import GaussGaussModel


class TestGaussGaussModel:
    """Test class for GaussGaussModel."""
    
    def test_model_initialization(self):
        """Test that GaussGaussModel initializes correctly."""
        model = GaussGaussModel(mu0=0.0, sigma0=2.0, sigma=0.5)
        
        assert model.mu0 == 0.0
        assert model.sigma0 == 2.0
        assert model.sigma == 0.5
        
        # Test model args serialization
        args = model.get_model_args()
        assert args == {'mu0': 0.0, 'sigma0': 2.0, 'sigma': 0.5}
    
    def test_model_validation(self):
        """Test parameter validation."""
        # Valid model
        model = GaussGaussModel(mu0=0.0, sigma0=1.0, sigma=1.0)
        assert model.validate_parameters(jnp.array(2.5))
        
        # Invalid standard deviations should raise error
        with pytest.raises(ValueError, match="Standard deviations must be positive"):
            GaussGaussModel(mu0=0.0, sigma0=-1.0, sigma=1.0)
    
    def test_prior_sampling(self):
        """Test prior sampling functionality."""
        model = GaussGaussModel(mu0=1.0, sigma0=2.0, sigma=0.5)
        key = random.PRNGKey(42)
        
        # Sample multiple times to check randomness
        samples = []
        for i in range(100):
            key, subkey = random.split(key)
            sample = model.prior_sample(subkey)
            samples.append(float(sample))
        
        samples = jnp.array(samples)
        
        # Check that samples have approximately correct mean and std
        sample_mean = jnp.mean(samples)
        sample_std = jnp.std(samples)
        
        # Should be close to prior parameters (with some tolerance for randomness)
        assert abs(sample_mean - 1.0) < 0.5  # Should be close to mu0=1.0
        assert abs(sample_std - 2.0) < 0.5   # Should be close to sigma0=2.0
    
    def test_data_simulation(self):
        """Test data simulation given parameters."""
        model = GaussGaussModel(mu0=0.0, sigma0=2.0, sigma=0.5)
        key = random.PRNGKey(123)
        
        theta = 2.5  # Parameter value
        n_obs = 100
        
        # Simulate data
        simulated_data = model.simulate(key, theta, n_obs)
        
        # Check output shape and properties
        assert simulated_data.shape == (n_obs,)
        
        # Data should be approximately centered around theta
        data_mean = jnp.mean(simulated_data)
        assert abs(data_mean - theta) < 0.2  # Should be close to theta=2.5
    
    def test_summary_statistics(self):
        """Test summary statistics computation."""
        model = GaussGaussModel(mu0=0.0, sigma0=2.0, sigma=0.5)
        
        # Test data
        test_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        summary = model.summary_stat_fn(test_data)
        
        # Should return mean
        expected_mean = jnp.mean(test_data)
        assert jnp.allclose(summary, jnp.array([expected_mean]))
        assert summary.shape == (1,)
    
    def test_discrepancy_function(self):
        """Test distance computation."""
        model = GaussGaussModel(mu0=0.0, sigma0=2.0, sigma=0.5)
        
        data1 = jnp.array([1.0, 2.0, 3.0])
        data2 = jnp.array([1.1, 2.1, 3.1])
        
        distance = model.discrepancy_fn(data1, data2)
        
        # Should be Euclidean distance
        expected_distance = jnp.linalg.norm(data1 - data2)
        assert jnp.allclose(distance, expected_distance)
    
    def test_analytical_posterior(self):
        """Test analytical posterior computation."""
        model = GaussGaussModel(mu0=0.0, sigma0=2.0, sigma=0.5)
        
        # Create some observed data
        observed_data = jnp.array([2.1, 1.9, 2.0, 2.2, 1.8])  # Mean ≈ 2.0
        
        # Get analytical posterior
        posterior_stats = model.get_analytical_posterior_stats(observed_data)
        
        # Check that we get reasonable posterior parameters
        assert 'posterior_mean' in posterior_stats
        assert 'posterior_std' in posterior_stats
        
        # Posterior mean should be between prior mean and sample mean
        sample_mean = jnp.mean(observed_data)
        posterior_mean = posterior_stats['posterior_mean']
        
        # Should be a weighted average of prior and sample
        assert 0.0 <= posterior_mean <= sample_mean or sample_mean <= posterior_mean <= 0.0


class TestABCSimulatorWithGaussGauss:
    """Test ABCSimulator with GaussGaussModel."""
    
    def setup_method(self):
        """Set up test data."""
        # Create synthetic observed data
        key = random.PRNGKey(42)
        true_theta = 2.5
        true_sigma = 0.5
        n_obs = 50
        
        self.observed_data = true_theta + true_sigma * random.normal(key, shape=(n_obs,))
        self.true_theta = true_theta
        
        # Create model
        self.model = GaussGaussModel(mu0=0.0, sigma0=3.0, sigma=0.5)
    
    def test_simulator_initialization(self):
        """Test ABCSimulator initialization with GaussGaussModel."""
        simulator = ABCSimulator(
            model=self.model,
            observed_data=self.observed_data,
            epsilon=1.0
        )
        
        assert simulator.model is self.model
        assert jnp.array_equal(simulator.observed_data, self.observed_data)
        assert simulator.epsilon == 1.0
        assert simulator.has_summary_stats()  # GaussGauss has summary stats
    
    def test_single_sample_generation(self):
        """Test generating a single ABC sample."""
        simulator = ABCSimulator(
            model=self.model,
            observed_data=self.observed_data,
            epsilon=1.0  # Fairly permissive for testing
        )
        
        key = random.PRNGKey(123)
        result = simulator.generate_single_sample(key)
        
        # Check result structure
        assert hasattr(result, 'sim_data')
        assert hasattr(result, 'theta')
        assert hasattr(result, 'distance')
        assert hasattr(result, 'summary_stat')
        
        # Check shapes
        assert result.sim_data.shape == self.observed_data.shape
        assert jnp.isscalar(result.theta)
        assert jnp.isscalar(result.distance)
        assert result.summary_stat.shape == (1,)  # Should be sample mean
        
        # Distance should be less than epsilon
        assert result.distance <= 1.0
    
    def test_multiple_samples_generation(self):
        """Test generating multiple ABC samples."""
        simulator = ABCSimulator(
            model=self.model,
            observed_data=self.observed_data,
            epsilon=2.0  # More permissive for multiple samples
        )
        
        key = random.PRNGKey(456)
        n_samples = 20
        
        result = simulator.generate_samples(key, n_samples)
        
        # Check result structure
        assert hasattr(result, 'sim_data')
        assert hasattr(result, 'theta_samples')
        assert hasattr(result, 'distances')
        assert hasattr(result, 'summary_stats')
        
        # Check shapes
        assert result.sim_data.shape == (n_samples,) + self.observed_data.shape
        assert result.theta_samples.shape == (n_samples,)
        assert result.distances.shape == (n_samples,)
        assert result.summary_stats.shape == (n_samples, 1)
        
        # All distances should be less than epsilon
        assert jnp.all(result.distances <= 2.0)
        
        # Posterior samples should be reasonable
        posterior_mean = jnp.mean(result.theta_samples)
        observed_mean = jnp.mean(self.observed_data)
        
        # ABC posterior should be somewhat close to observed mean
        assert abs(posterior_mean - observed_mean) < 1.0
    
    def test_training_samples_generation(self):
        """Test generating training samples for NRE."""
        simulator = ABCSimulator(
            model=self.model,
            observed_data=self.observed_data,
            epsilon=2.0
        )
        
        key = random.PRNGKey(789)
        n_samples = 100  # Will be split into 50 joint + 50 marginal
        
        result = simulator.generate_training_samples(key, n_samples, marginal_index=0)
        
        # Check result structure
        assert hasattr(result, 'features')
        assert hasattr(result, 'labels')
        assert hasattr(result, 'distances')
        
        # Check shapes
        expected_features = n_samples
        expected_feature_dim = 1 + 1  # theta (1D) + summary_stat (1D)
        
        assert result.features.shape == (expected_features, expected_feature_dim)
        assert result.labels.shape == (expected_features,)
        
        # Check labels are balanced (50% each class)
        n_joint = jnp.sum(result.labels == 0)
        n_marginal = jnp.sum(result.labels == 1)
        assert n_joint == n_marginal == n_samples // 2
        
        # Check feature ranges are reasonable
        theta_features = result.features[:, 0]  # First column is theta
        summary_features = result.features[:, 1]  # Second column is summary stat
        
        assert jnp.all(jnp.isfinite(theta_features))
        assert jnp.all(jnp.isfinite(summary_features))
    
    def test_epsilon_quantile_computation(self):
        """Test automatic epsilon computation from quantiles."""
        # First create simulator without automatic epsilon
        simulator = ABCSimulator(
            model=self.model,
            observed_data=self.observed_data,
            epsilon=jnp.inf  # Start with infinite epsilon
        )
        
        key = random.PRNGKey(999)
        
        # Compute 10% quantile epsilon
        epsilon_01, distances, _ = simulator.get_epsilon_quantile(key, alpha=0.1, n_samples=200)
        
        # Check that epsilon is reasonable
        assert epsilon_01 > 0.0
        assert epsilon_01 < jnp.inf
        
        # Check that approximately 10% of distances are below this epsilon
        fraction_below = jnp.mean(distances <= epsilon_01)
        assert abs(fraction_below - 0.1) < 0.05  # Within 5% tolerance
    
    def test_simulator_with_quantile_initialization(self):
        """Test initializing simulator with quantile_distance."""
        simulator = ABCSimulator(
            model=self.model,
            observed_data=self.observed_data,
            quantile_distance=0.05  # 5% acceptance rate
        )
        
        # Epsilon should be automatically computed
        assert simulator.epsilon < jnp.inf
        assert simulator.epsilon > 0.0
        
        # Test that we can generate samples with this epsilon
        key = random.PRNGKey(111)
        result = simulator.generate_samples(key, n_samples=10)
        
        # Should successfully generate samples
        assert result.theta_samples.shape == (10,)
        assert jnp.all(result.distances <= simulator.epsilon)


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    # 1. Create synthetic "observed" data
    key = random.PRNGKey(42)
    true_theta = 3.0
    observed_data = true_theta + 0.3 * random.normal(key, shape=(100,))
    
    # 2. Create model
    model = GaussGaussModel(mu0=0.0, sigma0=5.0, sigma=0.3)
    
    # 3. Test different epsilon values (50%, 10%, 1% quantiles)
    quantiles = [0.5, 0.1, 0.01]
    results = {}
    
    for i, quantile in enumerate(quantiles):
        print(f"\n--- Testing with {quantile*100:.0f}% quantile ---")
        
        # Create simulator with automatic epsilon
        simulator = ABCSimulator(
            model=model,
            observed_data=observed_data,
            quantile_distance=quantile
        )
        
        print(f"Computed epsilon = {simulator.epsilon:.6f}")
        
        # Generate ABC samples
        key, subkey = random.split(key)
        abc_samples = simulator.generate_samples(subkey, n_samples=500)
        
        # Store results
        results[f"{quantile*100:.0f}%"] = {
            'quantile': quantile,
            'epsilon': simulator.epsilon,
            'theta_samples': abc_samples.theta_samples,
            'distances': abc_samples.distances
        }
        
        # Print stats
        abc_posterior_mean = jnp.mean(abc_samples.theta_samples)
        abc_posterior_std = jnp.std(abc_samples.theta_samples)
        print(f"ABC posterior mean: {abc_posterior_mean:.3f}")
        print(f"ABC posterior std: {abc_posterior_std:.3f}")
        print(f"Mean distance: {jnp.mean(abc_samples.distances):.6f}")
    
    # 4. Create plots
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('ABC Posterior Analysis - Gaussian Model', fontsize=16, fontweight='bold')
        
        # Plot 1: KDE plot of theta samples
        ax1 = axes[0, 0]
        for label, data in results.items():
            sns.kdeplot(data['theta_samples'], ax=ax1, label=f"ε {label} quantile", linewidth=2)
        
        # Add true value
        ax1.axvline(true_theta, color='red', linestyle='--', linewidth=2, label='True θ')
        ax1.axvline(jnp.mean(observed_data), color='orange', linestyle=':', linewidth=2, label='Observed mean')
        
        ax1.set_xlabel('θ (parameter)')
        ax1.set_ylabel('Density')
        ax1.set_title('Posterior Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distance distributions
        ax2 = axes[0, 1]
        for label, data in results.items():
            sns.histplot(data['distances'], ax=ax2, alpha=0.6, label=f"ε {label} quantile", bins=30)
        
        ax2.set_xlabel('ABC Distance')
        ax2.set_ylabel('Count')
        ax2.set_title('Distance Distributions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Epsilon vs Posterior Mean
        ax3 = axes[1, 0]
        epsilons = [data['epsilon'] for data in results.values()]
        posterior_means = [jnp.mean(data['theta_samples']) for data in results.values()]
        quantile_labels = [data['quantile']*100 for data in results.values()]
        
        ax3.scatter(epsilons, posterior_means, s=100, c=['red', 'blue', 'green'], alpha=0.7)
        for i, (eps, mean, q) in enumerate(zip(epsilons, posterior_means, quantile_labels)):
            ax3.annotate(f'{q:.0f}%', (eps, mean), xytext=(5, 5), textcoords='offset points')
        
        ax3.axhline(true_theta, color='red', linestyle='--', alpha=0.7, label='True θ')
        ax3.set_xlabel('Epsilon (tolerance)')
        ax3.set_ylabel('Posterior Mean')
        ax3.set_title('Epsilon vs Posterior Mean')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        data_for_table = []
        for label, data in results.items():
            theta_mean = float(jnp.mean(data['theta_samples']))
            theta_std = float(jnp.std(data['theta_samples']))
            mean_dist = float(jnp.mean(data['distances']))
            data_for_table.append([label, f"{data['epsilon']:.6f}", f"{theta_mean:.3f}", f"{theta_std:.3f}", f"{mean_dist:.6f}"])
        
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=data_for_table,
                         colLabels=['Quantile', 'Epsilon', 'Post. Mean', 'Post. Std', 'Avg. Distance'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig('abc_posterior_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Plot saved as 'abc_posterior_analysis.png'")
        
    except ImportError:
        print("⚠️  matplotlib/seaborn not available for plotting")
    
    # 5. Generate training data
    simulator_final = ABCSimulator(
        model=model,
        observed_data=observed_data,
        quantile_distance=0.02  # Use 2% for training data
    )
    
    key, subkey = random.split(key)
    training_data = simulator_final.generate_training_samples(subkey, n_samples=200)
    
    # 6. Final checks
    observed_mean = jnp.mean(observed_data)
    
    print(f"\n🎯 Final Results:")
    print(f"   True theta: {true_theta:.3f}")
    print(f"   Observed mean: {observed_mean:.3f}")
    
    for label, data in results.items():
        abc_mean = jnp.mean(data['theta_samples'])
        print(f"   ABC mean ({label}): {abc_mean:.3f}")
    
    # Training data should be well-formed
    assert training_data.features.shape[0] == 200
    assert jnp.sum(training_data.labels == 0) == 100  # 50% joint samples
    assert jnp.sum(training_data.labels == 1) == 100  # 50% marginal samples
    
    print(f"   Training data shape: {training_data.features.shape}")
    print(f"   Labels balanced: {jnp.sum(training_data.labels == 0)}/{jnp.sum(training_data.labels == 1)}")
    
    print(f"\n✅ Complete workflow test passed!")


if __name__ == "__main__":
    # Run the enhanced end-to-end test
    test_end_to_end_workflow()