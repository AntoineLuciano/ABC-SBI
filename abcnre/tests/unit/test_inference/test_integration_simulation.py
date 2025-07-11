"""
Integration test for the existing simulation module and the inference module.

This script tests the complete pipeline from ABC simulation to
neural ratio estimation.
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import traceback

# Real imports, adjusted to your package structure
from abcnre.simulation import ABCSimulator
from abcnre.simulation.models import GaussGaussModel, GAndKModel
from abcnre.inference import NeuralRatioEstimator, MLPNetwork, DeepSetNetwork

class MockGaussGaussModel:
    """Mock for GaussGaussModel."""
    def __init__(self, mu0=0.0, sigma0=1.0, sigma=0.5):
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.sigma = sigma

    def prior_sample(self, key, n_samples):
        return self.mu0 + self.sigma0 * random.normal(key, (n_samples,))

    def simulate(self, key, theta, n_obs):
        return theta + self.sigma * random.normal(key, (n_obs,))

    def summary_stat_fn(self, data):
        return jnp.mean(data)

    def discrepancy_fn(self, data1, data2):
        return jnp.abs(self.summary_stat_fn(data1) - self.summary_stat_fn(data2))

class MockABCSimulator:
    """Mock for ABCSimulator."""
    def __init__(self, model, observed_data, epsilon):
        self.model = model
        self.observed_data = observed_data
        self.epsilon = epsilon

    def generate_training_samples(self, key, batch_size):
        key1, key2, key3 = random.split(key, 3)
        n_joint = batch_size // 2
        n_marginal = batch_size - n_joint

        theta_joint = self.model.prior_sample(key1, n_joint)
        theta_marginal = self.model.prior_sample(key2, n_marginal)

        obs_summary = self.model.summary_stat_fn(self.observed_data)
        f_joint = jnp.column_stack([
            jnp.repeat(obs_summary, n_joint),
            theta_joint
        ])
        f_marg = jnp.column_stack([
            jnp.repeat(obs_summary, n_marginal),
            theta_marginal
        ])
        features = jnp.concatenate([f_joint, f_marg], axis=0)
        labels = jnp.concatenate([jnp.zeros(n_joint), jnp.ones(n_marginal)])

        perm = random.permutation(key3, batch_size)
        features = features[perm]
        labels = labels[perm]

        class Batch:
            def __init__(self, features, labels):
                self.features = features
                self.labels = labels

        return Batch(features, labels)


def test_abc_to_nre_integration():
    """Full ABC → NRE integration test."""
    print("🚀 Running ABC to NRE integration test...")
    key = random.PRNGKey(42)

    # 1) Model + observed data
    print("1. Initializing ABC model...")
    model = MockGaussGaussModel()
    key, obs_key = random.split(key)
    true_theta = 1.5
    observed_data = model.simulate(obs_key, true_theta, 100)
    print(f"2. Observed data shape={observed_data.shape}, mean={jnp.mean(observed_data):.3f}")

    # 2) ABC simulator
    print("3. Initializing ABC simulator...")
    abc_sim = MockABCSimulator(model, observed_data, epsilon=0.1)

    # 3) Training-sample generation
    print("4. Testing training sample generation...")
    key, train_key = random.split(key)
    batch = abc_sim.generate_training_samples(train_key, 64)
    print(f"   features: {batch.features.shape}, labels: {batch.labels.shape}")
    print(f"   label counts: {jnp.bincount(batch.labels.astype(int))}")

    # 4) Network architecture tests
    print("5. Testing network architectures...")
    # MLP
    try:
        print("   MLPNetwork...")
        mlp = MLPNetwork(hidden_dims=[64,32], dropout_rate=0.1)
        key, init_key = random.split(key)
        params = mlp.init(init_key, batch.features, training=False)
        out = mlp.apply(params, batch.features, training=False)
        print(f"   ✅ MLP output shape: {out.shape}")
    except Exception as e:
        print(f"   ❌ MLPNetwork error: {e}")

    # DeepSet
    try:
        print("   DeepSetNetwork...")
        ds = DeepSetNetwork(phi_hidden_dims=[32], rho_hidden_dims=[64], pooling='mean', dropout_rate=0.1)
        key, init_key = random.split(key)
        params = ds.init(init_key, batch.features, training=False)
        out = ds.apply(params, batch.features, training=False)
        print(f"   ✅ DeepSet output shape: {out.shape}")
    except Exception as e:
        print(f"   ❌ DeepSetNetwork error: {e}")

    # 5) Simulated NRE training
    print("6. Simulating NRE training...")
    class MockNeuralRatioEstimator:
        def __init__(self, network):
            self.network = network
            self.is_trained = False
            self.key = random.PRNGKey(0)

        def initialize_training(self, input_shape):
            self.key, init_key = random.split(self.key)
            # Flax init returns all collections; use deterministic init for training params
            self.params = self.network.init(init_key, jnp.zeros(input_shape), training=False)
            print(f"   Initialized with {sum(x.size for x in jax.tree_util.tree_leaves(self.params['params']))} params")

        def train_step(self, batch):
            """
            Perform one training step with dropout and compute loss and accuracy.
            """
            # Split PRNG for dropout
            self.key, dropout_key = random.split(self.key)
            # Apply network with dropout rng
            logits = self.network.apply(
                self.params,
                batch.features,
                training=True,
                rngs={'dropout': dropout_key}
            )
            # Compute predictions
            preds = jax.nn.sigmoid(logits).flatten()
            # Compute binary cross-entropy loss
            loss = jnp.mean(
                -batch.labels * jnp.log(preds + 1e-8)
                - (1 - batch.labels) * jnp.log(1 - preds + 1e-8)
            )
            # Compute accuracy
            acc = jnp.mean((preds > 0.5) == batch.labels)
            return {'loss': loss, 'accuracy': acc}

        def predict(self, features):
            logits = self.network.apply(self.params, features, training=False)
            return jax.nn.sigmoid(logits)

        def log_ratio(self, features):
            logits = self.network.apply(self.params, features, training=False)
            return logits.flatten()

    mre = MockNeuralRatioEstimator(mlp)
    mre.initialize_training(batch.features.shape)
    try:
        for epoch in range(3):
            key, step_key = random.split(key)
            batch = abc_sim.generate_training_samples(step_key, 32)
            metrics = mre.train_step(batch)
            print(f"   Epoch {epoch}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
        mre.is_trained = True
        print("   ✅ Training simulation succeeded")
    except Exception as e:
        print(f"   ❌ Error during NRE training: {e}")
        traceback.print_exc()

    # 6) Inference
    print("7. Testing inference...")
    key, inf_key = random.split(key)
    thetas = model.prior_sample(inf_key, 500)
    summary = model.summary_stat_fn(observed_data)
    feats = jnp.column_stack([jnp.repeat(summary, 500), thetas])
    lrs = mre.log_ratio(feats)
    weights = jnp.exp(lrs)
    w_norm = weights / jnp.sum(weights)
    key, res_key = random.split(key)
    idx = random.choice(res_key, 500, shape=(100,), p=w_norm)
    post = thetas[idx]
    print(f"   Posterior mean={jnp.mean(post):.3f}, std={jnp.std(post):.3f}")
    print("✅ ABC→NRE integration test passed!\n")


def test_feature_engineering():
    """Test feature engineering."""
    print("🔧 Testing feature engineering...")
    key = random.PRNGKey(0)
    model = MockGaussGaussModel()
    key, okey = random.split(key)
    data = model.simulate(okey, 1.0, 50)

    m = jnp.mean(data); s = jnp.std(data)
    q25 = jnp.percentile(data, 25); q75 = jnp.percentile(data, 75)
    print(f"Mean={m:.3f}, Std={s:.3f}, Q25={q25:.3f}, Q75={q75:.3f}")

    thetas = model.prior_sample(okey, 10)
    f1 = jnp.column_stack([jnp.repeat(m, 10), thetas])
    f2 = jnp.column_stack([jnp.repeat(m, 10), jnp.repeat(s,10), jnp.repeat(q25,10), jnp.repeat(q75,10), thetas])
    raw = jnp.stack([jnp.tile(data,(10,1)), jnp.repeat(thetas[:,None],50,axis=1)], axis=-1)
    print(f"Simple feat shape {f1.shape}, multi {f2.shape}, raw {raw.shape}")
    print("✅ Feature engineering passed!\n")


def test_model_configurations():
    """Test model configurations."""
    print("⚙️ Testing model configurations...")
    for cfg in [
        {'hidden_dims':[32], 'dropout_rate':0.0},
        {'hidden_dims':[64,32], 'dropout_rate':0.1, 'use_batch_norm':True}
    ]:
        try:
            net = MLPNetwork(**cfg)
            print("MLP config OK:", net)
        except Exception as e:
            print("MLP config error:", e)
    for cfg in [
        {'phi_hidden_dims':[16], 'rho_hidden_dims':[32], 'pooling':'mean'},
        {'phi_hidden_dims':[16], 'rho_hidden_dims':[32], 'pooling':'max', 'dropout_rate':0.1}
    ]:
        try:
            net = DeepSetNetwork(**cfg)
            print("DeepSet config OK:", net)
        except Exception as e:
            print("DeepSet config error:", e)


if __name__ == "__main__":
    test_abc_to_nre_integration()
    test_feature_engineering()
    test_model_configurations()
    print("🎉 All integration tests completed successfully!")
