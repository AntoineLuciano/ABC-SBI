import yaml
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from abcnre.simulation import ABCSimulator
from abcnre.simulation.models import GaussGaussModel
from abcnre.inference import MLPNetwork, NeuralRatioEstimator
from abcnre.persistence import EstimatorCheckpoint
from abcnre.validation import run_comprehensive_validation

# 1. Load simulator config from YAML
with open('gauss_1D_simulator.yml') as f:
    sim_cfg = yaml.safe_load(f)
model_cfg = sim_cfg['model']
model = GaussGaussModel(
    mu0=model_cfg['parameters']['mu0'],
    sigma0=model_cfg['parameters']['sigma0'],
    sigma=model_cfg['parameters']['sigma']
)

# 2. Generate observed data
key = random.PRNGKey(sim_cfg.get('training', {}).get('seed', 0))
key, obs_key = random.split(key)
theta_true = sim_cfg['observed_data']['theta']
n_obs = sim_cfg['observed_data']['n_obs']
observed = model.simulate(obs_key, theta_true, n_obs)

# 3. Initialize ABCSimulator
simulator = ABCSimulator(
    model=model,
    observed_data=observed,
    epsilon=sim_cfg['epsilon']
)

# 4. Load MLP training config
with open('config_mlp_simple.yml') as f:
    train_cfg = yaml.safe_load(f)
net_cfg = train_cfg['network']
train_params = train_cfg['training']

network = MLPNetwork(
    hidden_dims=net_cfg['hidden_dims'],
    dropout_rate=net_cfg['dropout_rate']
)

estimator = NeuralRatioEstimator(
    network=network,
    learning_rate=train_params['learning_rate'],
    random_seed=train_params['seed']
)

# 5. Train classifier
def data_gen(k, bs):
    return simulator.generate_training_samples(k, bs)

training_results = estimator.train(
    data_generator=data_gen,
    num_epochs=train_params['num_epochs'],
    batch_size=train_params['batch_size']
)

# 6. Save weights in NPZ
weights = estimator.state.params
np.savez('gauss_classifier_weights.npz',
         **jax.tree_util.tree_map(lambda x: np.array(x), weights))

# 7. Persist classifier config and weight path
clf_cfg = {
    'classifier': {
        'weight_path': 'gauss_classifier_weights.npz',
        'network_type': 'MLPNetwork'
    }
}
with open('gauss_1D_classifier.yml', 'w') as f:
    yaml.dump(clf_cfg, f)

# 8. Run diagnostics
results = run_comprehensive_validation(
    estimator,
    simulator,
    num_test_batches=5,
    batch_size=64,
    num_posterior_samples=500,
    true_value=theta_true,
    save_plots=False
)

print("Validation summary:", results['validation_summary'])
