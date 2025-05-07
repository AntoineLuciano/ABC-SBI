import os
import sys
from jax import random, jit
import jax.numpy as jnp
import scipy.stats as stats
from functions.plots import plot_metric_for_many_datasets 
from functions.nre_cluster import for_a_dataset
from functions.metrics import c2stest, ranksumstest_stat, ranksumstest_pvalue

# Set path
path = os.getcwd()
path = path.split("/")
path = path[: path.index("ABC-SBI") + 1]
os.chdir("/".join(path))
sys.path.append("/".join(path))

# Hyperparameters for the priors
MU0 = 0.0
SIGMA0 = 10.0  # prior std dev for mu
a_sigma2 = 2.0
b_sigma2 = 2.0  # prior on sigma^2 ~ InvGamma(a, b)

@jit
def prior_simulator(key):
    key_mu, key_sigma2 = random.split(key)
    mu = random.normal(key_mu) * SIGMA0 + MU0
    gamma_sample = random.gamma(key_sigma2, a_sigma2)
    sigma2 = b_sigma2 / gamma_sample
    return jnp.array([mu, sigma2])

@jit
def data_simulator(key, theta):
    mu = theta[0]
    sigma2 = theta[1]
    sigma = jnp.sqrt(sigma2)
    return (random.normal(key, (N_DATA,)) * sigma + mu).astype(float)

@jit
def discrepancy(y, y_true):
    return (jnp.mean(y) - jnp.mean(y_true))**2

# No analytical posterior in this case (used dummy versions)
def true_posterior_sample(key, TRUE_DATA, N_SAMPLE):
    return jnp.zeros((N_SAMPLE, 2))

def true_posterior_pdf(theta, TRUE_DATA):
    return 0.0  # Not used

# Prior and model args
PRIOR_ARGS = {"MU0": MU0, "SIGMA0": SIGMA0, "a_sigma2": a_sigma2, "b_sigma2": b_sigma2}
MODEL_ARGS = {}
PRIOR_DIST = None  # not used

# Experiment setup
key = random.PRNGKey(0)
N_DATA = 100
N_KDE = 10000
N_POINTS = 500000
N_SAMPLE = 10000
N_SAMPLES = 3
N_DATASETS = 10
N_EPOCHS = 100
N_GRID = 1000
ALPHAS = [1.0, .9, .5, .1, .05, .01, .005, .001]
INDEX_MARGINAL = 0

PATH_RESULTS = "/".join(path) + "/examples/Gauss-Gauss/Gauss_Gauss_1D_unknown_sigma/results/"
PATH_FIGURES = PATH_RESULTS + "figures/"
PATH_POSTERIORS = PATH_FIGURES + "posterior_check/"
PATH_PICKLES = PATH_RESULTS + "pickles/"
PATH_CSV = PATH_RESULTS + "csv/"

for p in [PATH_RESULTS, PATH_FIGURES, PATH_PICKLES, PATH_CSV, PATH_POSTERIORS]:
    os.makedirs(p, exist_ok=True)

# Network and training hyperparams
NN_ARGS = {
    "N_EPOCH": N_EPOCHS,
    "NUM_LAYERS": 2,
    "HIDDEN_SIZE": 256,
    "BATCH_SIZE": 256,
    "NUM_BATCH": 1024,
    "LEARNING_RATE": 0.001,
    "WDECAY": 0.001,
    "PATIENCE": 7,
    "COOLDOWN": 0,
    "FACTOR": 0.5,
    "RTOL": 1e-4,
    "ACCUMULATION_SIZE": 200,
    "LEARNING_RATE_MIN": 1e-6,
}

PARAMS, TEST_ACCURACY, TRAIN_ACCURACY = {}, {}, {}
TEST_LOSSES, TRAIN_LOSSES = {}, {}
TIME_SIMULATIONS, TIME_TRAINING, TIME_EVAL = {}, {}, {}
EPSILONS, TRUE_DATAS, TRUE_THETAS, METRICS = {}, {}, {}, {}

METRICS_TO_TEST = {"C2ST": c2stest, "RS_stat": ranksumstest_stat, "RS_pvalue": ranksumstest_pvalue}

for i_dataset in range(N_DATASETS):
    key, key_i = random.split(key)
    (
        PARAMS[i_dataset], TEST_ACCURACY[i_dataset], TRAIN_ACCURACY[i_dataset],
        TEST_LOSSES[i_dataset], TRAIN_LOSSES[i_dataset], TIME_SIMULATIONS[i_dataset],
        TIME_TRAINING[i_dataset], TIME_EVAL[i_dataset], METRICS[i_dataset],
        TRUE_DATAS[i_dataset], TRUE_THETAS[i_dataset]
    ) = for_a_dataset(
        i_dataset=i_dataset, key=key_i, N_POINTS=N_POINTS,
        prior_simulator=prior_simulator, data_simulator=data_simulator,
        discrepancy=discrepancy, true_posterior_pdf=true_posterior_pdf,
        true_posterior_sample=true_posterior_sample, ALPHAS=ALPHAS,
        PRIOR_DIST=PRIOR_DIST, PRIOR_ARGS=PRIOR_ARGS, MODEL_ARGS=MODEL_ARGS,
        NN_ARGS=NN_ARGS, N_GRID=N_GRID, N_KDE=N_KDE, N_SAMPLE=N_SAMPLE,
        N_SAMPLES=N_SAMPLES, METRICS_TO_TEST=METRICS_TO_TEST,
        PATH=PATH_RESULTS, index_marginal=INDEX_MARGINAL
    )

plot_metric_for_many_datasets("C2ST", ALPHAS, METRICS, N_SAMPLES, N_DATASETS, PATH_FIGURES + "c2st.png", show=False, title="For 10 different $\\theta$")
plot_metric_for_many_datasets("RS_stat", ALPHAS, METRICS, N_SAMPLES, N_DATASETS, PATH_FIGURES + "ranksums.png", show=False, title="For 10 different $\\theta$")