import os
import sys
path = os.getcwd()
print("Old path:", path)
path = path.split("/")
path = path[: path.index("ABC-SBI") + 1]
path = "/".join(path)
print("New path:", path)
os.chdir(path)
sys.path.append(path)
from functions.plots import plot_metric_for_many_datasets 
from functions.nre_cluster import for_a_dataset
from functions.metrics import c2stest, ranksumstest_stat, ranksumstest_pvalue
from jax import random,jit
import scipy.stats as stats
import jax.numpy as jnp
from scipy.stats import beta

if len(sys.argv)>1:
    PRIOR_ALPHA = float(sys.argv[1])
    PRIOR_BETA = PRIOR_ALPHA
else :
    PRIOR_ALPHA = 1
    PRIOR_BETA = 1

PATH_RESULTS = (
    os.getcwd()
    + "/examples/Beta-Binomial/new_clean_results/alpha_beta_{}".format(PRIOR_BETA)
)

@jit
def prior_simulator(key):
    return random.beta(key = key, a= PRIOR_ALPHA, b= PRIOR_BETA, shape = (1,))


@jit
def data_simulator(key, theta):
    return random.binomial(key, n = MODEL_N, p = theta, shape= (N_DATA,))


@jit
def discrepancy(y, y_true):
    return (jnp.mean(y)- jnp.mean(y_true))**2

def true_posterior_sample(key, TRUE_DATA, N_SAMPLE):
    alpha_post = PRIOR_ALPHA+ jnp.sum(TRUE_DATA)
    beta_post = PRIOR_BETA + len(TRUE_DATA)* MODEL_N - jnp.sum(TRUE_DATA)
    return random.beta(key, a = alpha_post, b = beta_post, shape = (N_SAMPLE,))

def true_posterior_pdf(theta, TRUE_DATA):
    alpha_post = PRIOR_ALPHA+ jnp.sum(TRUE_DATA)
    beta_post = PRIOR_BETA + len(TRUE_DATA)* MODEL_N - jnp.sum(TRUE_DATA)
    return beta.pdf(theta, alpha_post, beta_post)

MODEL_N = 1
PRIOR_DIST = beta(PRIOR_ALPHA, PRIOR_BETA)
MODEL_ARGS = {"N": MODEL_N}
PRIOR_ARGS = {'ALPHA': PRIOR_ALPHA, "BETA": PRIOR_BETA}


key = random.PRNGKey(5)


N_DATA = 100
N_KDE = 10000
N_POINTS = 500000
N_SAMPLE = 10000
N_SAMPLES = 3
N_DATASETS = 1
N_EPOCHS = 100
N_GRID = 1000
ALPHAS = [1.0, .9, .5,.1,.05,.01, .005, .001]
INDEX_MARGINAL = 0


PATH_FIGURES = PATH_RESULTS + "figures/"
PATH_POSTERIORS = PATH_FIGURES + "posterior_check/"
PATH_PICKLES = PATH_RESULTS + "pickles/"
PATH_CSV = PATH_RESULTS + "csv/"
 
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)
if not os.path.exists(PATH_FIGURES):
    os.makedirs(PATH_FIGURES)
if not os.path.exists(PATH_PICKLES):
    os.makedirs(PATH_PICKLES)
if not os.path.exists(PATH_CSV):
    os.makedirs(PATH_CSV)
if not os.path.exists(PATH_POSTERIORS):
    os.makedirs(PATH_POSTERIORS)
    
LEARNING_RATE = 0.001
PATIENCE = 7
COOLDOWN = 0
FACTOR = 0.5
RTOL = 1e-4
ACCUMULATION_SIZE = 200
LEARNING_RATE_MIN = 1e-6

BATCH_SIZE = 256
NUM_BATCH = 1024
NUM_CLASSES = 2
HIDDEN_SIZE = 256
NUM_LAYERS = 2
WDECAY = 0.001

NN_ARGS = {
    "N_EPOCH": N_EPOCHS,
    "NUM_LAYERS": NUM_LAYERS,
    "HIDDEN_SIZE": HIDDEN_SIZE,
    "BATCH_SIZE": BATCH_SIZE,
    "NUM_BATCH": NUM_BATCH,
    "LEARNING_RATE": LEARNING_RATE,
    "WDECAY": WDECAY,
    "PATIENCE": PATIENCE,
    "COOLDOWN": COOLDOWN,
    "FACTOR": FACTOR,
    "RTOL": RTOL,
    "ACCUMULATION_SIZE": ACCUMULATION_SIZE,
    "LEARNING_RATE_MIN": LEARNING_RATE_MIN
}


PARAMS = {}
TEST_ACCURACY = {}
TRAIN_ACCURACY = {}
TEST_LOSSES = {}
TRAIN_LOSSES = {}

TIME_SIMULATIONS = {}
TIME_TRAINING = {}
TIME_EVAL = {}
EPSILONS = {}

TRUE_DATAS = {}
TRUE_THETAS = {}

METRICS = {}

METRICS_TO_TEST ={"C2ST": c2stest, "RS_stat": ranksumstest_stat, "RS_pvalue": ranksumstest_pvalue}


for i_dataset in range(N_DATASETS):
    key, key_i = random.split(key)
    (
        PARAMS[i_dataset],
        TEST_ACCURACY[i_dataset],
        TRAIN_ACCURACY[i_dataset],
        TEST_LOSSES[i_dataset],
        TRAIN_LOSSES[i_dataset],
        TIME_SIMULATIONS[i_dataset],
        TIME_TRAINING[i_dataset],
        TIME_EVAL[i_dataset],
        METRICS[i_dataset],
        TRUE_DATAS[i_dataset],
        TRUE_THETAS[i_dataset],
    ) = for_a_dataset(i_dataset= i_dataset, key = key_i, N_POINTS= N_POINTS, prior_simulator= prior_simulator, data_simulator= data_simulator, discrepancy= discrepancy, true_posterior_pdf= true_posterior_pdf, true_posterior_sample= true_posterior_sample, ALPHAS= ALPHAS, PRIOR_DIST= PRIOR_DIST, PRIOR_ARGS = PRIOR_ARGS, MODEL_ARGS= MODEL_ARGS, NN_ARGS= NN_ARGS, N_GRID= N_GRID, N_KDE= N_KDE, N_SAMPLE= N_SAMPLE, N_SAMPLES= N_SAMPLES, METRICS_TO_TEST= METRICS_TO_TEST, PATH= PATH_RESULTS, index_marginal= INDEX_MARGINAL)

plot_metric_for_many_datasets("C2ST", ALPHAS, METRICS, N_SAMPLES, N_DATASETS, PATH_FIGURES + "c2st.png", show = False, title = "For 10 differents $\\theta$")
plot_metric_for_many_datasets("RS_stat", ALPHAS, METRICS, N_SAMPLES, N_DATASETS, PATH_FIGURES + "ranksums.png", show = False, title = "For 10 differents $\\theta$")