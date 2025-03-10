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
from jax import random,jit
import scipy.stats as stats
import jax.numpy as jnp
from scipy.stats import norm

if len(sys.argv)>1:
    D = int(sys.argv[1])
else :
    D = 3
print("D = ", D)

PATH_RESULTS = (
    os.getcwd()
    + "/examples/Gauss-Gauss/Gauss_Gauss_Muli_known_sigma/new_clean_results/D_{}/".format(D)
)

@jit
def prior_simulator(key):
    return random.normal(key, (D,)) * SIGMA0 + MU0


@jit
def data_simulator(key, theta):
    return (random.normal(key, (D,N_DATA)) * SIGMA + theta[:,None]).astype(float)


@jit
def discrepancy(y, y_true):
    return jnp.sum((jnp.mean(y, axis= 1) - jnp.mean(y_true, axis =1)) ** 2)


def true_posterior_sample(key, TRUE_DATA, N_SAMPLE):
    TRUE_DATA_i = TRUE_DATA[INDEX_MARGINAL]
    mu_post = (MU0 * SIGMA**2 + SIGMA0**2 * jnp.sum(TRUE_DATA_i)) / (
        SIGMA0**2 * len(TRUE_DATA_i) + SIGMA**2
    )
    sigma2_post = 1 / (1 / SIGMA0**2 + len(TRUE_DATA_i) / SIGMA**2)
    return random.normal(key, (N_SAMPLE,)) * jnp.sqrt(sigma2_post) + mu_post

def true_posterior_pdf(theta, TRUE_DATA):
    TRUE_DATA_i = TRUE_DATA[INDEX_MARGINAL]
    mu_post = (MU0 * SIGMA**2 + SIGMA0**2 * jnp.sum(TRUE_DATA_i)) / (
        SIGMA0**2 * len(TRUE_DATA_i) + SIGMA**2
    )
    sigma2_post = 1 / (1 / SIGMA0**2 + len(TRUE_DATA_i) / SIGMA**2)
    return norm.pdf(theta, loc=mu_post, scale=jnp.sqrt(sigma2_post))
MU0 = 0
if len(sys.argv)>1:
    SIGMA0 = float(sys.argv[1])
else: 
    SIGMA0 = 20.
SIGMA = 1.
PRIOR_DIST = stats.norm(loc= MU0, scale= SIGMA0)
MODEL_ARGS = {"SIGMA": SIGMA}
PRIOR_ARGS = {"MU0": MU0, "SIGMA0":SIGMA0}


key = random.PRNGKey(0)


N_DATA = 100
N_KDE = 10000
N_POINTS = 500000
N_SAMPLE = 10000
N_SAMPLES = 3
N_DATASETS = 10
N_EPOCHS = 100
N_GRID = 1000
ALPHAS = [1.0, .9, .5,.1,.05,.01, .005, .001, .0005, .0001]
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

METRICS_ABC = {}
METRICS_NRE = {}
METRICS_CORRECTED_NRE = {}


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
        METRICS_ABC[i_dataset],
        METRICS_NRE[i_dataset],
        METRICS_CORRECTED_NRE[i_dataset],
        TRUE_DATAS[i_dataset],
        TRUE_THETAS[i_dataset],
    ) = for_a_dataset(i_dataset= i_dataset, key = key_i, N_POINTS= N_POINTS, prior_simulator= prior_simulator, data_simulator= data_simulator, discrepancy= discrepancy, true_posterior_pdf= true_posterior_pdf, true_posterior_sample= true_posterior_sample, ALPHAS= ALPHAS, PRIOR_DIST= PRIOR_DIST, PRIOR_ARGS = PRIOR_ARGS, MODEL_ARGS= MODEL_ARGS, NN_ARGS= NN_ARGS, N_GRID= N_GRID, N_KDE= N_KDE, N_SAMPLE= N_SAMPLE, N_SAMPLES= N_SAMPLES, PATH= PATH_RESULTS, index_marginal= INDEX_MARGINAL)

plot_metric_for_many_datasets("C2ST", ALPHAS, METRICS_ABC, METRICS_NRE, METRICS_CORRECTED_NRE, N_SAMPLES, N_DATASETS, PATH_FIGURES + "c2st.png", show = False, title = "For 10 differents $\\theta$")
plot_metric_for_many_datasets("RS_stat", ALPHAS, METRICS_ABC, METRICS_NRE, METRICS_CORRECTED_NRE, N_SAMPLES, N_DATASETS, PATH_FIGURES + "ranksums.png", show = False, title = "For 10 differents $\\theta$")