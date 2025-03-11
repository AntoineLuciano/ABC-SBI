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
import jax.numpy as jnp
from scipy.stats import norm

if len(sys.argv)>1:
    K = int(sys.argv[1])
else: 
    K = 5
INDEX_MARGINAL = 0
PATH_RESULTS = (
    os.getcwd()
    + "/examples/Linear-Reg/new_clean_results/K_{}/".format(K)
)

@jit
def prior_simulator(key):
    return random.normal(key, (K,))*SIGMA0 + MU0

@jit
def data_simulator(key, betas):
    return random.normal(key, (X_DESIGN.shape[0],))*SIGMA+ jnp.dot(X_DESIGN, betas)

@jit
def discrepancy(y, y_true):
    return jnp.sum((jnp.dot(jnp.transpose(X_DESIGN),y-y_true))**2)
    # return jnp.sum((y-y_true)**2)

def x_design_simulator(key, n_data, K):
    X = random.normal(key, (n_data, K))
    X = (X-jnp.mean(X, axis=0))/jnp.std(X, axis=0)
    return X



def true_posterior_sample(key, TRUE_DATA, N_SAMPLE):
    COV0 = jnp.diag(jnp.array([SIGMA0**2]*K))
    PREC0 = jnp.linalg.inv(COV0)
    PREC_n = PREC0 + (1 / SIGMA**2) * (X_DESIGN.T @ X_DESIGN)
    Sigma_n = jnp.linalg.inv(PREC_n)
    mu_n = Sigma_n @ (PREC0 @ jnp.ones(X_DESIGN.shape[1])* MU0 + (1 / SIGMA**2) * (X_DESIGN.T @ TRUE_DATA))
    return random.normal(key, (N_SAMPLE,)) * jnp.sqrt(Sigma_n[INDEX_MARGINAL, INDEX_MARGINAL]) + mu_n[INDEX_MARGINAL]

def true_posterior_pdf(theta, TRUE_DATA):
    COV0 = jnp.diag(jnp.array([SIGMA0**2]*K))
    PREC0 = jnp.linalg.inv(COV0)
    PREC_n = PREC0 + (1 / SIGMA**2) * (X_DESIGN.T @ X_DESIGN)
    Sigma_n = jnp.linalg.inv(PREC_n)
    mu_n = Sigma_n @ (PREC0 @ jnp.ones(X_DESIGN.shape[1])* MU0 + (1 / SIGMA**2) * (X_DESIGN.T @ TRUE_DATA))
    return norm.pdf(theta, loc=mu_n[INDEX_MARGINAL], scale=jnp.sqrt(Sigma_n[INDEX_MARGINAL, INDEX_MARGINAL]))


key = random.PRNGKey(0)


N_DATA = 200
N_KDE = 10000
N_POINTS = 500000
N_SAMPLE = 10000
N_SAMPLES = 3
N_DATASETS = 10
N_EPOCHS = 100
N_GRID = 1000
ALPHAS = [1.0, .9, .5,.1,.05,.01, .005, .001]
SIGMA = 1.0
MU0, SIGMA0 = 0.0, 20.

PRIOR_DIST = norm(loc=MU0, scale=SIGMA0)

key, key_x = random.split(key)
X_DESIGN = x_design_simulator(key_x, N_DATA, K)
MODEL_ARGS = {"SIGMA": SIGMA, "X_DESIGN": X_DESIGN}
PRIOR_ARGS = {"MU0": MU0, "SIGMA0": SIGMA0}


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