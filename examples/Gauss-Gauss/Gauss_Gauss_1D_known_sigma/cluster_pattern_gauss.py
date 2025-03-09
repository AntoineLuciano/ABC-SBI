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
from functions.simulation import (
    get_dataset,
    NRE_corrected_posterior_sample,
    NRE_posterior_sample,
)
from functions.training import train_loop
from functions.plots import plot_metric_for_a_dataset, plot_metric_for_many_datasets, plot_posterior_comparison
from functions.metrics import evaluate_metrics
from functions.save import create_csv_for_a_dataset, create_pickle_for_a_dataset
from jax import random, jit, vmap
from sklearn.model_selection import train_test_split
import time
from scipy.stats import norm
import numpy as np
import jax.numpy as jnp

def ABC_gauss_single(key, true_data, epsilon):
    key, key_xbar = random.split(key)
    xbar = random.truncated_normal(key_xbar, lower = (jnp.mean(true_data)-jnp.sqrt(epsilon)-MU0)/jnp.sqrt(SIGMA0**2+SIGMA**2/len(true_data)), upper = (jnp.mean(true_data)+jnp.sqrt(epsilon)-MU0)/jnp.sqrt(SIGMA0**2+SIGMA**2/len(true_data)))*jnp.sqrt(SIGMA0**2+SIGMA**2/len(true_data)) + MU0
    dist = (jnp.mean(true_data)-xbar)**2
    key, key_z = random.split(key)
    z = random.normal(key_z, (len(true_data),))*SIGMA
    z = z-jnp.mean(z)+xbar
    key, key_mu = random.split(key)
    mu = random.normal(key_mu, (1,))*SIGMA/jnp.sqrt(len(true_data)) + xbar
    return z, mu, dist


def ABC_gauss(key, true_data, epsilon, N_ABC):
    keys = random.split(key, N_ABC+1)
    zs, mus, dists = vmap(jit(ABC_gauss_single), (0, None, None))(keys[1:], true_data, epsilon)
    return zs, mus, dists, keys[0]

def true_posterior_pdf(mus, TRUE_DATA):
    mu_post = (MU0 * SIGMA**2 + SIGMA0**2 * jnp.sum(TRUE_DATA)) / (
        SIGMA0**2 * len(TRUE_DATA) + SIGMA**2
    )
    sigma2_post = 1 / (1 / SIGMA0**2 + len(TRUE_DATA) / SIGMA**2)
    return norm.pdf(mus, loc=mu_post, scale=np.sqrt(sigma2_post))

def true_posterior_sample(key, TRUE_DATA, N_SAMPLE):
    mu_post = (MU0 * SIGMA**2 + SIGMA0**2 * jnp.sum(TRUE_DATA)) / (
        SIGMA0**2 * len(TRUE_DATA) + SIGMA**2
    )
    sigma2_post = 1 / (1 / SIGMA0**2 + len(TRUE_DATA) / SIGMA**2)
    return random.normal(key, (N_SAMPLE,)) * np.sqrt(sigma2_post) + mu_post

def get_dataset_gauss(key, n_points, prior_simulator, data_simulator, discrepancy, epsilon, true_data, index_marginal):
    n_points = n_points//2
    zs, thetas, dists, key = ABC_gauss(key, true_data, epsilon, n_points)
    thetas = thetas[:, index_marginal][:,None]
    key, key_perm = random.split(key)
    thetas_prime = thetas[random.permutation(key_perm, jnp.arange(n_points))]
    zs = jnp.concatenate([zs, zs], axis=0)
    thetas = jnp.concatenate([thetas, thetas_prime], axis=0)
    ys = jnp.append(jnp.zeros(n_points), jnp.ones(n_points)).astype(int)
    Xs = jnp.concatenate([thetas, zs], axis=1)
    return Xs, ys, dists, key




@jit
def prior_simulator(key):
    return random.normal(key, (1,)) * SIGMA0 + MU0


@jit
def data_simulator(key, theta):
    return (random.normal(key, (N_DATA,)) * SIGMA + theta).astype(float)


@jit
def discrepancy(y, y_true):
    return (jnp.mean(y) - jnp.mean(y_true)) ** 2




key = random.PRNGKey(0)

N_DATA = 100
N_KDE = 10000
N_POINTS = 1000000
N_SAMPLE = 10000
N_SAMPLES = 1
N_DATASETS = 10
N_EPOCHS = 100
N_GRID = 1000
ALPHAS = [1.0, .9, .5,.1,.05,.01,.005,.001, .0005, .0001]
# ALPHAS = [1., .1, .01, .001, .0001]
SIGMA = 1.0
MU0 = 0.
if len(sys.argv)>1:
    SIGMA0 = float(sys.argv[1])
    
print("SIGMA0 = ", SIGMA0)

PRIOR_DIST = norm(loc=MU0, scale=SIGMA0)
INDEX_MARGINAL = 0

MODEL_ARGS = {"SIGMA": SIGMA}
PRIOR_ARGS = {"MU0": MU0, "SIGMA0": SIGMA0}


PATH_RESULTS = (
    os.getcwd()
    + "/examples/Gauss-Gauss/Gauss_Gauss_1D_known_sigma/clean_results/sigma0_{}/".format(int(SIGMA0))
)
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


def ABC_NRE(
    key,
    N_POINTS,
    prior_simulator,
    data_simulator,
    discrepancy,
    TRUE_DATA,
    EPSILON,
    index_marginal=0,
):
    key, key_data = random.split(key)
    time_start = time.time()
    print("Simulation of the training dataset...")
    X, y, dists, key = get_dataset_gauss(
        key_data,
        N_POINTS,
        prior_simulator,
        data_simulator,
        discrepancy,
        EPSILON,
        TRUE_DATA,
        index_marginal,
    )

    key, key_split = random.split(key)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=np.random.RandomState(key_split)
    )
    time_simulations = time.time() - time_start
    print("Done in {} seconds!".format(time_simulations))
    N_POINTS_TRAIN = len(X_train)
    N_POINTS_TEST = len(X_test)

    print("Training the neural network...")
    params, train_accuracy, train_losses, test_accuracy, test_losses, key = train_loop(
        key,
        N_EPOCHS,
        NUM_LAYERS,
        HIDDEN_SIZE,
        NUM_CLASSES,
        BATCH_SIZE,
        NUM_BATCH,
        LEARNING_RATE,
        WDECAY,
        PATIENCE,
        COOLDOWN,
        FACTOR,
        RTOL,
        ACCUMULATION_SIZE,
        LEARNING_RATE_MIN,
        prior_simulator,
        data_simulator,
        discrepancy,
        true_data=TRUE_DATA,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        N_POINTS_TRAIN=N_POINTS_TRAIN,
        N_POINTS_TEST=N_POINTS_TEST,
        epsilon=EPSILON,
        verbose=True,
    )
    time_training = time.time() - time_simulations
    print("Done in {} seconds!".format(time_training))
    return (
        X,
        y,
        dists,
        params,
        train_accuracy,
        train_losses,
        test_accuracy,
        test_losses,
        key,
        time_simulations,
        time_training,
    )
    


def for_an_epsilon(
    i_dataset,
    alpha,
    key,
    N_POINTS,
    prior_simulator,
    data_simulator,
    discrepancy,
    TRUE_DATA,
    EPSILON,
    PRIOR_DIST,
    N_GRID,
    N_KDE,
    index_marginal=0,
):
    print("\n---------------------------------\nDATASET {} ALPHA {}\n---------------------------------".format(i_dataset, alpha))
    key, key_nre, key_kde, key_evaluate = random.split(key, 4)
    (
        X,
        y,
        dists,
        params,
        train_accuracy,
        train_losses,
        test_accuracy,
        test_losses,
        key,
        time_simulations,
        time_training,
    ) = ABC_NRE(
        key_nre, 
        N_POINTS,
        prior_simulator,
        data_simulator,
        discrepancy,
        TRUE_DATA,
        EPSILON,
        index_marginal,
    )
    THETAS_ABC = X[:, 0]
    
    print("Plotting the posterior...")
    plot_posterior_comparison(params, TRUE_DATA, THETAS_ABC, PRIOR_DIST, true_posterior_pdf=true_posterior_pdf, show = False, N_GRID = 10000, file_name = PATH_POSTERIORS + "{}_alpha_{}.png".format(i_dataset, alpha), N_KDE = 10000)
    
    
    METRICS_ABC_ij, METRICS_NRE_ij, METRICS_CORRECTED_NRE_ij, time_eval = evaluate_metrics(key_evaluate, TRUE_DATA, params, THETAS_ABC, PRIOR_DIST, N_GRID, N_SAMPLE, N_SAMPLES, true_posterior_sample, N_KDE)
    return (
        dists,
        params,
        train_accuracy,
        train_losses,
        test_accuracy,
        test_losses,
        time_simulations,
        time_training,
        time_eval,
        METRICS_ABC_ij,
        METRICS_NRE_ij,
        METRICS_CORRECTED_NRE_ij,
    )


def for_a_dataset(
    i_dataset,
    key,
    N_POINTS,
    prior_simulator,
    data_simulator,
    discrepancy,
    ALPHAS,
    PRIOR_DIST,
    index_marginal=0,
):  
    print("\n---------------------------------\nDATASET {}\n---------------------------------".format(i_dataset))
    PARAMS_i, TEST_ACCURACY_i, TRAIN_ACCURACY_i, TEST_LOSSES_i, TRAIN_LOSSES_i = (
        {},
        {},
        {},
        {},
        {},
    )
    TIME_SIMULATIONS_i, TIME_TRAINING_i, TIME_EVAL_i = {}, {}, {}
    METRICS_ABC_i, METRICS_NRE_i, METRICS_CORRECTED_NRE_i = {}, {}, {}

    key, key_theta, key_data = random.split(key, 3)
    TRUE_THETA = prior_simulator(key_theta)
    TRUE_DATA = data_simulator(key_data, TRUE_THETA)
    key, key_epsilon = random.split(key)
    time_iterations = {}
    EPSILONS_i = {1.0: np.inf}
    for alpha in ALPHAS:
        time_iteration = time.time()
        EPSILON = EPSILONS_i[alpha]
        key, key_epsilon = random.split(key_epsilon)
        (
            dists,
            params,
            train_accuracy,
            train_losses,
            test_accuracy,
            test_losses,
            time_simulations,
            time_training,
            time_eval,
            METRICS_ABC_ij,
            METRICS_NRE_ij,
            METRICS_CORRECTED_NRE_ij,
        ) = for_an_epsilon(
            i_dataset,
            alpha,
            key_epsilon,
            N_POINTS,
            prior_simulator,
            data_simulator,
            discrepancy,
            TRUE_DATA,
            EPSILON,
            PRIOR_DIST,
            N_GRID,
            N_KDE,
            index_marginal,
        )

        if alpha == 1:
            for alpha_not_1 in ALPHAS[1:]:
                EPSILONS_i[alpha_not_1] = float(jnp.quantile(dists, alpha_not_1))

        PARAMS_i[alpha] = params
        TEST_ACCURACY_i[alpha] = test_accuracy
        TRAIN_ACCURACY_i[alpha] = train_accuracy
        TEST_LOSSES_i[alpha] = test_losses
        TRAIN_LOSSES_i[alpha] = train_losses
        TIME_SIMULATIONS_i[alpha] = time_simulations
        TIME_TRAINING_i[alpha] = time_training
        TIME_EVAL_i[alpha] = time_eval
        METRICS_ABC_i[alpha] = METRICS_ABC_ij
        METRICS_NRE_i[alpha] = METRICS_NRE_ij
        METRICS_CORRECTED_NRE_i[alpha] = METRICS_CORRECTED_NRE_ij
        time_iterations[alpha] = time.time() - time_iteration
        
    plot_metric_for_a_dataset("C2ST", ALPHAS, METRICS_ABC_i, METRICS_NRE_i, METRICS_CORRECTED_NRE_i, N_SAMPLES, PATH_FIGURES + "c2st_{}.png".format(i_dataset), show = False, title = "$\\theta  =s {:.3}".format(float(TRUE_THETA[index_marginal])))
    plot_metric_for_a_dataset("RS_stat", ALPHAS, METRICS_ABC_i, METRICS_NRE_i, METRICS_CORRECTED_NRE_i, N_SAMPLES, PATH_FIGURES + "ranksums_{}.png".format(i_dataset), show = False, title = "$\\theta  =s {:.3}".format(float(TRUE_THETA[index_marginal])))
    create_csv_for_a_dataset(i_dataset, ALPHAS, TEST_ACCURACY_i, TRAIN_ACCURACY_i, TEST_LOSSES_i, TRAIN_LOSSES_i, TIME_SIMULATIONS_i, TIME_TRAINING_i, TIME_EVAL_i, METRICS_ABC_i, METRICS_NRE_i, METRICS_CORRECTED_NRE_i, TRUE_DATA, TRUE_THETA, PATH_CSV + "{}_theta_{:.3}.csv".format(i_dataset,float(TRUE_THETA[index_marginal])))
    create_pickle_for_a_dataset(ALPHAS, PARAMS_i, METRICS_ABC_i, METRICS_NRE_i, METRICS_CORRECTED_NRE_i, TRUE_DATA, TRUE_THETA, TIME_SIMULATIONS_i, TIME_TRAINING_i, TIME_EVAL_i, MODEL_ARGS, PRIOR_ARGS, PATH_PICKLES + "{}_theta_{:.3}.pkl".format(i_dataset,float(TRUE_THETA[index_marginal])))
    return (
        PARAMS_i,
        TEST_ACCURACY_i,
        TRAIN_ACCURACY_i,
        TEST_LOSSES_i,
        TRAIN_LOSSES_i,
        TIME_SIMULATIONS_i,
        TIME_TRAINING_i,
        TIME_EVAL_i,
        METRICS_ABC_i,
        METRICS_NRE_i,
        METRICS_CORRECTED_NRE_i,
        TRUE_DATA,
        TRUE_THETA,
    )


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
    ) = for_a_dataset(
        i_dataset,
        key_i,
        N_POINTS,
        prior_simulator,
        data_simulator,
        discrepancy,
        ALPHAS,
        PRIOR_DIST,
        INDEX_MARGINAL,
    )

plot_metric_for_many_datasets("C2ST", ALPHAS, METRICS_ABC, METRICS_NRE, METRICS_CORRECTED_NRE, N_SAMPLES, N_DATASETS, PATH_FIGURES + "c2st.png", show = False, title = "For 10 differents $\\theta$")
plot_metric_for_many_datasets("RS_stat", ALPHAS, METRICS_ABC, METRICS_NRE, METRICS_CORRECTED_NRE, N_SAMPLES, N_DATASETS, PATH_FIGURES + "ranksums.png", show = False, title = "For 10 differents $\\theta$")
    