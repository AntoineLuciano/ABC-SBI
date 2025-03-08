
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
from jax import random, jit
from sklearn.model_selection import train_test_split
import time
from scipy.stats import gaussian_kde, ranksums, norm
import numpy as np
import jax.numpy as jnp
import torch
import pickle
from sbibm.metrics import c2st
import pandas as pd
import matplotlib.pyplot as plt

@jit
def prior_simulator(key):
    return random.normal(key, (1,)) * SIGMA0 + MU0


@jit
def data_simulator(key, theta):
    return (random.normal(key, (N_DATA,)) * SIGMA + theta).astype(float)


@jit
def discrepancy(y, y_true):
    return (jnp.mean(y) - jnp.mean(y_true)) ** 2


def true_posterior_sample(key, TRUE_DATA, N_SAMPLE):
    mu_post = (MU0 * SIGMA**2 + SIGMA0**2 * jnp.sum(TRUE_DATA)) / (
        SIGMA0**2 * len(TRUE_DATA) + SIGMA**2
    )
    sigma2_post = 1 / (1 / SIGMA0**2 + len(TRUE_DATA) / SIGMA**2)
    return random.normal(key, (N_SAMPLE,)) * np.sqrt(sigma2_post) + mu_post


key = random.PRNGKey(0)

N_DATA = 100
N_KDE = 10000
N_POINTS = 500000
N_SAMPLE = 10000
N_SAMPLES = 1
N_DATASETS = 10
N_EPOCHS = 100
N_GRID = 10000
ALPHAS = [1.0, .9, .5,.1,.05,.01,.005,.001] 

SIGMA = 1.0
MU0, SIGMA0 = 0.0, 10.0

PRIOR_DIST = norm(loc=MU0, scale=SIGMA0)
INDEX_MARGINAL = 0

MODEL_ARGS = {"SIGMA": SIGMA}
PRIOR_ARGS = {"MU0": MU0, "SIGMA0": SIGMA0}


PATH_RESULTS = (
    os.getcwd()
    + "/examples/Gauss-Gauss/Gauss_Gauss_1D_known_sigma/cluster_pattern_gauss/"
)
PATH_FIGURES = PATH_RESULTS + "figures/"
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
    X, y, dists, key = get_dataset(
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


def evaluate_metrics(
    key, TRUE_DATA, params, X, PRIOR_DIST, N_GRID, N_SAMPLE, N_SAMPLES, N_KDE
):
    print("Evaluation of the metrics...")
    METRICS_ABC_ij, METRICS_NRE_ij, METRICS_CORRECTED_NRE_ij = {"C2ST": np.zeros(N_SAMPLES), "RS_stat": np.zeros(N_SAMPLES), "RS_pvalue": np.zeros(N_SAMPLES)}, {"C2ST": np.zeros(N_SAMPLES), "RS_stat": np.zeros(N_SAMPLES), "RS_pvalue": np.zeros(N_SAMPLES)}, {"C2ST": np.zeros(N_SAMPLES), "RS_stat": np.zeros(N_SAMPLES), "RS_pvalue": np.zeros(N_SAMPLES)}
    time_start = time.time()
    PRIOR_LOGPDF = lambda theta: PRIOR_DIST.logpdf(theta)
    key, key_kde = random.split(key)
    KDE_APPROX = gaussian_kde(X[random.choice(key_kde, np.arange(len(X)), (N_KDE,)), 0])
    MIN_INIT, MAX_INIT = PRIOR_DIST.interval(0.999)

    for j in range(N_SAMPLES):
        key, key_true, key_nre, key_corrected_nre, key_abc = random.split(key, 5)
        sample_true = true_posterior_sample(key_true, TRUE_DATA, N_SAMPLE)
        sample_nre = NRE_posterior_sample(
            key_nre,
            params,
            TRUE_DATA,
            PRIOR_LOGPDF,
            N_GRID,
            MIN_INIT,
            MAX_INIT,
            N_SAMPLE,
        )
        sample_corrected_nre = NRE_corrected_posterior_sample(
            key_corrected_nre,
            params,
            TRUE_DATA,
            KDE_APPROX,
            N_GRID,
            MIN_INIT,
            MAX_INIT,
            N_SAMPLE,
        )
        sample_abc = X[random.choice(key_abc, np.arange(len(X)), (N_SAMPLE,)), 0]
        if np.isnan(sample_abc).any():
            METRICS_ABC_ij["C2ST"][j] = 1.0
            METRICS_ABC_ij["RS_pvalue"][j] = 0.0
            METRICS_ABC_ij["RS_stat"][j]
        else:
            METRICS_ABC_ij["C2ST"][j] = c2st(
                torch.tensor(sample_true)[:, None], torch.tensor(sample_abc)[:, None]
            )
            METRICS_ABC_ij["RS_stat"][j], METRICS_ABC_ij["RS_pvalue"][j] = ranksums(
                sample_true, sample_abc
            )

        if np.isnan(sample_nre).any():
            METRICS_NRE_ij["C2ST"][j] = 1.0
            METRICS_NRE_ij["RS_pvalue"][j] = 0.0
            METRICS_NRE_ij["RS_stat"][j]
        else:
            METRICS_NRE_ij["C2ST"][j] = c2st(
                torch.tensor(sample_true)[:, None], torch.tensor(sample_nre)[:, None]
            )
            METRICS_NRE_ij["RS_stat"][j], METRICS_NRE_ij["RS_pvalue"][j] = ranksums(
                sample_true, sample_nre
            )

        if np.isnan(sample_corrected_nre).any():
            METRICS_CORRECTED_NRE_ij["C2ST"][j] = 1.0
            METRICS_CORRECTED_NRE_ij["RS_pvalue"][j] = 0.0
            METRICS_CORRECTED_NRE_ij["RS_stat"][j]
        else:
            METRICS_CORRECTED_NRE_ij["C2ST"][j] = c2st(
                torch.tensor(sample_true)[:, None],
                torch.tensor(sample_corrected_nre)[:, None],
            )
            (
                METRICS_CORRECTED_NRE_ij["RS_stat"][j],
                METRICS_CORRECTED_NRE_ij["RS_pvalue"][j],
            ) = ranksums(sample_true, sample_corrected_nre)
    return (
        METRICS_ABC_ij,
        METRICS_NRE_ij,
        METRICS_CORRECTED_NRE_ij,
        time.time() - time_start,
    )


def create_csv_for_a_dataset(
    i_datasets,
    TEST_ACCURACY,
    TRAIN_ACCURACY,
    TEST_LOSSES,
    TRAIN_LOSSES,
    TIME_SIMULATIONS,
    TIME_TRAINING,
    TIME_EVAL,
    METRICS_ABC,
    METRICS_NRE,
    METRICS_CORRECTED_NRE,
    TRUE_DATA,
    TRUE_THETA,
):
    df = pd.DataFrame()
    df["ALPHA"] = ALPHAS
    # df["TRUE_DATA"] = [TRUE_DATA] * len(ALPHAS)
    df["TRUE_THETA"] = [TRUE_THETA] * len(ALPHAS)
    df["TEST_ACCURACY"] = [TEST_ACCURACY[a] for a in ALPHAS]
    df["TRAIN_ACCURACY"] = [TRAIN_ACCURACY[a] for a in ALPHAS]
    df["TEST_LOSSES"] = [TEST_LOSSES[a] for a in ALPHAS]
    df["TRAIN_LOSSES"] = [TRAIN_LOSSES[a] for a in ALPHAS]
    df["TIME_SIMULATIONS"] = [TIME_SIMULATIONS[a] for a in ALPHAS]
    df["TIME_TRAINING"] = [TIME_TRAINING[a] for a in ALPHAS]
    df["TIME_EVAL"] = [TIME_EVAL[a] for a in ALPHAS]
    df["RANKSUMS_STAT_ABC"] = [METRICS_ABC[a]["RS_stat"] for a in ALPHAS]
    df["RANKSUMS_PVALUE_ABC"] = [METRICS_ABC[a]["RS_pvalue"] for a in ALPHAS]
    df["C2ST_ABC"] = [METRICS_ABC[a]["C2ST"] for a in ALPHAS]
    df["RANKSUMS_STAT_NRE"] = [METRICS_NRE[a]["RS_stat"] for a in ALPHAS]
    df["RANKSUMS_PVALUE_NRE"] = [METRICS_NRE[a]["RS_pvalue"] for a in ALPHAS]
    df["C2ST_NRE"] = [METRICS_NRE[a]["C2ST"] for a in ALPHAS]
    df["RANKSUMS_STAT_CORRECTED_NRE"] = [METRICS_CORRECTED_NRE[a]["RS_stat"] for a in ALPHAS]
    df["RANKSUMS_PVALUE_CORRECTED_NRE"] = [METRICS_CORRECTED_NRE[a]["RS_pvalue"] for a in ALPHAS]
    df["C2ST_CORRECTED_NRE"] = [METRICS_CORRECTED_NRE[a]["C2ST"] for a in ALPHAS]
    df.to_csv(PATH_CSV + "/{}_metrics.csv".format(i_datasets))
    print("CSV CREATED at {}".format(PATH_CSV + "/{}_metrics.csv".format(i_datasets)))

def create_pickle_for_a_dataset(i_dataset, PARAMS, METRICS_ABC, METRICS_NRE, METRICS_CORRECTED_NRE, TRUE_DATA, TRUE_THETA, TIME_SIMULATIONS, TIME_TRAINING, TIME_EVAL, MODEL_ARGS, PRIOR_ARGS):
    dico = {
        "PARAMS": PARAMS,
        "METRICS_ABC": METRICS_ABC,
        "METRICS_NRE": METRICS_NRE,
        "METRICS_CORRECTED_NRE": METRICS_CORRECTED_NRE,
        "TRUE_DATA": TRUE_DATA,
        "TRUE_THETA": TRUE_THETA,
        "TIME_SIMULATIONS": TIME_SIMULATIONS,
        "TIME_TRAINING": TIME_TRAINING,
        "TIME_EVAL": TIME_EVAL,
        "MODEL_ARGS": MODEL_ARGS,
        "PRIOR_ARGS": PRIOR_ARGS,
        }
    with open(PATH_PICKLES + "/{}_dico.pkl".format(i_dataset), "wb") as f:
        pickle.dump(dico, f)
    print("PICKLE CREATED at {}".format(PATH_PICKLES + "/{}_dico.pkl".format(i_dataset)))
          
    
    
def plot_c2st_for_a_dataset(i_dataset, METRICS_ABC, METRICS_NRE, METRICS_CORRECTED_NRE):
    abc = np.array([[METRICS_ABC[a]["C2ST"] for j in range(N_SAMPLES)] for a in ALPHAS]).reshape(len(ALPHAS), -1)
    nre = np.array([[METRICS_NRE[a]["C2ST"] for j in range(N_SAMPLES)] for a in ALPHAS]).reshape(len(ALPHAS), -1)
    corrected_nre = np.array([[METRICS_CORRECTED_NRE[a]["C2ST"] for j in range(N_SAMPLES)] for a in ALPHAS]).reshape(len(ALPHAS), -1)
    
    mean_abc = np.mean(abc, axis=1)
    std_abc = np.std(abc, axis=1)
    mean_nre = np.mean(nre, axis=1)
    std_nre = np.std(nre, axis=1)
    mean_corrected_nre = np.mean(corrected_nre, axis=1)
    std_corrected_nre = np.std(corrected_nre, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.errorbar(ALPHAS, mean_abc, yerr=std_abc, label="ABC", fmt="o-")
    ax.errorbar(ALPHAS, mean_nre, yerr=std_nre, label="NRE", fmt="o-")
    ax.errorbar(ALPHAS, mean_corrected_nre, yerr=std_corrected_nre, label="ABC-corrected NRE", fmt="o-")
    ax.set_xlabel("$\\alpha$")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylabel("C2ST")
    ax.legend()
    fig.savefig(PATH_FIGURES + "/{}_c2st.png".format(i_dataset))
    plt.close(fig)
    print("FIGURE CREATED at {}".format(PATH_FIGURES + "/{}_c2st.png".format(i_dataset)))
          
def plot_ranksums_for_a_dataset(i_dataset, METRICS_ABC, METRICS_NRE, METRICS_CORRECTED_NRE):
    abc = np.array([[METRICS_ABC[a]["RS_stat"] for j in range(N_SAMPLES)] for a in ALPHAS]).reshape(len(ALPHAS), -1)
    nre = np.array([[METRICS_NRE[a]["RS_stat"] for j in range(N_SAMPLES)] for a in ALPHAS]).reshape(len(ALPHAS), -1)
    corrected_nre = np.array([[METRICS_CORRECTED_NRE[a]["RS_stat"] for j in range(N_SAMPLES)] for a in ALPHAS]).reshape(len(ALPHAS), -1)
    
    mean_abc = np.mean(abc, axis=1)
    std_abc = np.std(abc, axis=1)
    mean_nre = np.mean(nre, axis=1)
    std_nre = np.std(nre, axis=1)
    mean_corrected_nre = np.mean(corrected_nre, axis=1)
    std_corrected_nre = np.std(corrected_nre, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.errorbar(ALPHAS, mean_abc, yerr=std_abc, label="ABC", fmt="o-")
    ax.errorbar(ALPHAS, mean_nre, yerr=std_nre, label="NRE", fmt="o-")
    ax.errorbar(ALPHAS, mean_corrected_nre, yerr=std_corrected_nre, label="ABC-corrected NRE", fmt="o-")
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("Ranksums statistic")
    ax.legend()
    ax.set_xscale("log")
    ax.invert_xaxis()
    fig.savefig(PATH_FIGURES + "/{}_ranksums.png".format(i_dataset))
    plt.close(fig)
    print("FIGURE CREATED at {}".format(PATH_FIGURES + "/{}_ranksums.png".format(i_dataset)))
    


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
        key,
        N_POINTS,
        prior_simulator,
        data_simulator,
        discrepancy,
        TRUE_DATA,
        EPSILON,
        index_marginal,
    )
    METRICS_ABC_ij, METRICS_NRE_ij, METRICS_CORRECTED_NRE_ij, time_eval = (
        evaluate_metrics(
            key, TRUE_DATA, params, X, PRIOR_DIST, N_GRID, N_SAMPLE, N_SAMPLES, N_KDE
        )
    )
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
TIME_SAMPLING = {}
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
        TIME_SAMPLING[i_dataset],
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
    create_csv_for_a_dataset(
        i_dataset,
        TEST_ACCURACY[i_dataset],
        TRAIN_ACCURACY[i_dataset],
        TEST_LOSSES[i_dataset],
        TRAIN_LOSSES[i_dataset],
        TIME_SIMULATIONS[i_dataset],
        TIME_TRAINING[i_dataset],
        TIME_SAMPLING[i_dataset],
        METRICS_ABC[i_dataset],
        METRICS_NRE[i_dataset],
        METRICS_CORRECTED_NRE[i_dataset],
        TRUE_DATAS[i_dataset],
        TRUE_THETAS[i_dataset],
    )
    plot_c2st_for_a_dataset(i_dataset, METRICS_ABC[i_dataset], METRICS_NRE[i_dataset], METRICS_CORRECTED_NRE[i_dataset])
    plot_ranksums_for_a_dataset(i_dataset, METRICS_ABC[i_dataset], METRICS_NRE[i_dataset], METRICS_CORRECTED_NRE[i_dataset])
    
