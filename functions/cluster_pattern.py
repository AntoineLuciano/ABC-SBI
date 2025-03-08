from functions.simulation import get_dataset, ABC_epsilon, NRE_corrected_posterior_sample, NRE_posterior_sample
from function.training import train_loop
from jax import random, jit
import os
import sys
from sklearn.model_selection import train_test_split
import time 
from scipy.stats import gaussian_kde, ranksums
import numpy as np
import jax.numpy as jnp
import torch
import pickle
from sbibm.metrics import c2st
@jit 
def prior_simulator(key):
    return ...

@jit
def data_simulator(key, theta):
    return ...

@jit
def discrepancy(data, true_data):
    return ...

def true_posterior_sample(key):
    return ...

key = random.PRNGKey(0)

N_KDE = 1000
N_POINTS = 1000000
N_SAMPLE = 100000
N_SAMPLES = 1
N_EPSILON = 1000
N_DATASETS = 2
N_EPOCHS = 1   
N_GRID = 10000
ALPHAS = [1., .99]

#HYPERPARAMETERS PRIOR AND DATA SIMULATOR
...

PRIOR_DIST = ...



PATH_RESULTS = os.getcwd() + "/examples/..."
PATH_FIGURES = PATH_RESULTS + "figures/"
PATH_PICKLES = PATH_RESULTS + "pickles/"
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)
if not os.path.exists(PATH_FIGURES):
    os.makedirs(PATH_FIGURES)
if not os.path.exists(PATH_PICKLES):
    os.makedirs(PATH_PICKLES)
    
    
LEARNING_RATE = 0.001
PATIENCE = 7
COOLDOWN = 0
FACTOR = .5
RTOL = 1e-4  
ACCUMULATION_SIZE = 200
LEARNING_RATE_MIN = 1e-6

BATCH_SIZE = 256
NUM_BATCH = 1024
NUM_CLASSES = 2
HIDDEN_SIZE = 256
NUM_LAYERS = 2
WDECAY = .001


def ABC_NRE(key, N_POINTS, prior_simulator, data_simulator, discrepancy, TRUE_DATA, EPSILON, index_marginal = 0):
    key, key_data = random.split(key)
    time_start = time.time()
    X,y, dists, key = get_dataset(key_data, N_POINTS, prior_simulator, data_simulator, discrepancy, EPSILON, TRUE_DATA, index_marginal)

    
    key, key_split = random.split(key)  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=key_split)
    time_simulations = time.time() - time_start
    
    params, train_accuracy, train_losses, test_accuracy, test_losses, key = train_loop(key, N_EPOCHS, NUM_LAYERS, HIDDEN_SIZE, NUM_CLASSES, BATCH_SIZE, NUM_BATCH, LEARNING_RATE, WDECAY, PATIENCE, COOLDOWN, FACTOR, RTOL, ACCUMULATION_SIZE, LEARNING_RATE_MIN, prior_simulator, data_simulator, discrepancy, true_data = TRUE_DATA, X_train = X_train, y_train = y_train, X_test = X_test, y_test =  y_test, N_POINTS_TRAIN = N_POINTS_TRAIN, N_POINTS_TEST = N_POINTS_TEST, epsilon = EPSILON, verbose = True)
    time_training = time.time() - time_simulations
    
    return X, y, dists, params, train_accuracy, train_losses, test_accuracy, test_losses, key, time_simulations, time_training


def evaluate_metrics(key, TRUE_DATA, params, X, PRIOR_DIST, N_GRID, N_SAMPLE, N_SAMPLES, N_KDE):
    
    METRICS_ABC_ij, METRICS_NRE_ij, METRICS_CORRECTED_NRE_ij = {}, {}, {}
    time_start = time.time()   
    PRIOR_LOGPDF = lambda theta: PRIOR_DIST.logpdf(theta)
    KDE_APPROX = gaussian_kde(X[:N_KDE,0])
    MIN_INIT, MAX_INIT = PRIOR_DIST.interval(0.999)
    
    for j in range(N_SAMPLES):
        key, key_true, key_nre, key_corrected_nre, key_abc = random.split(key, 5)
        sample_true = true_posterior_sample(key_true, TRUE_DATA, N_SAMPLE)
        sample_nre = NRE_posterior_sample(key_nre, params, TRUE_DATA, PRIOR_LOGPDF, N_GRID, MIN_INIT, MAX_INIT, N_SAMPLE)
        sample_corrected_nre = NRE_corrected_posterior_sample(key_corrected_nre, params, TRUE_DATA, KDE_APPROX, N_GRID, MIN_INIT, MAX_INIT, N_SAMPLE)
        sample_abc = X[random.choice(key_abc, np.arange(len(X)), (N_SAMPLE)),0]
        if np.isnan(sample_abc).any(): 
            METRICS_ABC_ij["C2ST"][j] = 1.
            METRICS_ABC_ij["RS_pvalue"][j] = 0.
            METRICS_ABC_ij["RS_stat"][j]
        else:
            METRICS_ABC_ij["C2ST"][j] = c2st(torch.tensor(sample_true)[:,None], torch.tensor(sample_abc)[:,None])
            METRICS_ABC_ij["RS_stat"][j], METRICS_ABC_ij["RS_pvalue"][j] = ranksums(sample_true, sample_abc)
        
        if np.isnan(sample_nre).any():
            METRICS_NRE_ij["C2ST"][j] = 1.
            METRICS_NRE_ij["RS_pvalue"][j] = 0.
            METRICS_NRE_ij["RS_stat"][j]
        else:
            METRICS_NRE_ij["C2ST"][j] = c2st(torch.tensor(sample_true)[:,None], torch.tensor(sample_nre)[:,None])
            METRICS_NRE_ij["RS_stat"][j], METRICS_NRE_ij["RS_pvalue"][j] = ranksums(sample_true, sample_nre)
        
        if np.isnan(sample_corrected_nre).any():
            METRICS_CORRECTED_NRE_ij["C2ST"][j] = 1.
            METRICS_CORRECTED_NRE_ij["RS_pvalue"][j] = 0.
            METRICS_CORRECTED_NRE_ij["RS_stat"][j]
        else:
            METRICS_CORRECTED_NRE_ij["C2ST"][j] = c2st(torch.tensor(sample_true)[:,None], torch.tensor(sample_corrected_nre)[:,None])
            METRICS_CORRECTED_NRE_ij["RS_stat"][j], METRICS_CORRECTED_NRE_ij["RS_pvalue"][j] = ranksums(sample_true, sample_corrected_nre)
    return METRICS_ABC_ij, METRICS_NRE_ij, METRICS_CORRECTED_NRE_ij, time.time() - time_start


def create_csv_for_a_dataset(i_datasets,TEST_ACCURACY, TRAIN_ACCURACY, TEST_LOSSES, TRAIN_LOSSES, TIME_SIMULATIONS, TIME_TRAINING, TIME_EVAL, METRICS_ABC, METRICS_NRE, METRICS_CORRECTED_NRE, TRUE_DATA, TRUE_THETA):
        df = pd.DataFrame()
        df["ALPHAS"] = ALPHAS
        
        df["TRUE_DATA"] = TRUE_DATA
        df["TRUE_THETA"] = TRUE_THETA
        df["TEST_ACCURACY"] = np.array([TEST_ACCURACY[a] for a in ALPHAS])
        df["TRAIN_ACCURACY"] = np.array([TRAIN_ACCURACY[a] for a in ALPHAS])
        df["TEST_LOSSES"] = np.array([TEST_LOSSES[a] for a in ALPHAS])
        df["TRAIN_LOSSES"] = np.array([TRAIN_LOSSES[a] for a in ALPHAS])
        df["TIME_SIMULATIONS"] = np.array([TIME_SIMULATIONS[a] for a in ALPHAS])
        df["TIME_TRAINING"] = np.array([TIME_TRAINING[a] for a in ALPHAS])
        df["TIME_EVAL"] = np.array([TIME_EVAL[a] for a in ALPHAS])
        df["RANKSUMS_STAT_ABC"] = np.array([METRICS_ABC[a]["RS_stat"] for a in ALPHAS])
        df["RANKSUMS_PVALUE_ABC"] = np.array([METRICS_ABC[a]["RS_pvalue"] for a in ALPHAS])
        df["C2ST_ABC"] = np.array([METRICS_ABC[a]["C2ST"] for a in ALPHAS])
        df["RANKSUMS_STAT_NRE"] = np.array([METRICS_NRE[a]["RS_stat"] for a in ALPHAS])
        df["RANKSUMS_PVALUE_NRE"] = np.array([METRICS_NRE[a]["RS_pvalue"] for a in ALPHAS])
        df["C2ST_NRE"] = np.array([METRICS_NRE[a]["C2ST"] for a in ALPHAS])
        df["RANKSUMS_STAT_CORRECTED_NRE"] = np.array([METRICS_CORRECTED_NRE[a]["RS_stat"] for a in ALPHAS])
        df["RANKSUMS_PVALUE_CORRECTED_NRE"] = np.array([METRICS_CORRECTED_NRE[a]["RS_pvalue"] for a in ALPHAS])
        df["C2ST_CORRECTED_NRE"] = np.array([METRICS_CORRECTED_NRE[a]["C2ST"] for a in ALPHAS])
        df.to_csv(PATH_RESULTS + "{}_results.csv".format(i_datasets))
        print("CSV CREATED at {}".format(PATH_RESULTS + "{}_results.csv".format(i_datasets)))
        
        
        

def for_an_epsilon(key, N_POINTS, prior_simulator, data_simulator, discrepancy, TRUE_DATA, EPSILON, PRIOR_DIST, N_GRID, N_KDE, index_marginal = 0):
    X, y, dists, params, train_accuracy, train_losses, test_accuracy, test_losses, key, time_simulations, time_training = ABC_NRE(key, N_POINTS, prior_simulator, data_simulator, discrepancy, TRUE_DATA, EPSILON, index_marginal)
    METRICS_ABC_ij, METRICS_NRE_ij, METRICS_CORRECTED_NRE_ij, time_eval = evaluate_metrics(key, TRUE_DATA, params, X, PRIOR_DIST, N_GRID, N_SAMPLE, N_SAMPLES, N_KDE)
    return dists, params, train_accuracy, train_losses, test_accuracy, test_losses, time_simulations, time_training, time_eval, METRICS_ABC_ij, METRICS_NRE_ij, METRICS_CORRECTED_NRE_ij
        

def for_a_dataset(key, N_POINTS, prior_simulator, data_simulator, discrepancy, TRUE_DATA, ALPHAS, PRIOR_DIST, index_marginal = 0):
    PARAMS_i, TEST_ACCURACY_i, TRAIN_ACCURACY_i, TEST_LOSSES_i, TRAIN_LOSSES_i = {}, {}, {}, {}, {}
    TIME_SIMULATIONS_i, TIME_TRAINING_i, TIME_EVAL_i = {}, {}, {}
    METRICS_ABC_i, METRICS_NRE_i, METRICS_CORRECTED_NRE_i = {}, {}, {}
    
    key, key_theta, key_data = random.split(key, 3)
    TRUE_THETA = prior_simulator(key_theta)
    TRUE_DATA = data_simulator(key_data, TRUE_THETA)
    key, key_epsilon = random.split(key)
    time_iterations = {}
    EPSILONS_i = {1.: np.inf}
    for alpha in ALPHAS:
        time_iteration = time.time()
        EPSILON = EPSILONS_i[alpha]
        key, key_epsilon = random.split(key_epsilon)
        dists, params, train_accuracy, train_losses, test_accuracy, test_losses, time_simulations, time_training, time_eval, METRICS_ABC_ij, METRICS_NRE_ij, METRICS_CORRECTED_NRE_ij = for_an_epsilon(key_epsilon, N_POINTS, prior_simulator, data_simulator, discrepancy, TRUE_DATA, EPSILON, PRIOR_DIST, N_GRID, N_KDE, index_marginal)
        
        if alpha == 1:
            for alpha_not_1 in ALPHAS[1:]:
                EPSILONS_i[alpha_not_1] = jnp.quantile(dists, alpha_not_1)
        
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
        METRICS_CORRECTED_NRE_i[alpha] = METRICS_ABC_ij
        
        time_iterations[alpha] = time.time() - time_iteration
        
    
    return PARAMS_i, TEST_ACCURACY_i, TRAIN_ACCURACY_i, TEST_LOSSES_i, TRAIN_LOSSES_i, TIME_SIMULATIONS_i, TIME_TRAINING_i, TIME_EVAL_i, METRICS_ABC_i, METRICS_NRE_i, METRICS_CORRECTED_NRE_i, TRUE_DATA, TRUE_THETA



            
        
    



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

for i_dataset in range(N_DATASETS):
    key, key_i = random.split(key)
    PARAMS[i_dataset], TEST_ACCURACY[i_dataset], TRAIN_ACCURACY[i_dataset], TEST_LOSSES[i_dataset], TRAIN_LOSSES[i_dataset], TIME_SIMULATIONS[i_dataset], TIME_TRAINING[i_dataset], TIME_SAMPLING[i_dataset], METRICS_ABC[i_dataset], METRICS_NRE[i_dataset], METRICS_CORRECTED_NRE[i_dataset], TRUE_DATAS[i_dataset], TRUE_THETAS[i_dataset] = for_a_dataset(key_i, N_POINTS, prior_simulator, data_simulator, discrepancy, TRUE_DATA, ALPHAS, PRIOR_DIST, PRIOR_LOGPDF, N_GRID, N_KDE, index_marginal = 0)
    create_csv_for_a_dataset(i_dataset, TEST_ACCURACY[i_dataset], TRAIN_ACCURACY[i_dataset], TEST_LOSSES[i_dataset], TRAIN_LOSSES[i_dataset], TIME_SIMULATIONS[i_dataset], TIME_TRAINING[i_dataset], TIME_SAMPLING[i_dataset], METRICS_ABC[i_dataset], METRICS_NRE[i_dataset], METRICS_CORRECTED_NRE[i_dataset], TRUE_DATAS[i_dataset], TRUE_THETAS[i_dataset])