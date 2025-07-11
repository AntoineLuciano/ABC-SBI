import os
import sys
# path = os.getcwd()
# print("Old path:", path)
# path = path.split("/")
# path = path[: path.index("ABC-SBI") + 1]
# path = "/".join(path)
# print("New path:", path)
# os.chdir(path)
# sys.path.append(path)
from .simulation import get_dataset
from .training import train_loop
from .plots import plot_metric_for_a_dataset, plot_posterior_comparison
from .metrics import evaluate_metrics
from .save import create_csv_for_a_dataset
from jax import random, jit, vmap
from sklearn.model_selection import train_test_split
import time
from scipy.stats import norm
import numpy as np
import jax.numpy as jnp
import lzma
import pickle


def ABC_NRE(
    key,
    N_POINTS,
    prior_simulator,
    data_simulator,
    discrepancy,
    TRUE_DATA,
    EPSILON,
    NN_ARGS, 
    index_marginal=0,
):
    key, key_data = random.split(key)
    time_start = time.time()
    print("Simulation of the training dataset...")
    X, y, dists, key = get_dataset(
        key = key_data, n_points = N_POINTS, prior_simulator= prior_simulator, data_simulator= data_simulator, 
        discrepancy=discrepancy, epsilon= EPSILON, true_data= TRUE_DATA, index_marginal= index_marginal)

    key, key_split = random.split(key)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=np.random.RandomState(key_split)
    )
    time_simulations = time.time() - time_start
    
    print("Done in {} seconds.".format(time_simulations))
    N_POINTS_TRAIN = len(X_train)
    N_POINTS_TEST = len(X_test)

    print("Training the neural network...")
    time_start = time.time()
    params, train_accuracy, train_losses, test_accuracy, test_losses, key = \
        train_loop(key = key, NN_ARGS = NN_ARGS, prior_simulator= prior_simulator, data_simulator= data_simulator, discrepancy=discrepancy, true_data=TRUE_DATA, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, N_POINTS_TRAIN=N_POINTS_TRAIN, N_POINTS_TEST=N_POINTS_TEST, epsilon=EPSILON, verbose=True)
    
    time_training = time.time() - time_start
    print("Done in {} seconds!".format(time_training))
    return (X[:,0], dists,params,train_accuracy,train_losses,test_accuracy,test_losses,time_simulations,time_training)
        
    


def for_an_epsilon(
    i_dataset,
    alpha,
    key,
    N_POINTS,
    prior_simulator,
    data_simulator,
    discrepancy,
    true_posterior_pdf, 
    true_posterior_sample, 
    TRUE_DATA,
    EPSILON,
    PRIOR_DIST,
    NN_ARGS,
    N_GRID,
    N_KDE,
    N_SAMPLE,
    N_SAMPLES,
    METRICS_TO_TEST, 
    PATH, 
    index_marginal=0,
):
    print("\n---------------------------------\nDATASET {} ALPHA {}\n---------------------------------".format(i_dataset, alpha))
    key, key_nre, key_kde, key_evaluate = random.split(key, 4)
    
    thetas_abc, dists, params, train_accuracy, train_losses, test_accuracy, test_losses, time_simulations, time_training = \
        ABC_NRE(key= key_nre, N_POINTS= N_POINTS, prior_simulator= prior_simulator, data_simulator= data_simulator, 
                discrepancy=discrepancy, TRUE_DATA= TRUE_DATA, EPSILON= EPSILON, NN_ARGS= NN_ARGS, index_marginal=  index_marginal)
    
    print("Plotting the posterior...")
    plot_posterior_comparison(
        params= params, TRUE_DATA= TRUE_DATA, thetas_abc= thetas_abc, prior_dist= PRIOR_DIST, 
        file_name = PATH+"figures/posterior_check/{}_alpha_{}.png".format(i_dataset+1, alpha), 
        show = False, N_GRID = N_GRID, true_posterior_pdf = true_posterior_pdf, N_KDE= N_KDE)
    
    metrics, time_eval = evaluate_metrics(
        key = key_evaluate, metrics_dico= METRICS_TO_TEST, TRUE_DATA= TRUE_DATA, params= params, 
        thetas_abc= thetas_abc, PRIOR_DIST= PRIOR_DIST, N_GRID= N_GRID, N_SAMPLE= N_SAMPLE, 
        N_SAMPLES= N_SAMPLES, true_posterior_sample= true_posterior_sample, N_KDE= N_KDE)
    
    return (dists,params,train_accuracy,train_losses,test_accuracy,test_losses,time_simulations,time_training,time_eval,metrics, thetas_abc)


def for_a_dataset(
    i_dataset,
    key,
    N_POINTS,
    prior_simulator,
    data_simulator,
    discrepancy,
    true_posterior_pdf,
    true_posterior_sample,
    ALPHAS,
    PRIOR_DIST,
    PRIOR_ARGS, 
    MODEL_ARGS,
    NN_ARGS,
    N_GRID,
    N_KDE,
    N_SAMPLE,
    N_SAMPLES,
    METRICS_TO_TEST,
    PATH,
    index_marginal=0,
):  
    print("\n---------------------------------\nDATASET {}\n---------------------------------".format(i_dataset+1))
    PARAMS_i, TEST_ACCURACY_i, TRAIN_ACCURACY_i, TEST_LOSSES_i, TRAIN_LOSSES_i = (
        {},
        {},
        {},
        {},
        {},
    )
    TIME_SIMULATIONS_i, TIME_TRAINING_i, TIME_EVAL_i = {}, {}, {}
    METRICS_i = {}
    THETAS_ABC_i = {}
    key, key_theta, key_data = random.split(key, 3)
    TRUE_THETA = prior_simulator(key_theta)
    TRUE_THETA = np.array([25.])
    TRUE_DATA = data_simulator(key_data, TRUE_THETA)
    TRUE_DATA = np.array([25.])
    key, key_epsilon = random.split(key)
    time_iterations = {}
    EPSILONS_i = {1.0: np.inf}
    for alpha in ALPHAS:
        time_iteration = time.time()
        EPSILON = EPSILONS_i[alpha]
        key, key_epsilon = random.split(key_epsilon)
        
        dists,params,train_accuracy,train_losses,test_accuracy,test_losses,time_simulations,time_training,time_eval,metrics, thetas_abc = for_an_epsilon(i_dataset=i_dataset, alpha=alpha, key=key_epsilon, N_POINTS=N_POINTS, prior_simulator=prior_simulator, data_simulator=data_simulator, discrepancy=discrepancy, true_posterior_pdf=true_posterior_pdf, true_posterior_sample=true_posterior_sample, TRUE_DATA=TRUE_DATA, EPSILON=EPSILON, PRIOR_DIST=PRIOR_DIST, NN_ARGS=NN_ARGS, N_GRID=N_GRID, N_KDE=N_KDE, N_SAMPLE=N_SAMPLE, N_SAMPLES=N_SAMPLES, METRICS_TO_TEST= METRICS_TO_TEST, PATH=PATH, index_marginal=index_marginal)

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
        METRICS_i[alpha] = metrics
        THETAS_ABC_i[alpha] = thetas_abc
        time_iterations[alpha] = time.time() - time_iteration
        
    plot_metric_for_a_dataset(
        metric_name="C2ST", ALPHAS=ALPHAS, METRICS= METRICS_i, N_SAMPLES=N_SAMPLES, 
        PATH_NAME=PATH+"figures/C2ST_dataset_{}.png".format(i_dataset+1), 
        show=False, title="C2ST for dataset {} $\\theta =$ {:.3}".format(i_dataset+1, TRUE_THETA[index_marginal]))
    
    plot_metric_for_a_dataset(
        metric_name="RS_stat", ALPHAS=ALPHAS, METRICS= METRICS_i, N_SAMPLES=N_SAMPLES, 
        PATH_NAME=PATH+"figures/RS_stats_dataset_{}.png".format(i_dataset+1), show=False, 
        title="Ranksums statistic for dataset {} $\\theta =$ {:.3}".format(i_dataset+1, TRUE_THETA[index_marginal]))
    
    with lzma.open(PATH+"pickles/dataset_{}_params.xz".format(i_dataset+1), "wb") as f:
        pickle.dump(PARAMS_i, f)
        print("Pickle created at {}".format(PATH+"pickles/dataset_{}_params.xz".format(i_dataset+1)))
        
    with lzma.open(PATH+"pickles/dataset_{}_thetas_abc.xz".format(i_dataset+1), "wb") as f:
        pickle.dump(THETAS_ABC_i, f)
        print("Pickle created at {}".format(PATH+"pickles/dataset_{}_thetas_abc.xz".format(i_dataset+1)))
        
    create_csv_for_a_dataset(
        ALPHAS=ALPHAS, EPSILONS=EPSILONS_i, TEST_ACCURACY=TEST_ACCURACY_i, TRAIN_ACCURACY=TRAIN_ACCURACY_i, 
        TIME_SIMULATIONS=TIME_SIMULATIONS_i, TIME_TRAINING=TIME_TRAINING_i, TIME_EVAL=TIME_EVAL_i, 
        METRICS=METRICS_i, TRUE_DATA=TRUE_DATA, TRUE_THETA=TRUE_THETA, PRIOR_ARGS=PRIOR_ARGS, 
        MODEL_ARGS=MODEL_ARGS, NN_ARGS=NN_ARGS, INDEX_MARGINAL=index_marginal, 
        file_name=PATH+"csv/dataset_{}.csv".format(i_dataset+1, TRUE_THETA[index_marginal]))

    return (
        PARAMS_i,
        TEST_ACCURACY_i,
        TRAIN_ACCURACY_i,
        TEST_LOSSES_i,
        TRAIN_LOSSES_i,
        TIME_SIMULATIONS_i,
        TIME_TRAINING_i,
        TIME_EVAL_i,
        METRICS_i, 
        TRUE_DATA,
        TRUE_THETA,
    )
    
    
