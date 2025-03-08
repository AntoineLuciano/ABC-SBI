from jax import random, jit, vmap
import os
path = os.getcwd()
print("Old path:", path)
path = (path.split('/'))
path = path[:path.index("ABC-SBI")+1]
path = '/'.join(path)
print("New path:", path)
os.chdir(path)
import sys
sys.path.append(path)
from functions.simulation import get_dataset, ABC_epsilon, get_epsilon_star
from functions.training import train_loop
from functions.SBC import SBC_epsilon, plot_SBC, find_grid_explorative, post_pdf_z,post_sample, new_post_pdf_z
import jax.numpy as jnp
import time
import pickle 
import lzma
from jax.scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from sbibm.metrics import c2st
import torch
from tqdm import tqdm
import pymc3 as pm
import pytensor
time_begin = time.time()
@jit
def prior_simulator(key):
    return random.normal(key, (1,))*SIGMA0 + MU0


def data_simulator_raw(key, beta, x_design, index_beta):
    key, subkey = random.split(key)
    betas = jnp.insert(random.normal(subkey, (x_design.shape[1]-1,))*SIGMA0 + MU0, index_beta, beta)
    return random.normal(key, (x_design.shape[0],))*SIGMA+ jnp.dot(x_design, betas)

def beta_simulator_raw(key, beta, x_design, index_beta):
    return jnp.insert(random.normal(key, (x_design.shape[1]-1,))*SIGMA0 + MU0, index_beta, beta)

@jit
def discrepancy(y, y_true):
    return jnp.mean((y-y_true)**2)

def x_design_simulator(key, n_data, K):
    return random.normal(key, (n_data, K))


key = random.PRNGKey(0)
import sys
if len(sys.argv)>1: 
    K = int(sys.argv[1])
    print("K:", K)
else: 
    K = 10
    
PATH_RESULTS = os.getcwd() + "/examples/Linear-Reg/plots_for_paper/local/K_{}/".format(int(K))
PATH_FIGURES = PATH_RESULTS + "figures/"
PATH_PICKLES = PATH_RESULTS + "pickles/"
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)
if not os.path.exists(PATH_FIGURES):
    os.makedirs(PATH_FIGURES)
if not os.path.exists(PATH_PICKLES):
    os.makedirs(PATH_PICKLES)
    

N_KDE = 1000
N_POINTS_TRAIN = 10000
N_POINTS_TEST = 10000
N_SAMPLE = 100
N_SAMPLES = 1
N_EPSILON = 1000
N_DATASETS = 2
N_EPOCHS = 1   


MU0 = 0.
SIGMA0 = 10.
SIGMA = 1.
N_DATA = 500
PRIOR_ARGS = [MU0, SIGMA0]

PRIOR_LOGPDF = lambda x: norm.logpdf(x, loc = MU0, scale = SIGMA0)
MINN, MAXX = norm.ppf(1e-5, loc = MU0, scale = SIGMA0), norm.ppf(1-1e-5, loc = MU0, scale = SIGMA0)
prior = stats.norm(loc = MU0, scale = SIGMA0)


key = random.PRNGKey(0)
key, key_beta = random.split(key)
TRUE_BETAS  = (random.normal(key_beta, (K,))*SIGMA0 + MU0)
TRUE_BETAS = TRUE_BETAS[jnp.argsort(jnp.abs(TRUE_BETAS))[::-1]]
       
key, key_design = random.split(key)
X_DESIGN = x_design_simulator(key_design, N_DATA, K)
    
MODEL_ARGS = [SIGMA, X_DESIGN]
key, key_data = random.split(key)
TRUE_DATA = random.normal(key_data, (N_DATA,))*SIGMA + jnp.dot(X_DESIGN, TRUE_BETAS)

        
        
trace_path = PATH_RESULTS + "data.pkl"

if not os.path.exists(trace_path):
    with pm.Model() as model:
        # Convert X_DESIGN to a tensor variable
        X_DESIGN_shared = np.array(X_DESIGN)
        # Priors for unknown model parameters
        betas = pm.Normal('betas', mu=MU0, sigma=SIGMA0, shape=K)
        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal('y_obs', mu=pytensor.tensor.dot(X_DESIGN_shared, betas), sigma=SIGMA, observed=TRUE_DATA)
        # Sample from the posterior
        trace = pm.sample(N_KDE, tune=1000)
    beta_post = np.array(trace.posterior.betas).reshape(-1, K)

    dico = {"X_DESIGN":X_DESIGN, "TRUE_DATA":TRUE_DATA, "TRUE_BETAS":TRUE_BETAS, "true_post":beta_post}
    with open(trace_path, "wb") as f:
        pickle.dump(dico, f)
else:
    with open(trace_path, "rb") as f:
        dico = pickle.load(f)




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
N_GRID_FINAL = 10000
N_GRID_EXPLO = 1000


EPSILONS = {}
C2ST_ABC = {}
C2ST_NN = {}
C2ST_KDE_NN = {}
C2ST_KDE_NN_EMPIRIC = {}
PARAMS = {}
TEST_ACCURACY = {}

key = random.PRNGKey(0)
TRUE_BETAS = dico["TRUE_BETAS"]
X_DESIGN = dico["X_DESIGN"]
TRUE_DATA = dico["TRUE_DATA"]
beta_post = dico["true_post"]
MODEL_ARGS = [SIGMA, X_DESIGN]


N_EPSILON = 1000000
zs, mus, dists, key = ABC_epsilon(key, N_EPSILON, prior_simulator, data_simulator, discrepancy, np.inf, TRUE_DATA)
ALPHAS = [.99,.9, .75,.5,.25, .1,.05, .01, .005, .001, .0005,.0001]
EPSILONS = {1.: jnp.inf}
for alpha in ALPHAS: 
    EPSILONS[alpha] = jnp.quantile(dists, alpha)

ALPHAS = [1.]+ALPHAS





if K > 2:
    INDEX_BETAS = [0, K-1, K//2]
else: 
    INDEX_BETAS = [0, K-1]
    for INDEX_BETA in INDEX_BETAS:
        data_simulator = jit(lambda key, beta: data_simulator_raw(key, beta, X_DESIGN, INDEX_BETA)) 
        TRUE_BETA = TRUE_BETAS[INDEX_BETA]
        if INDEX_BETA == 0:
            beta_mode = "max"
            PATH_BETA = PATH_RESULTS + "beta_max/"
        elif INDEX_BETA == K-1:
            beta_mode = "min"
            PATH_BETA = PATH_RESULTS + "beta_min/"
        else:
            beta_mode = "random"
            PATH_BETA = PATH_RESULTS + "beta_random/"
        if not os.path.exists(PATH_BETA):
                os.makedirs(PATH_BETA)
    
        EPSILONS_i = {}
        PARAMS_i = {}
        TEST_ACCURACY_i = {}
        C2ST_ABC_i = {}
        C2ST_NN_i = {}
        C2ST_KDE_NN_EMPIRIC_i = {}
        
    for alpha in (ALPHAS):
        if alpha == 1.:  EPSILONS_i[alpha] = jnp.inf
        else: EPSILONS_i[alpha] = jnp.quantile(dists, alpha)
        
        EPSILON_STAR = EPSILONS_i[alpha]
        print(f"----------------------\nBETA = {beta_mode} ALPHA = {alpha} EPSILON = {EPSILON_STAR:.2}\n----------------------")


        print("Simulations of the testing dataset...")
        time_sim = time.time()
        key, subkey = random.split(key)
        X_test, y_test, key = get_dataset(subkey, N_POINTS_TEST, prior_simulator, data_simulator, discrepancy, EPSILON_STAR, TRUE_DATA)
        print("Test check:", np.any(X_test[:,0]!=np.sort(X_test[:,0])))
        print('Time to simulate the testing dataset: {:.2f}s\n'.format(time.time()-time_sim))


        print("Simulations of the training dataset...")
        time_sim = time.time()
        key, subkey = random.split(key)
        X_train, y_train, key = get_dataset(subkey, N_POINTS_TRAIN, prior_simulator, data_simulator, discrepancy, EPSILON_STAR, TRUE_DATA)
        print("Train check:", np.any(X_train[:,0]!=np.sort(X_train[:,0])))
        print('Time to simulate the training dataset: {:.2f}s\n'.format(time.time()-time_sim))
        
        print("Training the neural network...")
        time_nn = time.time()
        params, train_accuracy, train_losses, test_accuracy, test_losses, key = train_loop(key, N_EPOCHS, NUM_LAYERS, HIDDEN_SIZE, NUM_CLASSES, BATCH_SIZE, NUM_BATCH, LEARNING_RATE, WDECAY, PATIENCE, COOLDOWN, FACTOR, RTOL, ACCUMULATION_SIZE, LEARNING_RATE_MIN, prior_simulator, data_simulator, discrepancy, true_data = TRUE_DATA, X_train = X_train, y_train = y_train, X_test = X_test, y_test =  y_test, N_POINTS_TRAIN = N_POINTS_TRAIN, N_POINTS_TEST = N_POINTS_TEST, epsilon = EPSILON_STAR, verbose = True)
        print('Time to train the neural network: {:.2f}s\n'.format(time.time()-time_nn))

        PARAMS_i[alpha] = params
        TEST_ACCURACY_i[alpha] = test_accuracy[-1]
        
        
        X = np.concatenate([X_train, X_test], axis=0)
        key, subkey = random.split(key)
        kde_approx_empiric = gaussian_kde(X[random.choice(subkey, X.shape[0], (N_KDE,)),0])

        print("Find grid KDE NN empiric approx...", end ="")
        time_grid = time.time()
        grid_kde_nn_empiric, pdf_kde_nn_empiric = find_grid_explorative(lambda x: new_post_pdf_z(params, x, TRUE_DATA, kde_approx_empiric), 10000, 10000, MINN, MAXX)
        print("Done in {:.2} sec.".format(time.time()-time_grid))
        print("Find grid NN approx...", end = "")
        time_grid= time.time()
        grid_nn, pdf_nn = find_grid_explorative(lambda x: post_pdf_z(params, x, TRUE_DATA, PRIOR_LOGPDF), 10000, 10000, MINN, MAXX)
        print("Done in {:.2} sec.".format(time.time()-time_grid))
        accuraccy_abc, accuraccy_nn, accuraccy_kde_nn, accuraccy_kde_nn_empiric = [], [], [], []
        print("Sampling {} times...".format(N_SAMPLES), end ="")
        time_sample = time.time()
        sample_true = dico["true_post"][:,INDEX_BETA]
        for _ in range(N_SAMPLES):
        
            key, key_abc, key_nn, key_kde_nn = random.split(key, 4)
            sample_nn = post_sample(key_nn, grid_nn, pdf_nn, N_SAMPLE)
            sample_abc = X[random.choice(key_abc, X.shape[0], (N_SAMPLE,)),0]
            sample_kde_nn_empiric = post_sample(key_kde_nn, grid_kde_nn_empiric, pdf_kde_nn_empiric, N_SAMPLE)
            if np.isnan(sample_nn).any(): accuraccy_nn.append(1)
            else: accuraccy_nn.append(np.array(c2st(torch.tensor(sample_true)[:,None], torch.tensor(sample_nn)[:,None]))[0])
            if np.isnan(sample_abc).any(): accuraccy_abc.append(1)
            else: accuraccy_abc.append(np.array(c2st(torch.tensor(sample_true)[:,None], torch.tensor(sample_abc)[:,None]))[0])
            if np.isnan(sample_kde_nn_empiric).any(): accuraccy_kde_nn_empiric.append(1)
            else: accuraccy_kde_nn_empiric.append(np.array(c2st(torch.tensor(sample_true)[:,None], torch.tensor(sample_kde_nn_empiric)[:,None]))[0])
        print("Done in {:.2} sec.".format(time.time()-time_sample)) 
        C2ST_ABC_i[alpha] = np.array(accuraccy_abc)
        C2ST_NN_i[alpha] = np.array(accuraccy_nn)
        C2ST_KDE_NN_EMPIRIC_i[alpha] = np.array(accuraccy_kde_nn_empiric)

        print("\n---------------------------------------nITERATION BETA = {} ALPHA = {} DONE IN {:.2} SECONDS!\n---------------------------------------".format(beta_mode+1,alpha, time.time()-time_sim))
    
    C2ST_ABC[beta_mode] = C2ST_ABC_i
    C2ST_NN[beta_mode] = C2ST_NN_i
    C2ST_KDE_NN[beta_mode] = C2ST_KDE_NN_i
    C2ST_KDE_NN_EMPIRIC[beta_mode] = C2ST_KDE_NN_EMPIRIC_i
    PARAMS[beta_mode] = PARAMS_i
    TEST_ACCURACY[beta_mode] = TEST_ACCURACY_i
    EPSILONS[beta_mode] = EPSILONS_i
    
    mean_abc = np.array([C2ST_ABC_i[alpha].mean() for alpha in ALPHAS])
    std_abc = np.array([C2ST_ABC_i[alpha].std() for alpha in ALPHAS])
    conf_interval_abc = 1.96*std_abc/np.sqrt(N_SAMPLES)

    mean_nn = np.array([C2ST_NN_i[alpha].mean() for alpha in ALPHAS])
    std_nn = np.array([C2ST_NN_i[alpha].std() for alpha in ALPHAS])
    conf_interval_nn = 1.96*std_nn/np.sqrt(N_SAMPLES)

    mean_kde_nn = np.array([C2ST_KDE_NN_i[alpha].mean() for alpha in ALPHAS])
    std_kde_nn = np.array([C2ST_KDE_NN_i[alpha].std() for alpha in ALPHAS])
    conf_interval_kde_nn = 1.96*std_kde_nn/np.sqrt(N_SAMPLES)

    mean_kde_nn_empiric = np.array([C2ST_KDE_NN_EMPIRIC_i[alpha].mean() for alpha in ALPHAS])
    std_kde_nn_empiric = np.array([C2ST_KDE_NN_EMPIRIC_i[alpha].std() for alpha in ALPHAS])
    conf_interval_kde_nn_empiric = 1.96*std_kde_nn_empiric/np.sqrt(N_SAMPLES)

    f, ax = plt.subplots(1, 1, figsize = (10, 5))
    plt.errorbar(ALPHAS, mean_abc, yerr = conf_interval_abc, label = "ABC", color = "orange", fmt='o-', capsize=5)
    plt.errorbar(ALPHAS, mean_nn, yerr = conf_interval_nn, label = "ABC-NRE without correction", color = "red", fmt='o-', capsize=5)
    plt.errorbar(ALPHAS, mean_kde_nn, yerr = conf_interval_kde_nn, label = "ABC-NRE with correction", color = "blue", fmt='o-', capsize=5)
    plt.errorbar(ALPHAS, mean_kde_nn_empiric, yerr = conf_interval_kde_nn_empiric, label = "ABC-NRE with correction empiric", color = "purple", fmt='o-', capsize=5)

    plt.title("Methods comparison for $\mu =$ {:.2} $\sigma_0 =$ {}".format(TRUE_MU, int(SIGMA0)))
    plt.xlabel("$\epsilon$")
    plt.gca().invert_xaxis()
    plt.xscale("log")
    plt.ylabel("C2ST accuracy")
    plt.axhline(.5, color = "grey", linestyle = "--")
    plt.legend(loc = "upper center")
    name_fig_i = PATH_FIGURES + "{}_method_comparison_single_mu_{:.3}_sigma0_{}.png".format(i_dataset+1,TRUE_MU,int(SIGMA0))
    plt.savefig(name_fig_i)
    plt.show()

    print("Figure saved in :", name_fig_i)
    
    dico_i = {"ALPHAS":ALPHAS, "C2ST_ABC":C2ST_ABC_i, "C2ST_NN":C2ST_NN_i, "C2ST_KDE_NN":C2ST_KDE_NN_i, "C2ST_KDE_NN_EMPIRIC":C2ST_KDE_NN_EMPIRIC_i, "PARAMS":PARAMS_i, "TEST_ACCURACY":TEST_ACCURACY_i, "EPSILONS":EPSILONS_i, "TRUE_MU":TRUE_MU, "TRUE_DATA":TRUE_DATA, "SIGMA0":SIGMA0, "MU0":MU0, "SIGMA":SIGMA, "N_DATA":N_DATA, "MODEL_ARGS":MODEL_ARGS, "PRIOR_ARGS":PRIOR_ARGS, "N_POINTS_TRAIN":N_POINTS_TRAIN, "N_POINTS_TEST":N_POINTS_TEST, "N_EPOCHS":N_EPOCHS, "LEARNING_RATE":LEARNING_RATE, "PATIENCE":PATIENCE, "COOLDOWN":COOLDOWN, "FACTOR":FACTOR, "RTOL":RTOL, "ACCUMULATION_SIZE":ACCUMULATION_SIZE, "LEARNING_RATE_MIN":LEARNING_RATE_MIN, "BATCH_SIZE":BATCH_SIZE, "NUM_BATCH":NUM_BATCH, "NUM_CLASSES":NUM_CLASSES, "HIDDEN_SIZE":HIDDEN_SIZE, "NUM_LAYERS":NUM_LAYERS, "WDECAY":WDECAY, "N_GRID_FINAL":N_GRID_FINAL, "N_GRID_EXPLO":N_GRID_EXPLO, "N_KDE":N_KDE, "N_SAMPLE":N_SAMPLE, "N_SAMPLES":N_SAMPLES}
    n_file_i = "examples/Gauss-Gauss/Gauss_Gauss_1D_known_sigma/pickles/dataset_{}_method_comparison_single_mu_{:.2}_sigma0_{}.xy".format(i_dataset+1, TRUE_MU,int(SIGMA0))
    
    name_file_i = PATH_PICKLES + "{}_method_comparison_single_mu_{:.2}_sigma0_{}.xy".format(i_dataset+1, TRUE_MU,int(SIGMA0))
    with lzma.open(name_file_i, "wb") as f:
        pickle.dump(dico_i, f)
    print("Data saved in ", name_file_i)
    
    print("\n\n\n\n---------------------------------------\nITERATION DATASET = {} DONE IN {:.2} SECONDS!\n---------------------------------------\n\n\n\n".format(i_dataset+1, time.time()-time_data_set))
    

C2ST_ABC_array = np.array([[C2ST_ABC[i_dataset][alpha][j] for i_dataset in range(N_DATASETS) for j in range(N_SAMPLES)] for alpha in ALPHAS])
C2ST_NN_array = np.array([[C2ST_NN[i_dataset][alpha][j] for i_dataset in range(N_DATASETS) for j in range(N_SAMPLES)] for alpha in ALPHAS])
C2ST_KDE_NN_array = np.array([[C2ST_KDE_NN[i_dataset][alpha][j] for i_dataset in range(N_DATASETS) for j in range(N_SAMPLES)] for alpha in ALPHAS])
C2ST_KDE_NN_EMPIRIC_array = np.array([[C2ST_KDE_NN_EMPIRIC[i_dataset][alpha][j] for i_dataset in range(N_DATASETS) for j in range(N_SAMPLES)] for alpha in ALPHAS])

mean_abc = np.array([C2ST_ABC_array[i].mean() for i in range(C2ST_ABC_array.shape[0])])
std_abc = np.array([C2ST_ABC_array[i].std() for i in range(C2ST_ABC_array.shape[0])])
conf_interval_abc = 1.96*std_abc/np.sqrt(N_SAMPLES)

mean_nn = np.array([C2ST_NN_array[i].mean() for i in range(C2ST_NN_array.shape[0])])
std_nn = np.array([C2ST_NN_array[i].std() for i in range(C2ST_NN_array.shape[0])])
conf_interval_nn = 1.96*std_nn/np.sqrt(N_SAMPLES)


f, ax = plt.subplots(1, 1, figsize = (10, 5))
plt.errorbar(ALPHAS, mean_abc, yerr = conf_interval_abc, label = "ABC", color = "orange", fmt='o-', capsize=5, alpha = .5)
plt.errorbar(ALPHAS, mean_nn, yerr = conf_interval_nn, label = "ABC-NRE without correction", color = "red", fmt='o-', capsize=5, alpha = .5)
plt.errorbar(ALPHAS, mean_kde_nn, yerr = conf_interval_kde_nn, label = "ABC-NRE with correction", color = "blue", fmt='o-', capsize=5, alpha = .5)
plt.errorbar(ALPHAS, mean_kde_nn_empiric, yerr = conf_interval_kde_nn_empiric, label = "ABC-NRE with correction empiric", color = "purple", fmt='o-', capsize=5, alpha = .5)

plt.title("Methods comparison for $\sigma_0 =$ {}".format(int(SIGMA0)))
plt.xlabel("$\epsilon$")
plt.gca().invert_xaxis()
plt.xscale("log")
plt.ylabel("C2ST accuracy")
plt.axhline(.5, color = "grey", linestyle = "--")
plt.legend(loc = "upper center")
name_fig = PATH_FIGURES + "method_comparison_sigma0_{}.png".format(int(SIGMA0))
name_file = PATH_PICKLES + "method_comparison_sigma0_{}.xy".format(int(SIGMA0))

plt.savefig(name_fig)
plt.show()

print("Figure saved in :", name_fig)


dico = {"EPSILONS": EPSILONS, "PARAMS": PARAMS, "TEST_ACCURACY": TEST_ACCURACY, "C2ST_ABC": C2ST_ABC, "C2ST_NN": C2ST_NN, "C2ST_KDE_NN": C2ST_KDE_NN, "C2ST_KDE_NN_EMPIRIC": C2ST_KDE_NN_EMPIRIC, "ALPHAS": ALPHAS, "TRUE_MUS": TRUE_MUS, "TRUE_DATAS": TRUE_DATAS, "SIGMA0": SIGMA0, "MU0": MU0, "SIGMA": SIGMA, "N_DATA": N_DATA, "MODEL_ARGS": MODEL_ARGS, "PRIOR_ARGS": PRIOR_ARGS, "N_POINTS_TRAIN": N_POINTS_TRAIN, "N_POINTS_TEST": N_POINTS_TEST, "N_EPOCHS": N_EPOCHS, "LEARNING_RATE": LEARNING_RATE, "PATIENCE": PATIENCE, "COOLDOWN": COOLDOWN, "FACTOR": FACTOR, "RTOL": RTOL, "ACCUMULATION_SIZE": ACCUMULATION_SIZE, "LEARNING_RATE_MIN": LEARNING_RATE_MIN, "BATCH_SIZE": BATCH_SIZE, "NUM_BATCH": NUM_BATCH, "NUM_CLASSES": NUM_CLASSES, "HIDDEN_SIZE": HIDDEN_SIZE, "NUM_LAYERS": NUM_LAYERS, "WDECAY": WDECAY, "N_GRID_FINAL": N_GRID_FINAL, "N_GRID_EXPLO": N_GRID_EXPLO, "N_KDE": N_KDE, "N_SAMPLE": N_SAMPLE, "N_SAMPLES": N_SAMPLES}

with lzma.open(name_file, "wb") as f:
    pickle.dump(dico, f)
print("Data saved in ", name_file)

print("\n\n\n\n---------------------------------------\nITERATION DONE IN {:.2} SECONDS!\n---------------------------------------\n\n\n\n".format(time.time()-time_begin))

