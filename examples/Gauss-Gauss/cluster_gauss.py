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

from functions.simulation import get_dataset, get_epsilon_star, get_newdataset
from functions.training import train_loop
from functions.SBC import SBC_epsilon, plot_SBC, find_grid_explorative, post_pdf_z
import jax.numpy as jnp
import time
import pickle 
import lzma
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
@jit
def prior_simulator(key):
    return random.normal(key, (1,))*SIGMA0 + MU0

@jit
def data_simulator(key, theta):
    return (random.normal(key, (N_DATA,))*SIGMA + theta).astype(float)

@jit
def discrepancy(y, y_true):
    return (jnp.mean(y) - jnp.mean(y_true))**2

def true_post(z):
    mu_post = (MU0*SIGMA**2 + SIGMA0**2 * np.sum(z))/(SIGMA0**2*len(z) + SIGMA**2)
    sigma2_post = 1/(1/SIGMA0**2 + len(z)/SIGMA**2)
    return stats.norm(loc = mu_post, scale = np.sqrt(sigma2_post))

key = random.PRNGKey(0)
 
MU0 = 0.
SIGMA = 1.
MODEL_ARGS = [SIGMA]
N_DATA = 100
N_POINTS_TRAIN = 1000000
N_POINTS_TEST = 100000
N_POINTS_EPS = 10000
sim_args = None


N_EPOCHS = 1
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
NUM_LAYERS = 7
WDECAY = .001
N_GRID_FINAL = 10000
N_GRID_EXPLO = 1000
L = 63
B = 16
N_SBC = (L+1)*1

PATH_RESULTS = os.getcwd() + "/examples/Gauss-Gauss/results/"
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)
    


SIGMAS0 = [2*SIGMA, 5*SIGMA, 10*SIGMA, 20*SIGMA]
ACCEPT_RATES = [1., .999, .99, .975, .95, .925, .9, .85, .8, .75]


for SIGMA0 in SIGMAS0:
    PATH_SIGMA0 = PATH_RESULTS+ "sigma0_{}/".format(SIGMA0)
    if not os.path.exists(PATH_SIGMA0):
        os.makedirs(PATH_SIGMA0)
        
    TRUE_MUS = [.1*SIGMA0, .5*SIGMA0, SIGMA0, 1.5*SIGMA0]
    PRIOR_ARGS = [MU0, SIGMA0]
    PRIOR_LOGPDF = lambda x: norm.logpdf(x, loc = MU0, scale = SIGMA0)
    MINN, MAXX = norm.ppf(1e-5, loc = MU0, scale = SIGMA0), norm.ppf(1-1e-5, loc = MU0, scale = SIGMA0)
    
    for TRUE_MU in TRUE_MUS:
        PATH_TRUE_MU = PATH_SIGMA0 + "mu_{}/".format(TRUE_MU)
        if not os.path.exists(PATH_TRUE_MU):
            os.makedirs(PATH_TRUE_MU)
        EPSILON_STAR = jnp.inf
        key, subkey = random.split(key)
        TRUE_DATA = data_simulator(subkey, TRUE_MU)
        for ACCEPT_RATE in ACCEPT_RATES:
            print("\n\n--------------------")
            print("SIGMA0 = {}, TRUE_MU = {}, ACCEPT_RATE = {}".format(SIGMA0, TRUE_MU, ACCEPT_RATE))
            print("--------------------\n\n")
                    
            
            time_eps = time.time()
            print("Selection of epsilon star...")
            EPSILON_STAR, key = get_epsilon_star(key, ACCEPT_RATE, N_POINTS_EPS, prior_simulator, data_simulator, discrepancy, TRUE_DATA, quantile_rate = .99, epsilon = EPSILON_STAR)
            print('Time to select epsilon star: {:.2f}s\n'.format(time.time()-time_eps))

            print("Simulations of the testing dataset...")
            time_sim = time.time()
            X_test, y_test, key = get_newdataset(key, N_POINTS_TEST, prior_simulator, data_simulator, discrepancy, EPSILON_STAR, TRUE_DATA)
            print('Time to simulate the testing dataset: {:.2f}s\n'.format(time.time()-time_sim))

            print("Simulations of the training dataset...")
            time_sim = time.time()
            X_train, y_train, key = get_newdataset(key, N_POINTS_TRAIN, prior_simulator, data_simulator, discrepancy, EPSILON_STAR, TRUE_DATA)
            print('Time to simulate the training dataset: {:.2f}s\n'.format(time.time()-time_sim))


            print("Training the neural network...")
            time_nn = time.time()
            params, train_accuracy, train_losses, test_accuracy, test_losses, key = train_loop(key, N_EPOCHS, NUM_LAYERS, HIDDEN_SIZE, NUM_CLASSES, BATCH_SIZE, NUM_BATCH, LEARNING_RATE, WDECAY, PATIENCE, COOLDOWN, FACTOR, RTOL, ACCUMULATION_SIZE, LEARNING_RATE_MIN, prior_simulator, data_simulator, discrepancy, true_data = TRUE_DATA, X_train = X_train, y_train = y_train, X_test = X_test, y_test =  y_test, N_POINTS_TRAIN = N_POINTS_TRAIN, N_POINTS_TEST = N_POINTS_TEST, epsilon = EPSILON_STAR, verbose = True)
            print('Time to train the neural network: {:.2f}s\n'.format(time.time()-time_nn))


            print("Simulation Based Calibration...")
            time_sbc = time.time()

            ranks, thetas_tilde, thetas, key = SBC_epsilon(key = key, N_SBC = N_SBC, L = L, params = params, epsilon = EPSILON_STAR, true_data = TRUE_DATA, prior_simulator = prior_simulator, prior_logpdf = PRIOR_LOGPDF, data_simulator = data_simulator, discrepancy = discrepancy, n_grid_explo = N_GRID_EXPLO, n_grid_final = N_GRID_FINAL, minn = MINN, maxx = MAXX)

            print('Time to perform SBC: {:.2f}s\n'.format(time.time()-time_sbc))


            pickle_dico = {"ACCEPT_RATE": ACCEPT_RATE,"ranks": ranks, "thetas_tilde": thetas_tilde, "thetas": thetas, "epsilon":EPSILON_STAR, "KEY":key, "N_SBC":N_SBC, "L":L, "N_GRID_EXPLO": N_GRID_EXPLO, 'N_GRID_FINAL': N_GRID_FINAL,"TRUE_DATA": TRUE_DATA, "TRUE_THETA": TRUE_MU, "params": params, "train_accuracy":train_accuracy, "test_accuracy":test_accuracy, "MODEL_ARGS":MODEL_ARGS, "PRIOR_ARGS":PRIOR_ARGS, "N_POINTS_TRAIN":N_POINTS_TRAIN, "N_POINTS_TEST":N_POINTS_TEST, "N_DATA":N_DATA, "N_EPOCHS":N_EPOCHS, "LEARNING_RATE":LEARNING_RATE, "PATIENCE":PATIENCE, "COOLDOWN":COOLDOWN, "FACTOR":FACTOR, "RTOL":RTOL, "ACCUMULATION_SIZE":ACCUMULATION_SIZE, "LEARNING_RATE_MIN":LEARNING_RATE_MIN, "BATCH_SIZE":BATCH_SIZE, "NUM_BATCH":NUM_BATCH, "NUM_CLASSES":NUM_CLASSES, "HIDDEN_SIZE":HIDDEN_SIZE, "NUM_LAYERS":NUM_LAYERS, "WDECAY":WDECAY}



            NAME = "GaussGauss_sigma_{}_sigma0_{}_mu_{}_acc_{:.3}_eps_{:.5}".format(SIGMA, SIGMA0, TRUE_MU, ACCEPT_RATE, EPSILON_STAR)
            NAMEFIG = PATH_TRUE_MU+'figures/'+NAME+".png"
            NAMEFILE = PATH_TRUE_MU+'pickles/'+NAME+".xy"
            
            if not os.path.exists(PATH_TRUE_MU+'figures/'):
                os.makedirs(PATH_TRUE_MU+'figures/')
            if not os.path.exists(PATH_TRUE_MU+'pickles/'):
                os.makedirs(PATH_TRUE_MU+'pickles/')
                
            
            with lzma.open(NAMEFILE, "wb") as f:
                pickle.dump(pickle_dico, f)
            print("Data saved in ", NAMEFILE)

            title = "Normal w/ known std\nsigma = {}, sigma0 = {} mu = {}\nalpha = {:.2%}, eps = {:.3} accuracy = {:.2%}".format(SIGMA, SIGMA0, TRUE_MU, ACCEPT_RATE, EPSILON_STAR, test_accuracy[-1])
    
            
            f, ax = plt.subplots(1,3, figsize = (15,5))
            sns.kdeplot(thetas_tilde, label = "Thetas_tilde", ax = ax[0])
            sns.kdeplot(thetas[:,0], label = "Thetas", ax = ax[0])

            ax[0].legend()
            f.suptitle(f'GaussGauss with sigma = {SIGMA} sigma0 = {SIGMA0} mu = {TRUE_MU} alpha = {ACCEPT_RATE:.1%}, epsilon = {EPSILON_STAR:.3} accuracy = {test_accuracy[-1]:.2%}')

            grid_approx, pdf_approx = find_grid_explorative(lambda x: post_pdf_z(params, x, TRUE_DATA, PRIOR_LOGPDF), N_GRID_EXPLO, N_GRID_FINAL, MINN, MAXX)
            grid_true, pdf_true = find_grid_explorative(lambda x: true_post(TRUE_DATA).pdf(x), N_GRID_EXPLO, N_GRID_FINAL, MINN, MAXX)
            
            Z_approx = np.trapz(pdf_approx, grid_approx)
            ax[1].plot(grid_approx, pdf_approx/Z_approx, label = "Approx")
            ax[1].plot(grid_true, pdf_true, label = "True")
            ax[1].legend()
            ax[1].set_title("Posterior comparison of the true data")
            plot_SBC(ranks, L, B, ax = ax[2])
            ax[2].set_title("SBC with Rank Statistics")
            f.savefig(NAMEFIG)
            plt.close(f)  
            
            
            print("\n\n--------------------")
            print("ITERATION (ACC = {}) DONE IN {} SECONDS!".format(ACCEPT_RATE, time.time()-time_eps))
            print("--------------------\n\n")