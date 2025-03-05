from jax import random, jit, vmap
import os
import sys
path = os.getcwd()
print("Old path:", path)
path = (path.split('/'))
path = path[:path.index("ABC-SBI")+1]
path = '/'.join(path)
print("New path:", path)
os.chdir(path)
sys.path.append(path)
print(sys.path)
from functions.SBC import logratio_z, logratio_batch_z, post_pdf_z, find_grid_explorative
# from functions.simulation import get_dataset, ABC_epsilon, get_epsilon_star
from functions.training import train_loop
from functions.SBC import SBC_epsilon, plot_SBC, find_grid_explorative, post_pdf_z, post_sample, new_post_pdf_z
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


def true_post(z):
    mu_post = (MU0*SIGMA**2 + SIGMA0**2 * np.sum(z))/(SIGMA0**2*len(z) + SIGMA**2)
    sigma2_post = 1/(1/SIGMA0**2 + len(z)/SIGMA**2)
    return stats.norm(loc = mu_post, scale = np.sqrt(sigma2_post))

def true_ratio_z(mus, z, prior, posterior):
    return posterior(z).pdf(mus)/prior.pdf(mus)

def true_decision_z(mus, z, prior, posterior):
    return 1/(1+1/true_ratio_z(mus, z, prior, posterior))

def true_pseudo_ratio_z(mus, z, bar_xobs, epsilon, prior, posterior):
    pseudo = true_pseudo_post(mus, bar_xobs, epsilon, prior)
    Z_pseudo = np.trapz(pseudo, mus)
    return posterior(z).pdf(mus)/pseudo*Z_pseudo

def true_pseudo_decision_z(mus, z, bar_xobs, epsilon, prior, posterior):
    return 1/(1+1/true_pseudo_ratio_z(mus, z, bar_xobs, epsilon, prior, posterior))

def true_pseudo_post(mus, bar_xobs, epsilon, prior):
    return prior.pdf(mus)*(norm.cdf(bar_xobs+np.sqrt(epsilon), loc = mus, scale = SIGMA/np.sqrt(N_DATA)) - norm.cdf(bar_xobs-np.sqrt(epsilon), loc = mus, scale = SIGMA/np.sqrt(N_DATA)))

def true_pseudo_decision_z(mus, z, bar_xobs, epsilon, prior, posterior):
    return 1/(1+1/true_pseudo_ratio_z(mus, z, bar_xobs, epsilon, prior, posterior))

def decision_z(params, mus, z):
    return 1/(1+jnp.exp(-logratio_z(params, mus, z)))
def decision_batch_z(params, mus, z):
    return 1/(1+jnp.exp(-logratio_batch_z(params, mus, z)))

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


def get_dataset_gauss(key, n_points, prior_simulator, data_simulator, discrepancy, epsilon, true_data):
    n_points = n_points//2
    zs0, thetas0, _, key = ABC_gauss(key, true_data, epsilon, n_points)
    _, thetas1, _, key = ABC_gauss(key, true_data, epsilon, n_points)    
    zs = jnp.concatenate([zs0, zs0], axis=0)
    thetas = jnp.concatenate([thetas0, thetas1], axis=0)
    ys = jnp.append(jnp.zeros(n_points), jnp.ones(n_points)).astype(int)
    Xs = jnp.concatenate([thetas, zs], axis=1)
    return Xs, ys, key
key = random.PRNGKey(0)
 
MU0 = 0.
SIGMA = 1.
MODEL_ARGS = [SIGMA]

N_POINTS_TRAIN = 100000
N_POINTS_TEST = 10000
N_POINTS_EPS = 1000000
N_MUS = 10
N_C2ST = 10000
sim_args = None


N_EPOCHS = 50
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


<<<<<<< HEAD
PATH_RESULTS = os.getcwd() + "/examples/Gauss-Gauss/Gauss_Gauss_1D_known_sigma/results/local/"
=======
PATH_RESULTS = os.getcwd() + "/examples/Gauss-Gauss/Gauss_Gauss_1D_known_sigma/results/cluster/"
>>>>>>> e9a9a6dc7e43f9c867082b515581cd309de9a3ae
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)
    

<<<<<<< HEAD
N_DATAS = [10]
SIGMAS0 = [10*SIGMA, 20*SIGMA, 5*SIGMA]
=======
N_DATAS = [50]
print("N_DATA = ", N_DATAS)
SIGMAS0 = [20*SIGMA]
>>>>>>> e9a9a6dc7e43f9c867082b515581cd309de9a3ae
ALPHAS = [1., .99, .9,  .5, .1, .01, .001]


for N_DATA in N_DATAS:
    PATH_N_DATA = PATH_RESULTS+ "N_data_{}/".format(N_DATA)
    if not os.path.exists(PATH_N_DATA):
        os.makedirs(PATH_N_DATA)

    
    for SIGMA0 in SIGMAS0:
        PATH_SIGMA0 = PATH_N_DATA+ "sigma0_{}/".format(int(SIGMA0))
        if not os.path.exists(PATH_SIGMA0):
            os.makedirs(PATH_SIGMA0)
        
        PRIOR_ARGS = [MU0, SIGMA0]
        PRIOR_LOGPDF = lambda x: norm.logpdf(x, loc = MU0, scale = SIGMA0)
        prior = stats.norm(loc = MU0, scale = SIGMA0)
        TRUE_MUS = np.array([-2.7*SIGMA0, -2*SIGMA0, 2.1*SIGMA0, 3*SIGMA0])

        MINN, MAXX = prior.interval(1-1e-5)
        dico_sigma0 = {"SIGMA0": SIGMA0, "TRUE_MUS": TRUE_MUS, "N_DATA":N_DATA}
        TRUE_DATAS = {}
        PARAMS_SIGMA0 = {}
        TEST_ACCURACY_SIGMA0 = {}
        EPSILONS_SIGMA0 = {}
        C2ST_ACCURACY_SIGMA0 = {}
        for i,TRUE_MU in enumerate(TRUE_MUS):
            i = i+1
            PATH_TRUE_MU = PATH_SIGMA0 + "{}_mu_{:.2}/".format(i, TRUE_MU)
            if not os.path.exists(PATH_TRUE_MU):
                os.makedirs(PATH_TRUE_MU)
            key, subkey = random.split(key)
            TRUE_DATA = data_simulator(subkey, TRUE_MU)
            TRUE_DATAS[i] = TRUE_DATA
            zs, mus, dists, key = ABC_gauss(key, TRUE_DATA, 1000, 1000000)
            EPSILONS = {1.: jnp.inf}
            PARAMS = {}
            TEST_ACCURACY = {}
            C2ST_ACCURACY = {}
            true_sample = true_post(TRUE_DATA).rvs(N_C2ST)
        
            for alpha in ALPHAS: 
                EPSILONS[alpha] = jnp.quantile(dists, alpha)
            dico_mu = {"TRUE_MU":TRUE_MU, "TRUE_DATA":TRUE_DATA, "N_DATA":N_DATA, "SIGMA0":SIGMA0, "EPSILONS":EPSILONS, "ALPHAS":ALPHAS}
            for ALPHA in ALPHAS:
                time_iter = time.time()
                EPSILON = EPSILONS[ALPHA]
                print("\n\n--------------------")
                print("SIGMA0 = {}, TRUE_MU = {}, ALPHA = {}".format(SIGMA0, TRUE_MU, ALPHA))
                print("--------------------\n\n")
                        
                print("Simulations of the testing dataset...")
                time_sim = time.time()
                key, subkey = random.split(key)
                X_test, y_test, key = get_dataset_gauss(subkey, N_POINTS_TEST, prior_simulator, data_simulator, discrepancy, EPSILON, TRUE_DATA)
                print("Test check:", np.any(X_test[:,0]!=np.sort(X_test[:,0])))
                print('Time to simulate the testing dataset: {:.2f}s\n'.format(time.time()-time_sim))


                print("Simulations of the training dataset...")
                time_sim = time.time()
                key, subkey = random.split(key)
                X_train, y_train, key = get_dataset_gauss(subkey, N_POINTS_TRAIN, prior_simulator, data_simulator, discrepancy, EPSILON, TRUE_DATA)
                print("Train check:", np.any(X_train[:,0]!=np.sort(X_train[:,0])))
                print('Time to simulate the training dataset: {:.2f}s\n'.format(time.time()-time_sim))
                
                print("Training the neural network...")
                time_nn = time.time()
                params, train_accuracy, train_losses, test_accuracy, test_losses, key = train_loop(key, N_EPOCHS, NUM_LAYERS, HIDDEN_SIZE, NUM_CLASSES, BATCH_SIZE, NUM_BATCH, LEARNING_RATE, WDECAY, PATIENCE, COOLDOWN, FACTOR, RTOL, ACCUMULATION_SIZE, LEARNING_RATE_MIN, prior_simulator, data_simulator, discrepancy, true_data = TRUE_DATA, X_train = X_train, y_train = y_train, X_test = X_test, y_test =  y_test, N_POINTS_TRAIN = N_POINTS_TRAIN, N_POINTS_TEST = N_POINTS_TEST, epsilon = EPSILON, verbose = True)
                print('Time to train the neural network: {:.2f}s\n'.format(time.time()-time_nn))
                
<<<<<<< HEAD
                kde_approx = gaussian_kde(X_test[:,0], bw_method = "scott")

=======
                kde_approx = gaussian_kde(X_train[:,0], bw_method = "scott")
                print("Find grid...")
>>>>>>> e9a9a6dc7e43f9c867082b515581cd309de9a3ae
                grid_kde_nn, pdf_kde_nn = find_grid_explorative(lambda x: new_post_pdf_z(params, x, TRUE_DATA, kde_approx), N_GRID_EXPLO, N_GRID_FINAL, MINN, MAXX)
                print("Sample...")
                key, subkey = random.split(key)
                sample_kde_nn = post_sample(subkey, grid_kde_nn, pdf_kde_nn, N_C2ST)
                if np.isnan(sample_kde_nn).any() or np.isnan(true_sample).any():
                    accuracy_c2st = [np.inf]
                else: 
                    accuracy_c2st = c2st(torch.tensor(true_sample[:,None]), torch.tensor(sample_kde_nn[:,None]))
                print("C2ST accuracy: ", np.array(accuracy_c2st)[0])
                

                PARAMS[ALPHA] = params
                TEST_ACCURACY[ALPHA] = test_accuracy[-1]
                C2ST_ACCURACY[ALPHA] = np.array(accuracy_c2st)[0]

                
                print("\n\n\n WE SAVED DATA FOR ALPHA = {} SO NOW C2ST, TEST AND PARAMS HAVE {}, {}, {} KEYS".format(ALPHA, len(C2ST_ACCURACY), len(TEST_ACCURACY), len(PARAMS)))
                pickle_dico = {"ACCEPT_RATE":ALPHA, "epsilon":EPSILON, "KEY":key, "TRUE_DATA": TRUE_DATA, "TRUE_THETA": TRUE_MU, "params": params, "train_accuracy":train_accuracy, "test_accuracy":test_accuracy, "MODEL_ARGS":MODEL_ARGS, "PRIOR_ARGS":PRIOR_ARGS, "N_POINTS_TRAIN":N_POINTS_TRAIN, "N_POINTS_TEST":N_POINTS_TEST, "N_DATA":N_DATA, "N_EPOCHS":N_EPOCHS, "LEARNING_RATE":LEARNING_RATE, "PATIENCE":PATIENCE, "COOLDOWN":COOLDOWN, "FACTOR":FACTOR, "RTOL":RTOL, "ACCUMULATION_SIZE":ACCUMULATION_SIZE, "LEARNING_RATE_MIN":LEARNING_RATE_MIN, "BATCH_SIZE":BATCH_SIZE, "NUM_BATCH":NUM_BATCH, "NUM_CLASSES":NUM_CLASSES, "HIDDEN_SIZE":HIDDEN_SIZE, "NUM_LAYERS":NUM_LAYERS, "WDECAY":WDECAY, "C2ST_ACCURACY":C2ST_ACCURACY}
                          #  "ranks": ranks, "thetas_tile": thetas_tilde, "thetas": thetas}



                NAME = "GaussGauss_ndata_{}_sigma0_{}_mu_{:.3}_alpha_{:.3}_eps_{:.5}".format(N_DATA, int(SIGMA0), TRUE_MU, ALPHA, EPSILON)
                NAMEFILE = PATH_TRUE_MU+NAME+".xy"
  
                
                with lzma.open(NAMEFILE, "wb") as f:
                    pickle.dump(pickle_dico, f)
                print("Data saved in ", NAMEFILE)
                
                
                print("\n\n--------------------")
                print("ITERATION (ALPHA = {}) DONE IN {} SECONDS!".format(ALPHA, time.time()-time_iter))
                print("--------------------\n\n")
            dico_mu["PARAMS"] = PARAMS
            dico_mu["TEST_ACCURACY"] = TEST_ACCURACY
            dico_mu["C2ST_ACCURACY"] = C2ST_ACCURACY
            name_dico_mu = "GaussGauss_ndata_{}_sigma0_{}_mu_{:.3}".format(N_DATA, int(SIGMA0), TRUE_MU)
            with lzma.open(PATH_SIGMA0+name_dico_mu+".xz", "wb") as f:
                pickle.dump(dico_mu, f)
            print("Data saved in ", PATH_SIGMA0+name_dico_mu+".xz")
            print("\n\n--------------------\n\n")
            
            PARAMS_SIGMA0[i] = PARAMS
            TEST_ACCURACY_SIGMA0[i] = TEST_ACCURACY
            EPSILONS_SIGMA0[i] = EPSILONS
            C2ST_ACCURACY_SIGMA0[i] = C2ST_ACCURACY
        dico_sigma0["C2ST_ACCURACY"] = C2ST_ACCURACY_SIGMA0
        dico_sigma0["PARAMS"] = PARAMS_SIGMA0
        dico_sigma0["TEST_ACCURACY"] = TEST_ACCURACY_SIGMA0
        dico_sigma0["EPSILONS"] = EPSILONS_SIGMA0
        dico_sigma0["TRUE_DATAS"] = TRUE_DATAS
        
        name_dico_sigma0 = "GaussGauss_ndata_{}_sigma0_{}".format(N_DATA, int(SIGMA0))
        with lzma.open(PATH_N_DATA+name_dico_sigma0+".xz", "wb") as f:
            pickle.dump(dico_sigma0, f)
        print("Data saved in ", PATH_N_DATA+name_dico_sigma0+".xz")
        print("\n\n--------------------\n\n")