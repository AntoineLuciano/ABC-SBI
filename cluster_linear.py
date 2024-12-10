from jax import random, jit, vmap
from functions.simulation import get_dataset, get_epsilon_star
from functions.training import train_loop
from functions.SBC import SBC_epsilon, plot_SBC
from jax.scipy.special import expit, logit
import jax.numpy as jnp
import time
import pickle 
import lzma
import matplotlib.pyplot as plt
import seaborn as sns

@jit
def prior_simulator(key):
    return random.normal(key, (1,))*SIGMA0 + MU0

@jit
def data_simulator(key, beta):
    key, key_betas= random.split(key)
    betas = jnp.insert(random.normal(key_betas, (N_VAR-1,))*SIGMA0 + MU0, INDEX_BETA, beta)
    return random.normal(key, (N_DATA,))*SIGMA + jnp.dot(X_DESIGN, betas)

@jit
def discrepancy(y, y_true):
    return jnp.mean((y-y_true)**2)


key = random.PRNGKey(1)

N_VAR = 50
N_DATA = 1000
INDEX_BETA = N_VAR-1

def x_design_simulator(key):
    return random.normal(key, (N_DATA, N_VAR))

key, key_design = random.split(key)
X_DESIGN = x_design_simulator(key_design)

MU0, SIGMA0 = 0., 10.
PRIOR_ARGS = [MU0, SIGMA0]

SIGMA = 1.
MODEL_ARGS = [SIGMA, X_DESIGN]


from jax.scipy.stats import norm
PRIOR_LOGPDF = lambda x: norm.logpdf(x, loc = MU0, scale = SIGMA0)


key, key_beta = random.split(key)
TRUE_BETAS = jnp.append(jnp.array([15.]), random.normal(key_beta, (N_VAR-2,))*5. + MU0)
TRUE_BETAS = jnp.append(TRUE_BETAS, jnp.array([.1]))
TRUE_BETA = TRUE_BETAS[INDEX_BETA]

key, key_data = random.split(key)
TRUE_DATA = random.normal(key_data, (N_DATA,))*SIGMA + jnp.dot(X_DESIGN, TRUE_BETAS)

N_POINTS_TRAIN = 1000000
N_POINTS_TEST = 100000


import sys
if len(sys.argv) > 1:
    ACCEPT_RATE = float(sys.argv[1])
else: 
    ACCEPT_RATE = 1.
N_POINTS_EPS = 10000



N_EPOCHS = 100
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
MINN, MAXX = -50.,50. 
L = 127
N_SBC = (L+1)*100
EPSILON_STAR = jnp.inf
print('SIGMA0 = {}'.format(SIGMA0))

for ACCEPT_RATE in [1., .999, .99, .975, .95, .925, .9, .8, .75, .7, .6, .5]:
    print("\n\n--------------------")
    print("ACCEPT RATE: ", ACCEPT_RATE)
    print("--------------------\n\n")
    
    
    time_eps = time.time()
    print("Selection of epsilon star...")
    EPSILON_STAR, key = get_epsilon_star(key, ACCEPT_RATE, N_POINTS_EPS, prior_simulator, data_simulator, discrepancy, TRUE_DATA, quantile_rate = 1., epsilon = EPSILON_STAR)
    print('Time to select epsilon star: {:.2f}s\n'.format(time.time()-time_eps))

    print("Simulations of the testing dataset...")
    time_sim = time.time()
    X_test, y_test, key = get_dataset(key, N_POINTS_TEST, prior_simulator, data_simulator, discrepancy, EPSILON_STAR, TRUE_DATA)
    print('Time to simulate the testing dataset: {:.2f}s\n'.format(time.time()-time_sim))

    # print("Simulations of the training dataset...")
    # time_sim = time.time()
    # X_train, y_train, key = get_dataset(key, N_POINTS_TRAIN, prior_simulator, data_simulator, discrepancy, EPSILON_STAR, TRUE_DATA)
    # print('Time to simulate the training dataset: {:.2f}s\n'.format(time.time()-time_sim))


    print("Training the neural network...")
    time_nn = time.time()
    params, train_accuracy, train_losses, test_accuracy, test_losses, key = train_loop(key, N_EPOCHS, NUM_LAYERS, HIDDEN_SIZE, NUM_CLASSES, BATCH_SIZE, NUM_BATCH, LEARNING_RATE, WDECAY, PATIENCE, COOLDOWN, FACTOR, RTOL, ACCUMULATION_SIZE, LEARNING_RATE_MIN, prior_simulator, data_simulator, discrepancy, true_data = TRUE_DATA, X_train = None, y_train = None, X_test = X_test, y_test =  y_test, N_POINTS_TRAIN = N_POINTS_TRAIN, N_POINTS_TEST = N_POINTS_TEST, epsilon = EPSILON_STAR, verbose = True)
    print('Time to train the neural network: {:.2f}s\n'.format(time.time()-time_nn))


    print("Simulation Based Calibration...")
    time_sbc = time.time()

    ranks, thetas_tilde, thetas, key = SBC_epsilon(key = key, N_SBC = N_SBC, L = L, params = params, epsilon = EPSILON_STAR, true_data = TRUE_DATA, prior_simulator = prior_simulator, prior_logpdf = PRIOR_LOGPDF, data_simulator = data_simulator, discrepancy = discrepancy, n_grid_explo = N_GRID_EXPLO, n_grid_final = N_GRID_FINAL, minn = MINN, maxx = MAXX)

    print('Time to perform SBC: {:.2f}s\n'.format(time.time()-time_sbc))


    pickle_dico = {"ranks": ranks, "thetas_tilde": thetas_tilde, "thetas": thetas, "epsilon":EPSILON_STAR, "KEY":key, "N_SBC":N_SBC, "L":L, "N_GRID_EXPLO": N_GRID_EXPLO, 'N_GRID_FINAL': N_GRID_FINAL,"TRUE_DATA": TRUE_DATA, "TRUE_THETA": TRUE_BETA, "params": params, "train_accuracy":train_accuracy, "test_accuracy":test_accuracy, "MODEL_ARGS":MODEL_ARGS, "PRIOR_ARGS":PRIOR_ARGS, "N_POINTS_TRAIN":N_POINTS_TRAIN, "N_POINTS_TEST":N_POINTS_TEST, "N_DATA":N_DATA, "N_EPOCHS":N_EPOCHS, "LEARNING_RATE":LEARNING_RATE, "PATIENCE":PATIENCE, "COOLDOWN":COOLDOWN, "FACTOR":FACTOR, "RTOL":RTOL, "ACCUMULATION_SIZE":ACCUMULATION_SIZE, "LEARNING_RATE_MIN":LEARNING_RATE_MIN, "BATCH_SIZE":BATCH_SIZE, "NUM_BATCH":NUM_BATCH, "NUM_CLASSES":NUM_CLASSES, "HIDDEN_SIZE":HIDDEN_SIZE, "NUM_LAYERS":NUM_LAYERS, "WDECAY":WDECAY}

    name = "./pickle/Linear_Reg_notinfluent_{}_known_sigma_{}_sigma0_{}_acc_{:.2}_eps_{:.5}.xz".format(N_VAR, SIGMA, SIGMA0, ACCEPT_RATE, EPSILON_STAR)

    with lzma.open(name, "wb") as f:
        pickle.dump(pickle_dico, f)
    print("Data saved in ", name)

    title = "Linear Reg w/ known std\nsigma = {}, sigma0 = {} beta = {}\nalpha = {:.2%}, eps = {:.3} accuracy = {:.2%}".format(SIGMA, SIGMA0, TRUE_BETA, ACCEPT_RATE, EPSILON_STAR, test_accuracy[-1])

    name_plot = "./fig/Linear_Reg_notinfluent_{}_known_sigma_{}_sigma0_{}_acc_{:.2}_eps_{:.5}.png".format(N_VAR, SIGMA, SIGMA0, ACCEPT_RATE, EPSILON_STAR)

    plot_SBC(ranks, L, B = 16, title = title, save_name = name_plot)
    print("Plot saved in ", name_plot)
    
    print("\n\n--------------------")
    print("ITERATION (ACC = {}) DONE IN {} SECONDS!".format(ACCEPT_RATE, time.time()-time_eps))
    print("--------------------\n\n")