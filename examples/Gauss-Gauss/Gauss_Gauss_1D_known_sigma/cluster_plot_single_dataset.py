from jax import random, jit, vmap
import os
path = os.getcwd()
print("Old path:", path)
path = (path.split('/'))
path = path[:path.index("ABC-SBI")+1]
path = '/'.join(path)
print("New path:", path)
os.chdir(path)
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
from tqdm.notebook import tqdm

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

from functions.SBC import logratio_z, logratio_batch_z, post_pdf_z, find_grid_explorative
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
    zs, thetas, _, key = ABC_gauss(key, true_data, epsilon, n_points)
    key, key_perm = random.split(key)
    thetas_prime = thetas[random.permutation(key_perm, jnp.arange(n_points))]
    zs = jnp.concatenate([zs, zs], axis=0)
    thetas = jnp.concatenate([thetas, thetas_prime], axis=0)
    ys = jnp.append(jnp.zeros(n_points), jnp.ones(n_points)).astype(int)
    Xs = jnp.concatenate([thetas, zs], axis=1)
    return Xs, ys, key


key = random.PRNGKey(0)
import sys
if len(sys.argv)>1: 
    SIGMA0 = float(sys.argv[1])
    print("SIGMA0:", SIGMA0)
else: 
    SIGMA0 = 10.
MU0 = 0.
SIGMA = 1.
TRUE_MU = 2.*SIGMA0
N_DATA = 5
MODEL_ARGS = [SIGMA]
PRIOR_ARGS = [MU0, SIGMA0]

key, subkey = random.split(key)
TRUE_DATA = random.normal(subkey, (N_DATA,))*SIGMA + TRUE_MU

PRIOR_LOGPDF = lambda x: norm.logpdf(x, loc = MU0, scale = SIGMA0)
MINN, MAXX = norm.ppf(1e-5, loc = MU0, scale = SIGMA0), norm.ppf(1-1e-5, loc = MU0, scale = SIGMA0)
prior = stats.norm(loc = MU0, scale = SIGMA0)
N_EPSILON = 1000000
zs, mus, dists, key = ABC_gauss(key, TRUE_DATA, np.inf, N_EPSILON)
ALPHAS = [.99,.9, .75,.5,.25, .1,.05, .01, .005, .001, .0005,.0001]
EPSILONS = {1.: jnp.inf}
for alpha in ALPHAS: 
    EPSILONS[alpha] = jnp.quantile(dists, alpha)

ALPHAS = [1.]+ALPHAS
# Training loop
key = random.PRNGKey(0)

N_KDE = 100000
N_POINTS_TRAIN = 1000000
N_POINTS_TEST = 100000
N_POINTS_EPS = 100000
N_SAMPLE = 10000
N_SAMPLES = 10


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


C2ST_ABC = {}
C2ST_NN = {}
C2ST_KDE_NN = {}
C2ST_KDE_NN_EMPIRIC = {}
PARAMS = {}
TEST_ACCURACY = {}
for alpha in tqdm(ALPHAS):
    EPSILON_STAR = EPSILONS[alpha]
    print(f"----------------------\nALPHA = {alpha} EPSILON = {EPSILON_STAR:.2}\n----------------------")


    print("Simulations of the testing dataset...")
    time_sim = time.time()
    key, subkey = random.split(key)
    X_test, y_test, key = get_dataset_gauss(subkey, N_POINTS_TEST, prior_simulator, data_simulator, discrepancy, EPSILON_STAR, TRUE_DATA)
    print("Test check:", np.any(X_test[:,0]!=np.sort(X_test[:,0])))
    print('Time to simulate the testing dataset: {:.2f}s\n'.format(time.time()-time_sim))


    print("Simulations of the training dataset...")
    time_sim = time.time()
    key, subkey = random.split(key)
    X_train, y_train, key = get_dataset_gauss(subkey, N_POINTS_TRAIN, prior_simulator, data_simulator, discrepancy, EPSILON_STAR, TRUE_DATA)
    print("Train check:", np.any(X_train[:,0]!=np.sort(X_train[:,0])))
    print('Time to simulate the training dataset: {:.2f}s\n'.format(time.time()-time_sim))
    
    print("Training the neural network...")
    time_nn = time.time()
    params, train_accuracy, train_losses, test_accuracy, test_losses, key = train_loop(key, N_EPOCHS, NUM_LAYERS, HIDDEN_SIZE, NUM_CLASSES, BATCH_SIZE, NUM_BATCH, LEARNING_RATE, WDECAY, PATIENCE, COOLDOWN, FACTOR, RTOL, ACCUMULATION_SIZE, LEARNING_RATE_MIN, prior_simulator, data_simulator, discrepancy, true_data = TRUE_DATA, X_train = X_train, y_train = y_train, X_test = X_test, y_test =  y_test, N_POINTS_TRAIN = N_POINTS_TRAIN, N_POINTS_TEST = N_POINTS_TEST, epsilon = EPSILON_STAR, verbose = True)
    print('Time to train the neural network: {:.2f}s\n'.format(time.time()-time_nn))

    PARAMS[alpha] = params
    TEST_ACCURACY[alpha] = test_accuracy[-1]
    
    kde_approx = lambda x: true_pseudo_post(x, np.mean(TRUE_DATA), EPSILONS[alpha], prior)
    X = np.concatenate([X_train, X_test], axis=0)
    key, subkey = random.split(key)
    kde_approx_empiric = gaussian_kde(X[random.choice(subkey, X.shape[0], (N_KDE,)),0])
    grid_kde_nn, pdf_kde_nn = find_grid_explorative(lambda x: new_post_pdf_z(params, x, TRUE_DATA, kde_approx), 10000, 10000, MINN, MAXX)
    grid_kde_nn_empiric, pdf_kde_nn_empiric = find_grid_explorative(lambda x: new_post_pdf_z(params, x, TRUE_DATA, kde_approx_empiric), 10000, 10000, MINN, MAXX)
    grid_nn, pdf_nn = find_grid_explorative(lambda x: post_pdf_z(params, x, TRUE_DATA, PRIOR_LOGPDF), 10000, 10000, MINN, MAXX)
    accuraccy_abc, accuraccy_nn, accuraccy_kde_nn, accuraccy_kde_nn_empiric = [], [], [], []
    for _ in range(N_SAMPLES):
        sample_true = true_post(TRUE_DATA).rvs(N_SAMPLE)
        key, key_abc, key_nn, key_kde_nn = random.split(key, 4)
        sample_nn = post_sample(key_nn, grid_nn, pdf_nn, N_SAMPLE)
        sample_kde_nn = post_sample(key_kde_nn, grid_kde_nn, pdf_kde_nn, N_SAMPLE)
        sample_abc = X[random.choice(key_abc, X.shape[0], (N_SAMPLE,)),0]
        sample_kde_nn_empiric = post_sample(key_kde_nn, grid_kde_nn_empiric, pdf_kde_nn_empiric, N_SAMPLE)
        if np.isnan(sample_nn).any(): accuraccy_nn.append(1)
        else: accuraccy_nn.append(np.array(c2st(torch.tensor(sample_true)[:,None], torch.tensor(sample_nn)[:,None]))[0])
        if np.isnan(sample_kde_nn).any(): accuraccy_kde_nn.append(1)
        else: accuraccy_kde_nn.append(np.array(c2st(torch.tensor(sample_true)[:,None], torch.tensor(sample_kde_nn)[:,None]))[0])
        if np.isnan(sample_abc).any(): accuraccy_abc.append(1)
        else: accuraccy_abc.append(np.array(c2st(torch.tensor(sample_true)[:,None], torch.tensor(sample_abc)[:,None]))[0])
        if np.isnan(sample_kde_nn_empiric).any(): accuraccy_kde_nn_empiric.append(1)
        else: accuraccy_kde_nn_empiric.append(np.array(c2st(torch.tensor(sample_true)[:,None], torch.tensor(sample_kde_nn_empiric)[:,None]))[0])
        
    C2ST_ABC[alpha] = np.array(accuraccy_abc)
    C2ST_NN[alpha] = np.array(accuraccy_nn)
    C2ST_KDE_NN[alpha] = np.array(accuraccy_kde_nn)
    C2ST_KDE_NN_EMPIRIC[alpha] = np.array(accuraccy_kde_nn_empiric)



mean_abc = np.array([C2ST_ABC[alpha].mean() for alpha in ALPHAS])
std_abc = np.array([C2ST_ABC[alpha].std() for alpha in ALPHAS])
conf_interval_abc = 1.96*std_abc/np.sqrt(N_SAMPLES)

mean_nn = np.array([C2ST_NN[alpha].mean() for alpha in ALPHAS])
std_nn = np.array([C2ST_NN[alpha].std() for alpha in ALPHAS])
conf_interval_nn = 1.96*std_nn/np.sqrt(N_SAMPLES)

mean_kde_nn = np.array([C2ST_KDE_NN[alpha].mean() for alpha in ALPHAS])
std_kde_nn = np.array([C2ST_KDE_NN[alpha].std() for alpha in ALPHAS])
conf_interval_kde_nn = 1.96*std_kde_nn/np.sqrt(N_SAMPLES)

mean_kde_nn_empiric = np.array([C2ST_KDE_NN_EMPIRIC[alpha].mean() for alpha in ALPHAS])
std_kde_nn_empiric = np.array([C2ST_KDE_NN_EMPIRIC[alpha].std() for alpha in ALPHAS])
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
plt.savefig("examples/Gauss-Gauss/figures/method_comparison_outlier_mu_{:.2}_sigma0_{}.png".format(TRUE_MU,int(SIGMA0)))
plt.show()