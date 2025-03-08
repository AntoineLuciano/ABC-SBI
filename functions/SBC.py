import scipy.stats as stats
import jax.numpy as jnp
from jax import random, vmap, jit
from functions.simulation import ABC_epsilon
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

def plot_SBC(ranks, L, B = 0, err = .01, show = True, ax = None, save_name = "", title = ""):
    if ax is None: ax = plt.gca()
    N_sample = len(ranks)
    if B == 0: B = L+1
    N_sample = len(ranks)
    quantiles_lower = stats.binom.ppf(err/2, N_sample, 1/(B))
    quantiles_upper = stats.binom.ppf(1-err/2, N_sample, 1/(B))
    ax.hist(ranks, bins=np.arange(0, L+2, (L+2)//B), color='brown', edgecolor='darkred', align='mid')
    ax.fill_between(range(L+2), quantiles_lower, quantiles_upper, color='gray', alpha=0.3)
    ax.plot(range(L+2), N_sample/B * np.ones(L+2), color = "black", linestyle = "--")
    # Labels and title
    ax.set_xlabel('Rank Statistic')
    ax.set_ylabel('Frequency')
    
    if title != "": ax.set_title(title)
    if save_name != "": plt.savefig(save_name)
    if show:plt.show()
    

def find_grid_explorative(func, n_eval_explo, n_eval_final, min_grid, max_grid, threshold_factor=0.01, max_expansion=1000, expansion_factor=0.1):
    expand = True
    expansions_left = max_expansion
    while expand and expansions_left > 0:
        grid_initial = jnp.linspace(min_grid, max_grid, n_eval_explo)
        pdf_initial = func(grid_initial)
        max_pdf = jnp.max(pdf_initial)
        
        while max_pdf == 0:
            new_min_grid = min_grid - expansion_factor * (max_grid - min_grid)
            new_max_grid = max_grid + expansion_factor * (max_grid - min_grid)
            min_grid, max_grid = new_min_grid, new_max_grid
            grid_initial = jnp.linspace(min_grid, max_grid, n_eval_explo*100)
            pdf_initial = func(grid_initial)
            max_pdf = jnp.max(pdf_initial)
            
        threshold_value = threshold_factor * max_pdf
        significant_mask = pdf_initial > threshold_value

        if pdf_initial[0] > threshold_value:
            new_min_grid = min_grid - expansion_factor * (max_grid - min_grid)
        else:
            new_min_grid = min_grid
        if pdf_initial[-1] > threshold_value:
            new_max_grid = max_grid + expansion_factor * (max_grid - min_grid)
        else:
            new_max_grid = max_grid
        
        min_grid, max_grid = new_min_grid, new_max_grid

        if pdf_initial[0] <= threshold_value and pdf_initial[-1] <= threshold_value:
            expand = False
        
        expansions_left -= 1
        
    if jnp.any(significant_mask): 
        min_grid_opt, max_grid_opt = jnp.min(grid_initial[significant_mask]), jnp.max(grid_initial[significant_mask])
        new_min_grid = min_grid_opt - (max_grid_opt - min_grid_opt) * expansion_factor
        new_max_grid = max_grid_opt + (max_grid_opt - min_grid_opt) * expansion_factor
        min_grid_opt, max_grid_opt = new_min_grid, new_max_grid
    else:
        min_grid_opt, max_grid_opt = min_grid, max_grid
    
    grid_opt = jnp.linspace(min_grid_opt, max_grid_opt, n_eval_final)
    pdf_values = func(grid_opt)
    
    return grid_opt, pdf_values

def SBC_epsilon(key, N_SBC, L, params, epsilon, true_data, prior_simulator, prior_logpdf, data_simulator, discrepancy, n_grid_explo = 100, n_grid_final = 1000, minn = -50, maxx = 50, X = np.array([]), new_method = False, kde_estimator = None):
    if len(X) == 0:
        datas, thetas_tilde, _, key = ABC_epsilon(key, N_SBC, prior_simulator, data_simulator, discrepancy, epsilon, true_data)
    else:
        key, key_ = random.split(key)
        dim_theta = prior_simulator(key_).shape[0]
        datas = X[:, dim_theta:]
        thetas_tilde = X[:, :dim_theta].reshape(-1, dim_theta)
        N_SBC = len(datas)
    ranks = jnp.zeros(N_SBC)
    grids, pdf_values = jnp.zeros((N_SBC, n_grid_final)), jnp.zeros((N_SBC, n_grid_final))
    thetas = jnp.zeros((N_SBC, L))
    for i in tqdm(range(N_SBC), mininterval= N_SBC//20):
        if new_method and epsilon != np.inf:
            grid, pdf_value = find_grid_explorative(lambda x: new_post_pdf_z(params, x, datas[i], kde_estimator), n_grid_explo, n_grid_final, minn, maxx)
        else:
            grid, pdf_value = find_grid_explorative(lambda x: post_pdf_z(params, x, datas[i], prior_logpdf), n_grid_explo, n_grid_final, minn, maxx)
        grids = grids.at[i].set(grid)
        pdf_values = pdf_values.at[i].set(pdf_value)
    thetas, key = post_sample_batch(key, grids, pdf_values, L)
    ranks = jnp.sum(thetas < thetas_tilde, axis = 1)
    return ranks, thetas_tilde, thetas, key



