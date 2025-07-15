from jax import random, jit, vmap, lax
import jax.numpy as jnp


def ABC_single(key, prior_simulator, data_simulator, discrepancy, epsilon, true_data):
    def cond_fun(val):
        _, _, _, dist = val
        return dist >= epsilon
    
    def body_fun(val):
        key, z, theta, dist = val
        key, key_theta, key_data = random.split(key, 3)
        theta_prop = prior_simulator(key_theta)
        data_prop = data_simulator(key_data, theta_prop)
        dist = discrepancy(data_prop, true_data)
        return key, data_prop, theta_prop, dist 
    
    key, key_theta = random.split(key)
    fake_theta = prior_simulator(key_theta)
    fake_data = jnp.zeros_like(true_data).astype(float)
    key, data, theta, dist = lax.while_loop(cond_fun, body_fun, (key, fake_data, fake_theta, epsilon+1))
    return data, theta, dist

ABC_single = jit(ABC_single, static_argnums=(1,2,3))

def ABC_epsilon(key, n_points, prior_simulator, data_simulator, discrepancy, epsilon, true_data):
    keys = random.split(key, n_points+1)   
    datas, thetas, dists = vmap(ABC_single, in_axes=(0, None, None, None, None, None))(keys[1:], prior_simulator, data_simulator, discrepancy, epsilon, true_data)
    return datas, thetas, dists, keys[0]

# ABC_epsilon = jit(ABC_epsilon, static_argnums=(2,3,4))

def get_epsilon_star(key, acceptance_rate, n_points, prior_simulator, data_simulator, discrepancy, true_data, 
                     quantile_rate = .9, epsilon = jnp.inf, return_accept = False):
    new_epsilon = epsilon
    accept = 1.
    datas, thetas, dists, key = ABC_epsilon(key, n_points, prior_simulator, data_simulator, discrepancy, epsilon, true_data)
    if epsilon == jnp.inf:
        print("Distances: min = ", jnp.min(dists), "max = ", jnp.max(dists), "mean = ", jnp.mean(dists), "std = ", jnp.std(dists))
    while accept > acceptance_rate:
        epsilon = new_epsilon
        new_epsilon = float(jnp.quantile(dists, quantile_rate))
        datas, thetas, dists, key = ABC_epsilon(key, n_points, prior_simulator, data_simulator, discrepancy, new_epsilon, true_data)
        key, subkey = random.split(key)
        keys_pred = random.split(subkey, n_points)
        datas_pred = vmap(data_simulator, in_axes=(0, 0))(keys_pred, thetas)
        new_dists = vmap(discrepancy, in_axes=(0, None))(datas_pred, true_data)
        accept = jnp.mean(new_dists < new_epsilon)
        epsilon = new_epsilon
        print("epsilon: ", epsilon, "acceptance rate: ", accept)
    if return_accept: 
        return epsilon, accept, key
    return epsilon, key

def get_dataset(key, n_points, prior_simulator, data_simulator, discrepancy, epsilon, true_data, index_marginal = 0):
    n_points = n_points//2
    zs, thetas, dists, key = ABC_epsilon(key, n_points, prior_simulator, data_simulator, discrepancy, epsilon, true_data)
    zs = zs.reshape(n_points, -1)
    thetas = thetas[:, index_marginal][:,None]
    key, key_perm = random.split(key)
    thetas_prime = thetas[random.permutation(key_perm, thetas.shape[0])]
    zs = jnp.concatenate([zs, zs], axis=0)
    thetas = jnp.concatenate([thetas, thetas_prime], axis=0)
    ys = jnp.append(jnp.zeros(n_points), jnp.ones(n_points)).astype(int)
    Xs = jnp.concatenate([thetas, zs], axis=1)
    return Xs, ys, dists, key

# def get_newdataset(key, n_points, prior_simulator, data_simulator, discrepancy, epsilon, true_data):
#     n_points = n_points//2
#     zs0, thetas0, _, key = ABC_epsilon(key, n_points, prior_simulator, data_simulator, discrepancy, epsilon, true_data)
#     key, subkey = random.split(key)
#     thetas1 = vmap(prior_simulator)(random.split(subkey, n_points))
#     zs = jnp.concatenate([zs0, zs0], axis=0)
#     thetas = jnp.concatenate([thetas0, thetas1], axis=0)
#     ys = jnp.append(jnp.zeros(n_points), jnp.ones(n_points)).astype(int)
#     Xs = jnp.concatenate([thetas, zs], axis=1)
#     return Xs, ys, key


def sample_from_pdf(key, grid, pdf_values, size):
    cdf = jnp.cumsum(pdf_values)
    cdf = cdf/cdf[-1]
    us = random.uniform(key, shape = (size,))
    thetas = jnp.interp(us, cdf, grid)
    return thetas

def sample_from_pdfs(key, grids, pdf_values, L):
    keys = random.split(key, grids.shape[0]+1)
    return vmap(sample_from_pdf, in_axes = (0, 0, 0, None))(keys[1:], grids, pdf_values, L), keys[0]

@jit 
def elu(x):
    return jnp.where(x > 0, x, jnp.expm1(x))
@jit
def logratio_z(params, mus, z):
    activations = jnp.append(jnp.array([mus]),z)
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = elu(outputs)
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits[0]-logits[1]

logratio_batch_z = (vmap(logratio_z, in_axes = (None, 0,  None)))

def NRE_posterior_pdf(params, mus, z, prior_logpdf):
    return jnp.exp(prior_logpdf(mus)+logratio_batch_z(params, mus, z))

def NRE_corrected_posterior_pdf(params, mus, z, kde_estimator):
    return kde_estimator(mus)*jnp.exp(logratio_batch_z(params, mus, z))


def find_grid_explorative(func, n_eval_explo, n_eval_final, min_grid, max_grid, threshold_factor=0.01, max_expansion=30, expansion_factor=0.1):
    expand = True
    expansions_left = max_expansion
    while expand and expansions_left > 0:
        grid_initial = jnp.linspace(min_grid, max_grid, n_eval_explo)
        pdf_initial = func(grid_initial)
        max_pdf = jnp.max(pdf_initial)
        
        while max_pdf == 0:
            
            new_min_grid = min_grid - expansion_factor * (max_grid - min_grid)
            new_max_grid = max_grid + expansion_factor * (max_grid - min_grid)
            print("Expanding grid to ", new_min_grid, new_max_grid)
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


def NRE_posterior_sample(key, params, z, prior_logpdf, n_grid, min_init, max_init, n_sample):
    grid, pdf = find_grid_explorative(lambda x: NRE_posterior_pdf(params, x, z, prior_logpdf), n_grid, n_grid, min_init, max_init)
    return sample_from_pdf(key, grid, pdf, n_sample)

def NRE_corrected_posterior_sample(key, params, z, kde_estimator, n_grid, min_init, max_init, n_sample):
    grid, pdf = find_grid_explorative(lambda x: NRE_corrected_posterior_pdf(params, x, z, kde_estimator), n_grid, n_grid, min_init, max_init)
    return sample_from_pdf(key, grid, pdf, n_sample)


    