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

ABC_epsilon = jit(ABC_epsilon, static_argnums=(1,2,3,4,5))

def get_epsilon_star(key, acceptance_rate, n_points, prior_simulator, data_simulator, discrepancy, true_data, quantile_rate = .9, epsilon = jnp.inf, return_accept = False):
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

def get_dataset(key, n_points, prior_simulator, data_simulator, discrepancy, epsilon, true_data):
    n_points = n_points//2
    zs, thetas, _, key = ABC_epsilon(key, n_points, prior_simulator, data_simulator, discrepancy, epsilon, true_data)
    key, key_perm = random.split(key)
    thetas_prime = thetas[random.permutation(key_perm, thetas.shape[0])]
    zs = jnp.concatenate([zs, zs], axis=0)
    thetas = jnp.concatenate([thetas, thetas_prime], axis=0)
    ys = jnp.append(jnp.zeros(n_points), jnp.ones(n_points)).astype(int)
    Xs = jnp.concatenate([thetas, zs], axis=1)
    return Xs, ys, key

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