import unittest
from numpy.testing import assert_array_almost_equal
from abcnre.simulation.models import GaussGaussMultiDimModel
from abcnre.simulation.models.base import SummarizedStatisticalModel
from abcnre.simulation.sampler import RejectionSampler, get_epsilon_quantile
import jax
import jax.numpy as jnp

from jax import random, vmap

def get_2d_normal():
    return GaussGaussMultiDimModel(
        mu0=0., sigma0=1.0, sigma=2.0, dim=2, n_obs=5)


class TestSampler(unittest.TestCase):
    """
    A collection of unit tests for the 'add' function.
    """

    def test_sample_norm(self):
        model = get_2d_normal()
        key = jax.random.PRNGKey(123)
        theta_draw = model.get_prior_sample(key)
        theta_draws = model.get_prior_samples(key, 10)

        assert_array_almost_equal(theta_draw, theta_draws[0])
        self.assertEqual(theta_draw.shape, (2,))
        self.assertEqual(theta_draws.shape, (10, 2))

        _, x_draw = model.sample_theta_x(key)
        theta_draws, x_draws = model.sample_theta_x_multiple(key, 10)

        self.assertEqual(theta_draws.shape, (10, 2))
        self.assertEqual(x_draw.shape, (5, 2))
        self.assertEqual(x_draws.shape, (10, 5, 2))

    ##############
    def test_draw_summary(self):
        n_samples = 10
        model = get_2d_normal()
        key = jax.random.PRNGKey(123)

        theta_prior_draws = model.get_prior_samples(key, n_samples)
        theta_draws, x_draws = model.sample_theta_x_multiple(key, 10)

        model_marg0 = SummarizedStatisticalModel(
            model, lambda theta: jnp.array([ theta[0] ]))

        phi_prior_draws = model_marg0.get_prior_samples(key, n_samples)
        phi_draws, x_draws = model_marg0.sample_theta_x_multiple(key, n_samples)

        self.assertEqual(phi_prior_draws.shape, (10, 1))
        self.assertEqual(phi_draws.shape, (10, 1))

        # The x data has ten draws of datasets of length 5 and dimension 2
        self.assertEqual(x_draws.shape, (10, 5, 2))
        assert_array_almost_equal(
            theta_prior_draws[:,0], phi_prior_draws[:,0])
        assert_array_almost_equal(theta_draws[:,0], phi_draws[:,0])

        # Check with a non--indexical summary
        model_marg0 = SummarizedStatisticalModel(
            model, lambda theta: jnp.array([ jnp.sum(theta) ]))

        phi_prior_draws = model_marg0.get_prior_samples(key, n_samples)
        phi_draws, x_draws = model_marg0.sample_theta_x_multiple(key, n_samples)

        self.assertEqual(phi_prior_draws.shape, (10, 1))
        self.assertEqual(phi_draws.shape, (10, 1))
        self.assertEqual(x_draws.shape, (10, 5, 2))
        assert_array_almost_equal(
            theta_prior_draws[:,0] + theta_prior_draws[:,1],
            phi_prior_draws[:,0])
        assert_array_almost_equal(
            theta_draws[:,0] + theta_draws[:,1],
            phi_draws[:,0])

        # Check with a vector summary
        model_marg0 = SummarizedStatisticalModel(
            model, lambda theta: 2 * theta )

        phi_prior_draws = model_marg0.get_prior_samples(key, n_samples)
        phi_draws, x_draws = model_marg0.sample_theta_x_multiple(key, n_samples)

        self.assertEqual(phi_prior_draws.shape, (10, 2))
        self.assertEqual(phi_draws.shape, (10, 2))
        self.assertEqual(x_draws.shape, (10, 5, 2))
        assert_array_almost_equal(2 * theta_prior_draws, phi_prior_draws)
        assert_array_almost_equal(2 * theta_draws, phi_draws)


    ####################
    def test_rejection_sampler(self):
        n_samples = 10
        model = get_2d_normal()
        key = jax.random.PRNGKey(123)

        phi_draws, x_draws = model.sample_theta_x_multiple(key, n_samples)

        def d_fn(x):
            xm = jnp.mean(x)
            dist = xm ** 2
            return xm, dist

        means, dists = vmap(d_fn)(x_draws)

        # Check that the summary statistics are being computed correctly
        assert_array_almost_equal(jnp.mean(x_draws, axis=[1, 2]), means)

        key, key_eps = jax.random.split(key)
        epsilon, _ = get_epsilon_quantile(
            key_eps, model.sample_theta_x_multiple, d_fn, alpha=0.1)

        rej_sampler = RejectionSampler(model, d_fn, epsilon)

        theta_draws_reg, x_draws_reg = rej_sampler.sample_theta_x_multiple(key, n_samples)
        self.assertEqual(theta_draws_reg.shape, (n_samples, 2))
        self.assertEqual(x_draws_reg.shape, (n_samples, 5, 2))

        means_reg, dists_reg = vmap(d_fn)(x_draws_reg)
        self.assertTrue(jnp.all(dists_reg <= epsilon))

        # Test cacheing
        theta_draws_reg, x_draws_reg = \
            rej_sampler.sample_theta_x_multiple(key, n_samples, cache=True)
        metadata = rej_sampler.get_cache(key=key, n_samples=n_samples)

        means_reg, dists_reg = vmap(d_fn)(x_draws_reg)
        assert_array_almost_equal(dists_reg, metadata.distances)
        assert_array_almost_equal(means_reg, metadata.summary_stats)
        self.assertEqual(len(metadata.rejection_count), n_samples)

        # Check that cacheing fails if you call it with the wrong key or number of samples
        new_key, _ = jax.random.split(key)
        with self.assertRaises(ValueError) as msg:
            metadata = rej_sampler.get_cache(key=new_key)

        new_key, _ = jax.random.split(key)
        with self.assertRaises(ValueError) as msg:
            metadata = rej_sampler.get_cache(n_samples=n_samples + 1)



    ####################
    def test_get_epsilon_quantile(self):
        model = GaussGaussMultiDimModel(
            mu0=0., sigma0=1.0, sigma=1.0, dim=1, n_obs=1)
        key = jax.random.PRNGKey(123)

        def d_fn(x):
            xm = jnp.mean(x)
            dist = xm ** 2
            return xm, dist

        alpha = 0.4
        epsilon_quantile, epsilons = get_epsilon_quantile(
            key, model.sample_theta_x_multiple, d_fn, alpha=alpha
        )

        _, x_draws = model.sample_theta_x_multiple(key, 10000)
        xm = x_draws ** 2

        #print(epsilon_quantile)
        # Assert that approximation 100 * alpha % of summary statistics
        # are less than the target alpha
        self.assertTrue(
            jnp.abs(jnp.mean(xm <= epsilon_quantile) - alpha) < 1e-5)




# --- How to run the tests ---
if __name__ == '__main__':
    # This line discovers and runs all tests in the current file.
    unittest.main()
