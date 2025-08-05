import unittest
from numpy.testing import assert_array_almost_equal
from abcnre.simulation.models import GaussGaussMultiDimModel
from abcnre.simulation.models.base import SummarizedStatisticalModel
import jax


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

        x_draw = model.sample_theta_x(key)
        theta_draws, x_draws = model.sample_theta_x_multiple(key, 10)

        self.assertEqual(theta_draws.shape, (10, 2))
        self.assertEqual(x_draws.shape, (10, 5, 2))

    def test_draw_summary(self):
        model = get_2d_normal()
        key = jax.random.PRNGKey(123)
        theta_draws = model.get_prior_samples(key, 10)

        model_marg0 = SummarizedStatisticalModel(
            model, lambda theta: theta[0])


# --- How to run the tests ---
if __name__ == '__main__':
    # This line discovers and runs all tests in the current file.
    unittest.main()
