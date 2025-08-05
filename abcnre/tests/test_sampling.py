import unittest
from abcnre.simulation.models import GaussGaussMultiDimModel
import jax

class TestSampler(unittest.TestCase):
    """
    A collection of unit tests for the 'add' function.
    """

    def test_sample_norm(self):
        model = GaussGaussMultiDimModel(
            mu0=0., sigma0=1.0, sigma=2.0, dim=2)

        key = jax.random.PRNGKey(123)
        theta1 = model.get_prior_sample(key)
        theta2= model.get_prior_sample(key)
        print(theta1, theta2)


# --- How to run the tests ---
if __name__ == '__main__':
    # This line discovers and runs all tests in the current file.
    unittest.main()
