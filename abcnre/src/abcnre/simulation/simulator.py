"""
Main ABC simulator class.

This module contains the ABCSimulator class which is the main interface
for ABC simulation.
"""

import jax.numpy as jnp
from jax import random
from typing import Callable, Tuple, Any, Dict, Optional, Union
from jax import numpy as jnp
from pathlib import Path
import yaml

from .sampler import RejectionSampler
from .base import ABCSampleResult, ABCTrainingResult, ABCSingleResult
from functools import cached_property
from .io import generate_sampler_hash
from ..utils.comparison import are_simulators_equivalent
import jax
import flax.linen as nn

from ..training import (
    NNConfig,
    get_nn_config,
    train_regressor,
    TrainingConfig,
    NetworkConfig,
)


def create_summary_stats_fn(
    network: nn.Module,
    params: Dict[str, Any],
    network_type: str,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Create summary statistics function adapted to different network architectures.

    Args:
        network: Trained summary statistics network
        params: Trained parameters
        network_type: Type of network ("deepset", "conditioned_deepset", "MLP")

    Returns:
        Summary statistics function that takes observations and returns summary stats
    """

    def summary_stats_fn(x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute summary statistics S(x) from observations.

        Args:
            x: Observations, shape (batch, n, d) or (batch, d)

        Returns:
            Summary statistics, shape (batch, summary_dim)
        """
        if x.ndim == 2:
            x = x[None, :, :]  # (1, n, d)
        summary_stats = network.apply(params, x, training=False)

        return summary_stats  # (batch, summary_dim)

    return summary_stats_fn


# RG: I'm not sure what the ABCSimulator class is adding beyond RejectionSampler
# What is the difference between a simulator and a sampler?


class ABCSimulator:
    """
    Main class for ABC simulation and data generation.

    This class handles:
    - ABC rejection sampling through a RejectionSampler.
    - Epsilon management and quantile computation.
    - Provides a high-level interface for generating simulation data and persistence.
    """

    def __init__(
        self,
        model=None,
        observed_data: Optional[jnp.ndarray] = None,
        epsilon: Optional[float] = None,
        quantile_distance: Optional[float] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initializes the ABC simulator.

        Args:
            model: A statistical model instance with a defined interface.
            observed_data: The observed dataset to match against. Optional if epsilon=inf.
            epsilon: The ABC tolerance threshold. Defaults to infinity.
            quantile_distance: If provided (0,1), automatically computes epsilon
                             as this quantile of the distance distribution.
            config: An optional dictionary for advanced ABC parameters.
        """
        self.model = model
        self.trained_summary_stats = False

        # Store whether epsilon was explicitly provided
        self._epsilon_provided = epsilon is not None

        # Set epsilon first to determine if observed_data is required
        self.epsilon = jnp.inf if epsilon is None else epsilon

        # If epsilon is infinite, observed_data is optional (for prior sampling)
        if self.epsilon == jnp.inf:
            self.observed_data = observed_data  # Can be None
        else:
            if observed_data is None:
                raise ValueError(
                    "observed_data is required when epsilon is finite. "
                    "For prior sampling without observed data, use epsilon=np.inf"
                )
            self.observed_data = observed_data

        # Set default configuration and override with user-provided config
        self.config = {
            "quantile_n_samples": 10000,
            "verbose": True,
            "summary_stats_enabled": False,
            **(config if config is not None else {}),
        }

        self.quantile_distance = quantile_distance

        # Initialize observed summary statistics if applicable
        self._init_summary_stats()

        # Initialize the underlying sampler (always create it)
        self.sampler = None
        if self.model is not None:
            self._initialize_sampler()
            if self.observed_data is not None:
                self._init_epsilon()

    @cached_property
    def sampler_id(self) -> str:
        """A unique, deterministic identifier for this simulator configuration."""
        if self.model is None or self.observed_data is None:
            raise ValueError(
                "Model and observed_data must be set to generate a sampler ID."
            )

        model_config = getattr(self.model, "get_model_args", lambda: {})()

        return generate_sampler_hash(
            model_config=model_config,
            observed_data=self.observed_data,
            epsilon=self.epsilon,
        )

    def __eq__(self, other) -> bool:
        """
        Compare this simulator with another for equivalence.

        Uses the robust comparison function from utils.comparison that checks:
        - Model equivalence (type + configuration)
        - Observed data equivalence
        - Epsilon equivalence
        - Summary network equivalence (configuration + trained parameters)
        """
        if not isinstance(other, ABCSimulator):
            return False
        return are_simulators_equivalent(self, other)

    def __hash__(self):
        """Make simulator hashable for use in sets/dicts (based on sampler_id)."""
        try:
            return hash(self.sampler_id)
        except ValueError:
            # Fallback if sampler_id can't be computed
            return hash(
                (
                    id(self.model),
                    id(self.observed_data) if self.observed_data is not None else None,
                    self.epsilon,
                )
            )

    def _init_summary_stats(self):
        """Initializes summary statistics based on the model and data."""
        self.observed_summary_stats = None
        self.config["summary_stats_enabled"] = False
        if (
            self.model is not None
            and self.observed_data is not None
            and hasattr(self.model, "summary_stat_fn")
        ):
            try:
                self.observed_summary_stats = self.model.summary_stat_fn(
                    self.observed_data
                )
                self.config["summary_stats_enabled"] = True
            except (AttributeError, NotImplementedError):
                # Model is expected to have the function but it's not implemented
                pass

    def _init_epsilon(self):
        """Initializes epsilon, computing it from a quantile if requested."""

        # Only compute epsilon from quantile if epsilon was not explicitly provided
        if self.quantile_distance is not None and not self._epsilon_provided:
            if not (0 < self.quantile_distance <= 1):
                raise ValueError("quantile_distance must be between 0 and 1")

            if self.config.get("verbose", False):
                print(f"Computing epsilon for {self.quantile_distance:.1%} quantile...")
            if self.quantile_distance == 1.0:
                # If quantile_distance is 1.0, set epsilon to the maximum distance
                self.epsilon = jnp.inf
                if self.sampler:
                    self.sampler.update_epsilon(self.epsilon)
                print("Setting epsilon to infinity (maximum distance).")
            # Use a fixed key for this internal, automatic setup
            else:
                key = random.PRNGKey(0)
                computed_epsilon, _, _ = self.get_epsilon_quantile(
                    key, self.quantile_distance, self.config["quantile_n_samples"]
                )
                self.epsilon = computed_epsilon
                if self.sampler:
                    self.sampler.update_epsilon(self.epsilon)

                if self.config.get("verbose", False):
                    print(f"Computed epsilon = {self.epsilon:.6f}")
        elif self._epsilon_provided and self.config.get("verbose", False):
            print(f"Using provided epsilon = {self.epsilon:.6f}")

    def _initialize_sampler(self):
        """Initialize the rejection sampler with the current configuration."""
        if self.model is None:
            raise ValueError("Model must be set before initializing sampler.")

        # For prior sampling mode (epsilon=inf, observed_data=None),
        # create temporary observed data to initialize the sampler
        observed_data_for_sampler = self.observed_data

        if self.observed_data is None and self.epsilon == jnp.inf:
            # Generate temporary observed data with the same shape as model output
            temp_key = random.PRNGKey(42)  # Fixed seed for reproducibility
            temp_theta = self.model.get_prior_sample(temp_key)
            temp_key, sim_key = random.split(temp_key)
            observed_data_for_sampler = self.model.simulate_data(sim_key, temp_theta)

            if self.config.get("verbose", False):
                print("📝 Created temporary observed data for prior sampling mode")

        summary_stat_fn = getattr(self.model, "summary_stat_fn", None)

        # Check if model has transform_phi method
        if hasattr(self.model, "transform_phi"):
            transform_fn = self.model.transform_phi
        else:
            transform_fn = None

        # Calculate observed summary stats for the sampler
        observed_summary_stats_for_sampler = None
        if observed_data_for_sampler is not None and summary_stat_fn is not None:
            observed_summary_stats_for_sampler = summary_stat_fn(
                observed_data_for_sampler
            )

        self.sampler = RejectionSampler(
            prior_simulator=self.model.get_prior_sample,
            data_simulator=self.model.simulate_data,
            discrepancy_fn=self.model.discrepancy_fn,
            summary_stat_fn=summary_stat_fn,
            transform_fn=transform_fn,
            epsilon=self.epsilon,
            observed_data=observed_data_for_sampler,
            observed_summary_stats=observed_summary_stats_for_sampler,
        )

    def update_observed_data(self, observed_data: jnp.ndarray):
        """
        Update the observed data and then reinitialize the sampler.

        This is useful when starting with prior sampling mode and later wanting
        to switch to ABC mode with actual observed data.

        Args:
            observed_data: New observed data
        """
        # RG: This is mostly wrapping a sampler method

        self.observed_data = observed_data
        # Reinitialize summary statistics
        self._init_summary_stats()

        # Update the sampler with new observed data
        if self.sampler is not None:
            summary_stat_fn = getattr(self.model, "summary_stat_fn", None)

            # Calculate new observed summary stats
            new_observed_summary_stats = None
            if summary_stat_fn is not None:
                new_observed_summary_stats = summary_stat_fn(observed_data)

            # Update sampler directly
            self.sampler.observed_data = observed_data
            self.sampler.observed_summary_stats = new_observed_summary_stats
            self.sampler.epsilon = self.epsilon

            if self.config.get("verbose", False):
                print(f"🔄 Updated sampler with new observed data.")

    def update_epsilon(self, epsilon: float):
        """
        Update the epsilon tolerance threshold.

        Args:
            epsilon (float): Epsilon tolerance threshold.

        Raises:
            ValueError: If epsilon is not positive.
        """
        # RG: This is mostly wrapping a sampler method

        if epsilon <= 0:
            raise ValueError("Epsilon must be a positive value.")

        self.epsilon = epsilon
        if self.sampler is not None:
            self.sampler.update_epsilon(epsilon)

    def compute_epsilon_from_quantile(
        self,
        quantile_distance: float,
        n_samples: int = 10000,
        key: Optional[random.PRNGKey] = None,
    ) -> float:
        """
        Compute epsilon value for a given quantile of the distance distribution.

        Args:
            quantile_distance: Quantile level (0 < quantile_distance <= 1)
            n_samples: Number of samples for quantile estimation
            key: JAX random key (if None, uses a fixed seed)

        Returns:
            Computed epsilon value

        Example:
            # Compute epsilon for 95% quantile
            new_epsilon = simulator.compute_epsilon_from_quantile(0.95)
            simulator.update_epsilon(new_epsilon)
        """
        # RG: This is mostly wrapping a sampler method

        if not (0 < quantile_distance <= 1):
            raise ValueError("quantile_distance must be between 0 and 1")

        if key is None:
            key = random.PRNGKey(0)  # Fixed seed for reproducibility

        if quantile_distance == 1.0:
            return float("inf")

        computed_epsilon, _, _ = self.get_epsilon_quantile(
            key, quantile_distance, n_samples
        )

        if self.config.get("verbose", False):
            print(
                f"📊 Computed epsilon = {computed_epsilon:.6f} for {quantile_distance:.1%} quantile"
            )

        return float(computed_epsilon)

    def set_epsilon_from_quantile(
        self,
        quantile_distance: float,
        n_samples: int = 10000,
        key: Optional[random.PRNGKey] = None,
    ):
        """
        Set epsilon based on a quantile of the distance distribution.

        Combines compute_epsilon_from_quantile() and update_epsilon() for convenience.

        Args:
            quantile_distance: Quantile level (0 < quantile_distance <= 1)
            n_samples: Number of samples for quantile estimation
            key: JAX random key (if None, uses a fixed seed)

        Example:
            # Set epsilon to 90% quantile
            simulator.set_epsilon_from_quantile(0.90)
        """

        # RG: This is mostly wrapping a sampler method
        computed_epsilon = self.compute_epsilon_from_quantile(
            quantile_distance, n_samples, key
        )
        self.quantile_distance = quantile_distance  # Update for consistency
        self.update_epsilon(computed_epsilon)

    def generate_samples(self, key: random.PRNGKey, n_samples: int) -> ABCSampleResult:
        """
        Generates multiple ABC samples using vectorized sampling.

        Args:
            key: A JAX random key.
            n_samples: The number of samples to generate.

        Returns:
            An ABCSampleResult named tuple with all sampling results.
        """
        # RG: This is mostly wrapping a sampler method
        if self.sampler is None:
            self._initialize_sampler()
        return self.sampler.sample(key, n_samples)

    def generate_training_samples(
        self, key: random.PRNGKey, n_samples: int
    ) -> ABCTrainingResult:
        """
        Generates a training dataset for Neural Ratio Estimation.

        Uses the model's built-in transform_phi method for theta to phi transformation.

        Args:
            key: A JAX random key.
            n_samples: The total number of samples for the training set.

        Returns:
            An ABCTrainingResult with features, labels, and metadata.
        """
        # RG: This is mostly wrapping a sampler method

        if self.sampler is None:
            self._initialize_sampler()

        return self.sampler.generate_training_samples(key, n_samples)

    def get_epsilon_quantile(
        self, key: random.PRNGKey, alpha: float, n_samples: int = 10000
    ) -> Tuple[float, jnp.ndarray, random.PRNGKey]:
        """
        Gets the epsilon value for a given quantile of the distance distribution.

        Args:
            key: A JAX random key.
            alpha: The quantile level (e.g., 0.1 for the 10th percentile).
            n_samples: Number of simulations to estimate the distribution.

        Returns:
            A tuple of (epsilon_quantile, all_distances, updated_key).
        """
        # RG: This is mostly wrapping a sampler method
        if self.sampler is None:
            self._initialize_sampler()
        return self.sampler.get_epsilon_quantile(key, alpha, n_samples)

    def train_summary_network(
        self,
        key: random.PRNGKey,
        nn_config: Optional[Union[NNConfig, Dict[str, Any]]] = None,
        n_samples_max: int = jnp.inf,
        override_model_summary_stats: bool = True,
    ):
        """
        Train a summary statistics network using the unified training system.

        This method generates training data from the model and uses the unified
        training system to learn optimal summary statistics for the ABC procedure.

        Args:
            key: A JAX random key.
            nn_config: NNConfig instance or dictionary with configuration parameters.
                      If None, creates default configuration based on model properties.
            n_samples_max: Maximum number of samples for training.
            override_model_summary_stats: Whether to override existing summary stats.
        """
        # RG: This should be a standalone function taking a NN and sampler as input.

        # Validation
        if self.model is None:
            raise ValueError("Model must be set before learning summary statistics.")

        if self.sampler is None:
            self._initialize_sampler()

        # Create or validate configuration
        # RG: Require the user to intialize the nn outside this function
        if nn_config is None:
            # Create default configuration based on model properties
            learner_type = "DeepSet" if self.model.sample_is_iid else "MLP"
            # Create default NNConfig for summary learning
            nn_config = get_nn_config(network_name=learner_type, task_type="regressor")

        elif isinstance(nn_config, dict):
            # Convert dictionary to NNConfig
            nn_config = NNConfig.from_dict(nn_config)

        # Ensure task_type is correct
        if nn_config.task_type != "regressor":
            raise ValueError("nn_config.task_type must be 'regressor'")

        # Configure sample stopping rule based on n_samples_max
        if hasattr(nn_config.training, "stopping_rules"):
            # Check if stopping_rules is a dict or StoppingRulesConfig object
            # RG: Why is it possible for it to be both StoppingRulesConfig or Dict?
            if isinstance(nn_config.training.stopping_rules, dict):
                # Working with dictionary format - need to modify it
                if "sample_stopping" not in nn_config.training.stopping_rules:
                    nn_config.training.stopping_rules["sample_stopping"] = {}

                nn_config.training.stopping_rules["sample_stopping"]["enabled"] = True
                nn_config.training.stopping_rules["sample_stopping"][
                    "max_samples"
                ] = n_samples_max

            elif hasattr(nn_config.training.stopping_rules, "sample_stopping"):
                # Working with StoppingRulesConfig object
                nn_config.training.stopping_rules.sample_stopping.enabled = True
                nn_config.training.stopping_rules.sample_stopping.max_samples = (
                    n_samples_max
                )
        else:
            # Create stopping rules if they don't exist
            nn_config.training.stopping_rules = {
                "sample_stopping": {"enabled": True, "max_samples": n_samples_max}
            }

        # Create data generator that matches the expected interface
        def io_generator(key: random.PRNGKey, batch_size: int):
            """Adapter for the unified training interface."""
            results = self.model.sample_phi_x_multiple(key, batch_size)
            phi, x = results
            return {"input": x, "output": phi, "n_simulations": batch_size}

        # Train using the unified system
        key, train_key = random.split(key)
        summary_results = train_regressor(
            key=train_key, config=nn_config, io_generator=io_generator
        )

        # Update configuration
        self.config["summary_stats_enabled"] = True

        # Update the model if it doesn't have a summary_stat_fn or we want to override it
        if not hasattr(self.model, "summary_stat_fn") or override_model_summary_stats:
            if nn_config.training.verbose:
                print("Updating model's summary statistics function...")
            summary_fn = create_summary_stats_fn(
                network=summary_results.network,
                params=summary_results.params,
                network_type=nn_config.network.network_type,
            )
            self.model.summary_stat_fn = summary_fn
            self.summary_stat_fn = summary_fn
            self._summary_params = summary_results.params
            self._summary_config = summary_results.config
            self._summary_network = summary_results.network
            self._summary_training_history = summary_results.training_history

            if self.observed_data is not None:
                self.observed_summary_stats = self.summary_stat_fn(self.observed_data)

        # Reinitialize the sampler to use the new summary statistics function
        if self.sampler is not None:
            self._initialize_sampler()

        self.trained_summary_stats = True

        if nn_config.training.verbose:
            print("Summary statistics learned and updated successfully!")
            print(f"   - Original data dimension: {self.model.data_shape}")
            print(f"   - Learned summary function integrated into model")

        return summary_results

    def check_summary_stats_correlation(
        self, key: random.PRNGKey, n_samples: int = 10000
    ):
        """Check correlation between summary statistics and model phi.
        This method generates samples from the model and computes the correlation
        between the summary statistics and the model parameters.
        """

        # RG: This should be a standalone function

        if not self.config.get("summary_stats_enabled", False):
            raise ValueError(
                "Summary statistics are not enabled. Please learn summary statistics first."
            )

        if self.model is None:
            raise ValueError(
                "Model must be set before checking summary statistics correlation."
            )

        # Generate samples from the model
        key, subkey = random.split(key)
        phi_samples, x_samples = self.model.sample_phi_x_multiple(
            subkey, n_samples=n_samples
        )
        print("Shapes : phi_samples:", phi_samples.shape, "x_samples:", x_samples.shape)
        # Compute summary statistics for the generated samples
        summary_stats = self.model.summary_stat_fn(x_samples)

        # Compute correlation
        print(
            "Shapes: summary_stats:",
            summary_stats.shape,
            "phi_samples:",
            phi_samples.shape,
        )
        correlation_matrix = jnp.corrcoef(
            summary_stats.flatten(), phi_samples.flatten()
        )

        phi_summary_correlation = correlation_matrix[
            : summary_stats.shape[1], summary_stats.shape[1] :
        ]
        if self.config["verbose"]:
            print(
                "Correlation between summary statistics and model parameters:",
                phi_summary_correlation,
            )
        return phi_summary_correlation

    def get_true_posterior_samples(
        self, key: "jax.random.PRNGKey", n_samples: int
    ) -> jnp.ndarray:
        """
        Draws samples from the true analytical posterior, if available.

        This method relies on the underlying model having a method to sample
        from its analytical posterior (e.g., for conjugate models).

        Args:
            key: A JAX random key.
            n_samples: The number of samples to draw.

        Returns:
            An array of samples from the true posterior.

        Raises:
            NotImplementedError: If the model does not support analytical sampling.
        """

        # RG: This is mostly wrapping a model method

        if not hasattr(self.model, "get_posterior_distribution"):
            raise NotImplementedError(
                "The current model does not have a method for analytical posterior sampling."
            )

        seed = int(random.randint(key, (), 0, jnp.iinfo(jnp.int32).max))
        return self.model.get_posterior_distribution(self.observed_data).rvs(
            size=n_samples, random_state=seed
        )

    def __repr__(self) -> str:
        """Provides a clean string representation of the ABCSimulator."""
        model_name = self.model.__class__.__name__ if self.model else "None"
        obs_shape = (
            self.observed_data.shape if self.observed_data is not None else "None"
        )

        return (
            f"ABCSimulator(model={model_name}, "
            f"epsilon={self.epsilon:.4f}, observed_data_shape={obs_shape})"
        )


# Backward compatibility alias
# ABCDataGenerator = ABCSimulator

# Export main components
__all__ = ["ABCSimulator"]
