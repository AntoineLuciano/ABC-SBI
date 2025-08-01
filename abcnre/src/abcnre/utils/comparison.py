"""
Comparison utilities for ABC-SBI objects.

This module provides functions to compare models, simulators, estimators,
and networks for equivalence. Useful for testing I/O operations and
ensuring data integrity.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)


def deep_compare_dict(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
    rtol: float = 1e-7,
    atol: float = 1e-10,
) -> bool:
    """
    Compare two dictionaries recursively with numerical tolerance.

    Args:
        dict1: First dictionary to compare
        dict2: Second dictionary to compare
        rtol: Relative tolerance for numerical comparisons
        atol: Absolute tolerance for numerical comparisons

    Returns:
        True if dictionaries are equivalent within tolerance
    """
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    for key in dict1:
        val1, val2 = dict1[key], dict2[key]

        if isinstance(val1, dict) and isinstance(val2, dict):
            if not deep_compare_dict(val1, val2, rtol, atol):
                return False
        elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            if not _compare_sequences(val1, val2, rtol, atol):
                return False
        elif isinstance(val1, jnp.ndarray) and isinstance(val2, jnp.ndarray):
            if not jnp.allclose(val1, val2, rtol=rtol, atol=atol):
                return False
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.allclose(val1, val2, rtol=rtol, atol=atol):
                return False
        elif val1 != val2:
            return False

    return True


def _compare_sequences(
    seq1: Union[list, tuple], seq2: Union[list, tuple], rtol: float, atol: float
) -> bool:
    """Compare two sequences (lists or tuples) with numerical tolerance."""
    if len(seq1) != len(seq2):
        return False

    for item1, item2 in zip(seq1, seq2):
        if isinstance(item1, (jnp.ndarray, np.ndarray)) and isinstance(
            item2, (jnp.ndarray, np.ndarray)
        ):
            if isinstance(item1, jnp.ndarray):
                if not jnp.allclose(item1, item2, rtol=rtol, atol=atol):
                    return False
            else:
                if not np.allclose(item1, item2, rtol=rtol, atol=atol):
                    return False
        elif isinstance(item1, dict) and isinstance(item2, dict):
            if not deep_compare_dict(item1, item2, rtol, atol):
                return False
        elif item1 != item2:
            return False

    return True


def compare_network_params(
    params1: Any, params2: Any, rtol: float = 1e-7, atol: float = 1e-10
) -> bool:
    """
    Compare network parameters (JAX PyTrees) for equivalence.

    Args:
        params1: First set of network parameters (PyTree)
        params2: Second set of network parameters (PyTree)
        rtol: Relative tolerance for numerical comparisons
        atol: Absolute tolerance for numerical comparisons

    Returns:
        True if parameters are equivalent within tolerance
    """
    try:
        # Check if both are None
        if params1 is None and params2 is None:
            return True
        if (params1 is None) != (params2 is None):
            return False

        # Compare PyTree structures
        tree_def1 = jax.tree_util.tree_structure(params1)
        tree_def2 = jax.tree_util.tree_structure(params2)

        if tree_def1 != tree_def2:
            return False

        # Compare leaves (parameter values)
        leaves1 = jax.tree_util.tree_leaves(params1)
        leaves2 = jax.tree_util.tree_leaves(params2)

        return all(
            jnp.allclose(l1, l2, rtol=rtol, atol=atol)
            for l1, l2 in zip(leaves1, leaves2)
        )

    except Exception as e:
        logger.warning(f"Error comparing network params: {e}")
        return False


def are_models_equivalent(model1: Any, model2: Any) -> bool:
    """
    Compare two statistical models for equivalence.

    Args:
        model1: First model to compare
        model2: Second model to compare

    Returns:
        True if models are equivalent
    """
    # Check same class
    if type(model1) != type(model2):
        return False

    # Compare model arguments
    try:
        args1 = model1.get_model_args()
        args2 = model2.get_model_args()
        return deep_compare_dict(args1, args2)
    except Exception as e:
        logger.warning(f"Error comparing models: {e}")
        return False


def are_nnconfigs_equivalent(config1: Any, config2: Any) -> bool:
    """
    Compare two NNConfig objects for equivalence.

    Args:
        config1: First NNConfig to compare
        config2: Second NNConfig to compare

    Returns:
        True if configurations are equivalent
    """
    # NNConfig is a dataclass, so == works directly
    return config1 == config2


def are_networks_equivalent(net1: Any, net2: Any) -> bool:
    """
    Compare two networks for equivalence (configuration + parameters if trained).

    Args:
        net1: First network to compare
        net2: Second network to compare

    Returns:
        True if networks are equivalent
    """
    # Compare configurations if available
    if hasattr(net1, "config") and hasattr(net2, "config"):
        if not are_nnconfigs_equivalent(net1.config, net2.config):
            return False

    # Compare trained parameters if both networks have them
    if hasattr(net1, "params") and hasattr(net2, "params"):
        return compare_network_params(net1.params, net2.params)

    # If only one has parameters, they're not equivalent
    if hasattr(net1, "params") != hasattr(net2, "params"):
        return False

    return True


def are_simulators_equivalent(sim1: Any, sim2: Any) -> bool:
    """
    Compare two ABC simulators for equivalence.

    Args:
        sim1: First simulator to compare
        sim2: Second simulator to compare

    Returns:
        True if simulators are equivalent
    """
    # 1. Compare models
    if not are_models_equivalent(sim1.model, sim2.model):
        return False

    # 2. Compare observed data
    if not jnp.allclose(sim1.observed_data, sim2.observed_data):
        return False

    # 3. Compare epsilon
    if sim1.epsilon != sim2.epsilon:
        return False

    # 4. Compare summary network if both have trained summary stats
    if hasattr(sim1, "_summary_config") and hasattr(sim2, "_summary_config"):
        if sim1._summary_config is None and sim2._summary_config is None:
            pass  # Both don't have summary networks - OK
        elif sim1._summary_config is None or sim2._summary_config is None:
            return False  # Only one has summary network
        else:
            # Both have summary networks - compare them
            if not are_nnconfigs_equivalent(sim1._summary_config, sim2._summary_config):
                return False

            # Compare summary network parameters if both are trained
            if hasattr(sim1, "_summary_params") and hasattr(sim2, "_summary_params"):
                if not compare_network_params(
                    sim1._summary_params, sim2._summary_params
                ):
                    return False

    return True


def are_estimators_equivalent(est1: Any, est2: Any) -> bool:
    """
    Compare two neural ratio estimators for equivalence.

    Args:
        est1: First estimator to compare
        est2: Second estimator to compare

    Returns:
        True if estimators are equivalent
    """
    # 1. Compare simulators
    if not are_simulators_equivalent(est1.simulator, est2.simulator):
        return False

    # 2. Compare classifier configurations
    if not are_nnconfigs_equivalent(est1.nn_config, est2.nn_config):
        return False

    # 3. Compare trained classifier parameters if available
    if hasattr(est1, "params") and hasattr(est2, "params"):
        if not compare_network_params(est1.params, est2.params):
            return False
    elif hasattr(est1, "params") != hasattr(est2, "params"):
        return False  # Only one is trained

    # 4. Compare stored phis if available
    if hasattr(est1, "stored_phis") and hasattr(est2, "stored_phis"):
        if est1.stored_phis is None and est2.stored_phis is None:
            pass  # Both don't have stored phis
        elif est1.stored_phis is None or est2.stored_phis is None:
            return False  # Only one has stored phis
        else:
            # Both have stored phis - compare them
            if not jnp.allclose(est1.stored_phis, est2.stored_phis):
                return False

    return True


# Convenience function for testing I/O operations
def test_io_roundtrip(
    original_obj: Any,
    save_func: callable,
    load_func: callable,
    temp_path: str,
    comparison_func: callable,
) -> bool:
    """
    Test that save/load operations preserve object equivalence.

    Args:
        original_obj: Original object to test
        save_func: Function to save the object
        load_func: Function to load the object
        temp_path: Temporary file path for testing
        comparison_func: Function to compare original and loaded objects

    Returns:
        True if roundtrip preserves equivalence
    """
    try:
        # Save object
        save_func(original_obj, temp_path, overwrite=True)

        # Load object
        loaded_obj = load_func(temp_path)

        # Compare
        return comparison_func(original_obj, loaded_obj)

    except Exception as e:
        logger.error(f"I/O roundtrip test failed: {e}")
        return False


__all__ = [
    "are_models_equivalent",
    "are_networks_equivalent",
    "are_simulators_equivalent",
    "are_estimators_equivalent",
    "are_nnconfigs_equivalent",
    "compare_network_params",
    "deep_compare_dict",
    "test_io_roundtrip",
]
