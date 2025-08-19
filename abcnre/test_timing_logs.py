#!/usr/bin/env python3
"""
Script de test pour valider les améliorations de timing et logging.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import jax
import jax.numpy as jnp
from abcnre.training.config import get_nn_config
from abcnre.training.train import train_regressor


def create_test_io_generator():
    """Générateur de données simple pour les tests."""

    def io_generator(key, batch_size):
        key_theta, key_x = jax.random.split(key)
        theta = jax.random.normal(key_theta, (batch_size, 2))
        x = (
            theta @ jnp.array([1.0, 0.5])
            + jax.random.normal(key_x, (batch_size,)) * 0.1
        )
        return {
            "input": {"theta": theta},
            "output": x.reshape(-1, 1),
            "n_simulations": batch_size,
        }

    return io_generator


def test_timing_logs():
    """Test des logs de timing améliorés."""
    print("🧪 Test des logs de timing améliorés...")

    key = jax.random.PRNGKey(42)

    # Configuration avec données pré-simulées
    config = get_nn_config(
        network_name="mlp",
        network_size="default",
        training_size="fast",  # Training rapide pour le test
        task_type="regressor",
        training_set_size=1000,  # Petit dataset pour test rapide
        validation_set_size=200,
    )

    print(
        f"✅ Configuration créée avec use_presimulated_data: {config.training.use_presimulated_data}"
    )
    print(f"   Training set size: {config.training.training_set_size}")
    print(f"   Validation set size: {config.training.validation_set_size}")

    # Créer un io_generator
    io_generator = create_test_io_generator()

    print("\n🚀 Démarrage de l'entraînement avec logs de timing détaillés...")

    # Entraîner avec logs détaillés
    result = train_regressor(key=key, config=config, io_generator=io_generator)

    print("\n📊 Analyse des résultats de timing:")
    history = result.training_history

    if "total_simulation_time" in history:
        total_time = history.get("total_simulation_time", 0) + history.get(
            "total_training_time", 0
        )
        sim_time = history.get("total_simulation_time", 0)
        train_time = history.get("total_training_time", 0)

        print(f"   Total: {total_time:.2f}s")
        print(f"   Simulation: {sim_time:.2f}s ({sim_time/total_time*100:.1f}%)")
        print(f"   Entraînement: {train_time:.2f}s ({train_time/total_time*100:.1f}%)")

        if "avg_sim_to_train_ratio" in history:
            print(f"   Ratio sim/train: {history['avg_sim_to_train_ratio']:.2f}x")

    print("\n✅ Test terminé avec succès!")
    return result


if __name__ == "__main__":
    test_timing_logs()
