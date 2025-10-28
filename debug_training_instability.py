#!/usr/bin/env python3
"""
Script de diagnostic pour identifier les problèmes d'instabilité d'entraînement.
Usage: python debug_training_instability.py
"""

import sys
import os

sys.path.append("abcnre/src")

import jax
import jax.numpy as jnp
import numpy as np


def analyze_data_distribution(data, name="data"):
    """Analyse la distribution des données pour identifier les problèmes."""
    print(f"\n🔍 Analyse de {name}:")
    print(f"  Shape: {data.shape}")
    print(f"  Min: {jnp.min(data):.6e}")
    print(f"  Max: {jnp.max(data):.6e}")
    print(f"  Mean: {jnp.mean(data):.6e}")
    print(f"  Std: {jnp.std(data):.6e}")
    print(f"  Has NaN: {jnp.any(jnp.isnan(data))}")
    print(f"  Has Inf: {jnp.any(jnp.isinf(data))}")

    # Vérifier si les données ont une magnitude problématique
    if jnp.abs(jnp.mean(data)) > 1000:
        print(f"  ⚠️  PROBLÈME: Magnitude élevée détectée!")
    if jnp.std(data) > 1000:
        print(f"  ⚠️  PROBLÈME: Variance élevée détectée!")
    if jnp.any(jnp.isinf(data)) or jnp.any(jnp.isnan(data)):
        print(f"  ❌ PROBLÈME CRITIQUE: Valeurs invalides détectées!")


def check_gradient_scale(model, params, inputs, targets):
    """Vérifie l'échelle des gradients."""

    def loss_fn(params):
        predictions = model.apply(params, inputs, training=True)
        return jnp.mean((predictions - targets) ** 2)

    grads = jax.grad(loss_fn)(params)

    # Calculer les statistiques des gradients
    all_grads = []
    for key, grad in jax.tree_util.tree_leaves_with_path(grads):
        all_grads.extend(grad.flatten())

    all_grads = jnp.array(all_grads)

    print(f"\n🎯 Analyse des gradients:")
    print(f"  Gradient min: {jnp.min(all_grads):.6e}")
    print(f"  Gradient max: {jnp.max(all_grads):.6e}")
    print(f"  Gradient mean: {jnp.mean(all_grads):.6e}")
    print(f"  Gradient std: {jnp.std(all_grads):.6e}")

    if jnp.std(all_grads) > 10:
        print(f"  ⚠️  PROBLÈME: Gradients à variance élevée!")
    if jnp.max(jnp.abs(all_grads)) > 100:
        print(f"  ❌ PROBLÈME CRITIQUE: Gradients explosifs détectés!")


def suggest_solutions(data_stats, grad_stats):
    """Suggère des solutions basées sur l'analyse."""
    print(f"\n💡 Suggestions de correction:")

    # Problèmes de données
    if data_stats["high_magnitude"]:
        print("  📊 Données haute magnitude:")
        print("    - Normaliser les données d'entrée (z-score)")
        print("    - Utiliser LayerNorm ou BatchNorm")
        print("    - Réduire le learning rate")

    if data_stats["high_variance"]:
        print("  📈 Variance élevée:")
        print("    - Augmenter la régularisation (weight_decay)")
        print("    - Utiliser dropout")
        print("    - Réduire la complexité du modèle")

    # Problèmes de gradients
    if grad_stats["exploding"]:
        print("  🚀 Gradients explosifs:")
        print("    - Activer gradient clipping")
        print("    - Réduire drastiquement le learning rate")
        print("    - Utiliser LayerNorm")
        print("    - Simplifier l'architecture")


def main():
    """Fonction principale de diagnostic."""
    print("🔧 Diagnostic d'instabilité d'entraînement")
    print("=" * 50)

    try:
        from abcnre.training.networks.mlp import MLP
        from abcnre.training.components.loop import Inputs

        # Simuler des données problématiques
        print("\n🧪 Test avec données simulées...")

        # Données normales
        normal_data = jax.random.normal(jax.random.PRNGKey(42), (32, 100, 10))
        normal_inputs = Inputs(raw=normal_data, permutation_invariant=normal_data)
        normal_targets = jax.random.normal(jax.random.PRNGKey(43), (32, 1))

        analyze_data_distribution(normal_data, "données normales")
        analyze_data_distribution(normal_targets, "targets normales")

        # Données problématiques (haute magnitude)
        problem_data = normal_data * 1000  # Amplifier
        problem_inputs = Inputs(raw=problem_data, permutation_invariant=problem_data)
        problem_targets = normal_targets * 1000

        analyze_data_distribution(problem_data, "données problématiques")
        analyze_data_distribution(problem_targets, "targets problématiques")

        # Test avec MLP
        mlp = MLP(output_dim=1, hidden_dims=[64, 32], use_layer_norm=False)
        key = jax.random.PRNGKey(0)
        params = mlp.init(key, normal_inputs, training=False)

        # Test gradients normaux
        check_gradient_scale(mlp, params, normal_inputs, normal_targets)

        # Test gradients problématiques
        print("\n" + "=" * 30)
        check_gradient_scale(mlp, params, problem_inputs, problem_targets)

        # Suggérer des solutions
        data_stats = {
            "high_magnitude": jnp.abs(jnp.mean(problem_data)) > 1000,
            "high_variance": jnp.std(problem_data) > 1000,
        }
        grad_stats = {"exploding": True}  # Simulé

        suggest_solutions(data_stats, grad_stats)

    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
