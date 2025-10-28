#!/usr/bin/env python3
"""
Test de l'invariance par permutation pour MLP vs DeepSet
"""

import sys
import os
sys.path.append('abcnre/src')

import jax
import jax.numpy as jnp
import numpy as np

def test_permutation_invariance():
    """Test si MLP et DeepSet sont invariants par permutation."""
    
    try:
        from abcnre.training.networks.mlp import MLP
        from abcnre.training.networks.deepset import DeepSet
        from abcnre.training.components.loop import Inputs
        
        print("🧪 Test d'invariance par permutation")
        print("=" * 50)
        
        # Créer des données test (similaires à G&K)
        key = jax.random.PRNGKey(42)
        batch_size = 4
        n_samples = 100
        feature_dim = 1
        
        # Données originales
        data_original = jax.random.normal(key, (batch_size, n_samples, feature_dim))
        inputs_original = Inputs(raw=data_original, permutation_invariant=data_original)
        
        # Données permutées (même contenu, ordre différent)
        perm_key = jax.random.PRNGKey(123)
        permutation = jax.random.permutation(perm_key, n_samples)
        data_permuted = data_original[:, permutation, :]
        inputs_permuted = Inputs(raw=data_permuted, permutation_invariant=data_permuted)
        
        print(f"📊 Données test:")
        print(f"  Shape: {data_original.shape}")
        print(f"  Original mean: {jnp.mean(data_original):.6f}")
        print(f"  Permuted mean: {jnp.mean(data_permuted):.6f}")
        print(f"  Données identiques (contenu): {jnp.allclose(jnp.sort(data_original, axis=1), jnp.sort(data_permuted, axis=1))}")
        
        # Test MLP
        print(f"\n🔧 Test MLP:")
        mlp = MLP(
            output_dim=1, 
            hidden_dims=[64, 32], 
            sequence_pooling='flatten'  # Force flatten pour voir le problème
        )
        
        mlp_key = jax.random.PRNGKey(0)
        mlp_params = mlp.init(mlp_key, inputs_original, training=False)
        
        mlp_out_original = mlp.apply(mlp_params, inputs_original, training=False)
        mlp_out_permuted = mlp.apply(mlp_params, inputs_permuted, training=False)
        
        mlp_diff = jnp.mean(jnp.abs(mlp_out_original - mlp_out_permuted))
        
        print(f"  Output original: {mlp_out_original.flatten()}")
        print(f"  Output permuté:  {mlp_out_permuted.flatten()}")
        print(f"  Différence moyenne: {mlp_diff:.6f}")
        
        if mlp_diff > 1e-6:
            print(f"  ❌ MLP n'est PAS invariant par permutation!")
        else:
            print(f"  ✅ MLP est invariant par permutation")
            
        # Test MLP avec pooling
        print(f"\n🔧 Test MLP avec Mean Pooling:")
        mlp_pool = MLP(
            output_dim=1, 
            hidden_dims=[64, 32], 
            sequence_pooling='mean'  # Force mean pooling
        )
        
        mlp_pool_params = mlp_pool.init(mlp_key, inputs_original, training=False)
        
        mlp_pool_out_original = mlp_pool.apply(mlp_pool_params, inputs_original, training=False)
        mlp_pool_out_permuted = mlp_pool.apply(mlp_pool_params, inputs_permuted, training=False)
        
        mlp_pool_diff = jnp.mean(jnp.abs(mlp_pool_out_original - mlp_pool_out_permuted))
        
        print(f"  Output original: {mlp_pool_out_original.flatten()}")
        print(f"  Output permuté:  {mlp_pool_out_permuted.flatten()}")
        print(f"  Différence moyenne: {mlp_pool_diff:.6f}")
        
        if mlp_pool_diff > 1e-6:
            print(f"  ❌ MLP+Pooling n'est PAS invariant!")
        else:
            print(f"  ✅ MLP+Pooling est invariant!")
        
        # Test DeepSet
        print(f"\n🔧 Test DeepSet:")
        deepset = DeepSet(
            output_dim=1,
            phi_hidden_dims=[64, 32],
            rho_hidden_dims=[32, 16],
            pooling_type='mean'
        )
        
        deepset_key = jax.random.PRNGKey(1)
        deepset_params = deepset.init(deepset_key, inputs_original, training=False)
        
        deepset_out_original = deepset.apply(deepset_params, inputs_original, training=False)
        deepset_out_permuted = deepset.apply(deepset_params, inputs_permuted, training=False)
        
        deepset_diff = jnp.mean(jnp.abs(deepset_out_original - deepset_out_permuted))
        
        print(f"  Output original: {deepset_out_original.flatten()}")
        print(f"  Output permuté:  {deepset_out_permuted.flatten()}")
        print(f"  Différence moyenne: {deepset_diff:.6f}")
        
        if deepset_diff > 1e-6:
            print(f"  ❌ DeepSet n'est PAS invariant!")
        else:
            print(f"  ✅ DeepSet est invariant!")
        
        # Conclusion
        print(f"\n💡 CONCLUSION:")
        print(f"=" * 50)
        
        if mlp_diff > 1e-6:
            print(f"❌ MLP (flatten) échoue car il est sensible à l'ordre")
            print(f"   → Pour données i.i.d comme G&K, l'ordre est artificiel")
            print(f"   → Le réseau apprend des patterns d'ordre inexistants")
        
        if mlp_pool_diff < 1e-6:
            print(f"✅ MLP+Pooling réussit car il ignore l'ordre")
            print(f"   → Mean pooling = invariance par permutation")
        
        print(f"✅ DeepSet réussit par design (invariance garantie)")
        print(f"   → Architecture spécialement conçue pour données i.i.d")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_permutation_invariance()
