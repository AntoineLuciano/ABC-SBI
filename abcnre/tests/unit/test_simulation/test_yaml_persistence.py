#!/usr/bin/env python3
"""
Test de persistance YAML : créer → sauvegarder → charger → comparer

Ce test vérifie que :
1. On peut créer un simulator et générer des échantillons
2. On peut sauvegarder la configuration complète en YAML
3. On peut charger depuis YAML et recréer un simulator identique
4. Les deux simulators génèrent des distributions identiques
"""

import sys
from pathlib import Path
import tempfile
import os

# Add src path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root / "src"))

import jax.numpy as jnp
from jax import random
import numpy as np

from abcnre.simulation.models import GaussGaussModel
from abcnre.simulation import ABCSimulator

def test_yaml_persistence_cycle():
    """Test complet du cycle de persistance YAML."""
    
    print("🔄 Test de persistance YAML - Cycle complet")
    print("=" * 60)
    
    # Créer un répertoire temporaire pour les tests
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_simulator.yml"
        data_path = Path(temp_dir) / "observed_data.npy"
        
        # 1. CRÉATION - Créer des données et un simulator original
        print("\n1. 📋 Création du simulator original...")
        
        key = random.PRNGKey(42)
        true_theta = 2.5
        observed_data = true_theta + 0.2 * random.normal(key, shape=(80,))
        
        print(f"   True theta: {true_theta}")
        print(f"   Observed data: {observed_data.shape}, mean={jnp.mean(observed_data):.3f}")
        
        # Sauvegarder les données observées 
        np.save(data_path, np.array(observed_data))
        print(f"   Données sauvées: {data_path}")
        
        # Créer le modèle et simulator original
        model_original = GaussGaussModel(mu0=1.0, sigma0=3.0, sigma=0.2)
        simulator_original = ABCSimulator(
            model=model_original,
            observed_data=observed_data,
            quantile_distance=0.05  # 5% quantile
        )
        
        print(f"   Simulator original créé")
        print(f"   Epsilon calculé: {simulator_original.epsilon:.6f}")
        print(f"   Config: {simulator_original.get_config()}")
        
        # 2. SIMULATION - Générer des échantillons avec le simulator original
        print("\n2. 🎲 Génération d'échantillons originaux...")
        
        key, subkey = random.split(key)
        samples_original = simulator_original.generate_samples(subkey, n_samples=10000)
        
        original_mean = jnp.mean(samples_original.theta_samples)
        original_std = jnp.std(samples_original.theta_samples)
        
        print(f"   Échantillons générés: {samples_original.theta_samples.shape[0]}")
        print(f"   Posterior mean: {original_mean:.3f}")
        print(f"   Posterior std: {original_std:.3f}")
        print(f"   Distance moyenne: {jnp.mean(samples_original.distances):.6f}")
        
        # 3. SAUVEGARDE - Sauvegarder en YAML
        print("\n3. 💾 Sauvegarde en YAML...")
        
        # Modifier la config pour pointer vers le fichier de données
        simulator_original.save_configuration(config_path)
        
        # Lire et modifier le YAML pour inclure le path des données
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Ajouter le path des données observées
        config_data['observed_data_path'] = str(data_path)
        config_data['metadata'] = {
            'created_by': 'test_yaml_persistence',
            'true_theta': float(true_theta),
            'description': 'Test configuration for YAML persistence'
        }
        
        # Réécrire le YAML complet
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        print(f"   Configuration sauvée: {config_path}")
        print(f"   Taille du fichier: {config_path.stat().st_size} bytes")
        
        # Afficher le contenu du YAML
        print("\n   📄 Contenu du YAML:")
        with open(config_path, 'r') as f:
            yaml_content = f.read()
            for i, line in enumerate(yaml_content.split('\n')[:15], 1):
                print(f"      {i:2d}: {line}")
            if len(yaml_content.split('\n')) > 15:
                n_lines = len(yaml_content.split('\n')) - 15
                print(f"      ... ({n_lines} lignes supplémentaires)")
                # print(f"      ... ({len(yaml_content.split('\n'))-15} lignes supplémentaires)")
        
        # 4. CHARGEMENT - Créer un nouveau simulator depuis YAML
        print("\n4. 📂 Chargement depuis YAML...")
        
        # Charger la configuration
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        print(f"   Configuration chargée: {len(loaded_config)} clés")
        
        # Recréer le modèle
        from abcnre.simulation.utils import import_class_from_string
        model_class = import_class_from_string(loaded_config['model_class'])
        model_args = loaded_config['model_args']
        model_loaded = model_class(**model_args)
        
        print(f"   Modèle recréé: {model_loaded}")
        
        # Charger les données observées
        loaded_observed_data = jnp.array(np.load(loaded_config['observed_data_path']))
        print(f"   Données chargées: {loaded_observed_data.shape}")
        
        # Recréer le simulator
        sim_config = loaded_config['simulator_config']
        simulator_loaded = ABCSimulator(
            model=model_loaded,
            observed_data=loaded_observed_data,
            epsilon=sim_config['epsilon'],  # Utiliser l'epsilon sauvé
            config=sim_config['config']
        )
        
        print(f"   Simulator recréé")
        print(f"   Epsilon chargé: {simulator_loaded.epsilon:.6f}")
        
        # 5. VÉRIFICATION - Comparer les configurations
        print("\n5. 🔍 Vérification des configurations...")
        
        # Comparer les epsilons
        epsilon_diff = abs(simulator_original.epsilon - simulator_loaded.epsilon)
        print(f"   Epsilon original: {simulator_original.epsilon:.6f}")
        print(f"   Epsilon chargé:   {simulator_loaded.epsilon:.6f}")
        print(f"   Différence:       {epsilon_diff:.8f}")
        assert epsilon_diff < 1e-6, f"Epsilons différents: {epsilon_diff}"
        
        # Comparer les modèles
        orig_args = simulator_original.model.get_model_args()
        loaded_args = simulator_loaded.model.get_model_args()
        print(f"   Args originaux: {orig_args}")
        print(f"   Args chargés:   {loaded_args}")
        assert orig_args == loaded_args, "Arguments du modèle différents"
        
        # Comparer les données observées
        data_diff = jnp.max(jnp.abs(simulator_original.observed_data - simulator_loaded.observed_data))
        print(f"   Différence max données: {data_diff:.8f}")
        assert data_diff < 1e-6, "Données observées différentes"
        
        print("   ✅ Configurations identiques!")
        
        # 6. SIMULATION COMPARATIVE - Générer des échantillons avec le simulator chargé
        print("\n6. 🎲 Génération d'échantillons depuis YAML...")
        
        # Utiliser la même seed pour comparaison équitable
        key_comp = random.PRNGKey(42)  # Même seed que l'original
        _, subkey_comp = random.split(key_comp)
        samples_loaded = simulator_loaded.generate_samples(subkey_comp, n_samples=10000)
        
        loaded_mean = jnp.mean(samples_loaded.theta_samples)
        loaded_std = jnp.std(samples_loaded.theta_samples)
        
        print(f"   Échantillons générés: {samples_loaded.theta_samples.shape[0]}")
        print(f"   Posterior mean: {loaded_mean:.3f}")
        print(f"   Posterior std: {loaded_std:.3f}")
        
        print(f"   Distance moyenne: {jnp.mean(samples_loaded.distances):.6f}")
        
        # 7. COMPARAISON STATISTIQUE
        print("\n7. 📊 Comparaison statistique...")
        
        mean_diff = abs(original_mean - loaded_mean)
        std_diff = abs(original_std - loaded_std)
        
        print(f"   Différence means: {mean_diff:.6f}")
        print(f"   Différence stds:  {std_diff:.6f}")
        
        # Test de Kolmogorov-Smirnov pour comparer les distributions
        from scipy import stats
        ks_stat, ks_pvalue = stats.ks_2samp(
            np.array(samples_original.theta_samples), 
            np.array(samples_loaded.theta_samples)
        )
        
        print(f"   KS test statistic: {ks_stat:.6f}")
        print(f"   KS test p-value:   {ks_pvalue:.6f}")
        
        # 8. VISUALISATION - Comparer les KDE plots
        print("\n8. 📈 Génération des plots comparatifs...")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Test de Persistance YAML - Comparaison des Distributions', fontsize=14, fontweight='bold')
            
            # Plot 1: KDE plots superposés
            ax1 = axes[0]
            sns.kdeplot(samples_original.theta_samples, ax=ax1, label='Original', linewidth=2, color='blue')
            sns.kdeplot(samples_loaded.theta_samples, ax=ax1, label='Chargé depuis YAML', linewidth=2, color='red', linestyle='--')
            ax1.axvline(true_theta, color='green', linestyle=':', linewidth=2, label='True θ')
            ax1.set_xlabel('θ (parameter)')
            ax1.set_ylabel('Density')
            ax1.set_title('Superposition des Distributions')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Histogrammes côte à côte
            ax2 = axes[1]
            bins = np.linspace(min(np.min(samples_original.theta_samples), np.min(samples_loaded.theta_samples)),
                              max(np.max(samples_original.theta_samples), np.max(samples_loaded.theta_samples)), 30)
            ax2.hist(samples_original.theta_samples, bins=bins, alpha=0.7, label='Original', color='blue')
            ax2.hist(samples_loaded.theta_samples, bins=bins, alpha=0.7, label='Chargé', color='red')
            ax2.axvline(true_theta, color='green', linestyle=':', linewidth=2, label='True θ')
            ax2.set_xlabel('θ (parameter)')
            ax2.set_ylabel('Count')
            ax2.set_title('Histogrammes Comparatifs')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Q-Q plot
            ax3 = axes[2]
            stats.probplot(samples_original.theta_samples, dist="norm", plot=ax3, rvalue=True)
            ax3.set_title('Q-Q Plot (Original vs Normal)')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Sauvegarder dans le répertoire temporaire ET localement
            plot_path_temp = Path(temp_dir) / "yaml_persistence_test.png"
            plot_path_local = "yaml_persistence_test.png"
            
            plt.savefig(plot_path_temp, dpi=300, bbox_inches='tight')
            plt.savefig(plot_path_local, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"   📊 Plot sauvé: {plot_path_local}")
            
        except ImportError:
            print("   ⚠️  matplotlib/seaborn non disponible pour les plots")
        
        # 9. RÉSULTATS FINAUX
        print("\n9. 🎯 Résultats finaux...")
        
        print(f"   Configuration YAML: ✅ Identique")
        print(f"   Epsilon: ✅ Identique ({epsilon_diff:.2e})")
        print(f"   Modèle: ✅ Identique")
        print(f"   Données: ✅ Identiques ({data_diff:.2e})")
        print(f"   Distributions: {'✅ Similaires' if ks_pvalue > 0.05 else '⚠️  Différentes'} (p={ks_pvalue:.3f})")
        
        # Tests d'assertion finaux
        assert epsilon_diff < 1e-6, "Epsilon non préservé"
        assert data_diff < 1e-6, "Données non préservées"
        assert mean_diff < 0.1, f"Moyennes trop différentes: {mean_diff}"
        assert std_diff < 0.1, f"Écarts-types trop différents: {std_diff}"
        
        print(f"\n🎉 Test de persistance YAML: SUCCÈS!")
        print(f"   Le cycle créer → sauver → charger → simuler fonctionne parfaitement!")
        
        return True


if __name__ == "__main__":
    test_yaml_persistence_cycle()
