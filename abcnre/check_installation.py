#!/usr/bin/env python3
"""
Script de diagnostic pour vérifier l'installation du module d'inférence.

Usage: python check_installation.py
"""

import sys
import os
from pathlib import Path

def check_file_structure():
    """Vérifie la structure des fichiers."""
    print("🔍 Vérification de la structure des fichiers...")
    
    # Déterminer le répertoire racine
    current_dir = Path.cwd()
    print(f"📍 Répertoire actuel: {current_dir}")
    
    # Chercher le répertoire src
    possible_roots = [
        current_dir,
        current_dir.parent,
        current_dir.parent.parent,
        current_dir.parent.parent.parent
    ]
    
    src_dir = None
    for root in possible_roots:
        if (root / "src" / "abcnre").exists():
            src_dir = root / "src" / "abcnre"
            break
    
    if src_dir is None:
        print("❌ Impossible de trouver le répertoire src/abcnre")
        print("💡 Assurez-vous d'être dans le bon répertoire de projet")
        return False
    
    print(f"✅ Répertoire src trouvé: {src_dir}")
    
    # Vérifier les fichiers networks
    networks_dir = src_dir / "inference" / "networks"
    network_files = [
        "base.py",
        "mlp.py", 
        "deepset.py",
        "__init__.py"
    ]
    
    print(f"\n📁 Vérification des fichiers networks dans {networks_dir}:")
    networks_ok = True
    for file in network_files:
        file_path = networks_dir / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MANQUANT")
            networks_ok = False
    
    # Vérifier les fichiers du module d'inférence
    inference_dir = src_dir / "inference"
    inference_files = [
        "__init__.py",
        "estimator.py",
        "trainer.py",
        "utils.py",
        "config.py"
    ]
    
    print(f"\n📁 Vérification des fichiers d'inférence dans {inference_dir}:")
    inference_ok = True
    for file in inference_files:
        file_path = inference_dir / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MANQUANT")
            inference_ok = False
    
    return networks_ok and inference_ok

def check_dependencies():
    """Vérifie les dépendances."""
    print("\n📦 Vérification des dépendances...")
    
    deps = [
        ("jax", "JAX"),
        ("flax", "Flax"),
        ("optax", "Optax"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("sklearn", "Scikit-learn")
    ]
    
    missing = []
    for module, name in deps:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - MANQUANT")
            missing.append(name)
    
    if missing:
        print(f"\n💡 Pour installer les dépendances manquantes:")
        print(f"pip install {' '.join(missing.lower().split())}")
    
    return len(missing) == 0

def test_basic_imports():
    """Test les imports de base."""
    print("\n🧪 Test des imports de base...")
    
    # Ajouter le chemin src au sys.path
    current_dir = Path.cwd()
    for root in [current_dir, current_dir.parent, current_dir.parent.parent, current_dir.parent.parent.parent]:
        src_path = root / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
            print(f"📍 Ajouté au PATH: {src_path}")
            break
    
    # Test import networks
    try:
        from abcnre.inference.networks.base import NetworkBase
        print("✅ Import NetworkBase réussi")
    except ImportError as e:
        print(f"❌ Import NetworkBase échoué: {e}")
        return False
    
    try:
        from abcnre.inference.networks.mlp import MLPNetwork
        print("✅ Import MLPNetwork réussi")
    except ImportError as e:
        print(f"❌ Import MLPNetwork échoué: {e}")
        return False
    
    try:
        from abcnre.inference.networks.deepset import DeepSetNetwork
        print("✅ Import DeepSetNetwork réussi")
    except ImportError as e:
        print(f"❌ Import DeepSetNetwork échoué: {e}")
        return False
    
    # Test création d'un network
    try:
        network = MLPNetwork(hidden_dims=[32, 16])
        print("✅ Création MLPNetwork réussie")
    except Exception as e:
        print(f"❌ Création MLPNetwork échouée: {e}")
        return False
    
    # Test configuration
    try:
        config = network.get_config()
        print(f"✅ Configuration récupérée: {config}")
    except Exception as e:
        print(f"❌ Récupération configuration échouée: {e}")
        return False
    
    return True

def provide_instructions():
    """Fournit des instructions de correction."""
    print("\n🔧 Instructions de correction:")
    print("\n1. Vérifiez que vous êtes dans le bon répertoire:")
    print("   cd /path/to/your/abcnre/project")
    
    print("\n2. Créez la structure de répertoires:")
    print("   mkdir -p src/abcnre/inference/networks")
    print("   mkdir -p tests/unit/test_inference")
    
    print("\n3. Copiez les fichiers networks corrigés:")
    print("   cp base_fixed.py src/abcnre/inference/networks/base.py")
    print("   cp mlp_fixed.py src/abcnre/inference/networks/mlp.py")
    print("   cp deepset_fixed.py src/abcnre/inference/networks/deepset.py")
    print("   cp networks_init.py src/abcnre/inference/networks/__init__.py")
    
    print("\n4. Copiez les fichiers du module d'inférence:")
    print("   cp estimator.py src/abcnre/inference/")
    print("   cp trainer.py src/abcnre/inference/")
    print("   cp utils.py src/abcnre/inference/")
    print("   cp config.py src/abcnre/inference/")
    print("   cp inference_init.py src/abcnre/inference/__init__.py")
    
    print("\n5. Copiez le test corrigé:")
    print("   cp test_compatibility_fixed.py tests/unit/test_inference/test_compatibility.py")
    
    print("\n6. Exécutez le test depuis le répertoire racine:")
    print("   cd tests/unit/test_inference")
    print("   python test_compatibility.py")

def main():
    """Fonction principale."""
    print("🚀 Diagnostic d'installation du module d'inférence ABCNRE")
    print("=" * 60)
    
    # Vérifications
    structure_ok = check_file_structure()
    deps_ok = check_dependencies()
    imports_ok = test_basic_imports()
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DU DIAGNOSTIC")
    print("=" * 60)
    
    print(f"📁 Structure des fichiers: {'✅ OK' if structure_ok else '❌ PROBLÈME'}")
    print(f"📦 Dépendances: {'✅ OK' if deps_ok else '❌ PROBLÈME'}")
    print(f"🧪 Imports de base: {'✅ OK' if imports_ok else '❌ PROBLÈME'}")
    
    if structure_ok and deps_ok and imports_ok:
        print("\n🎉 INSTALLATION COMPLÈTE ET FONCTIONNELLE!")
        print("✅ Vous pouvez maintenant exécuter les tests de compatibilité")
        print("\n🚀 Commande suivante:")
        print("   cd tests/unit/test_inference")
        print("   python test_compatibility.py")
    else:
        print("\n❌ DES PROBLÈMES ONT ÉTÉ DÉTECTÉS")
        provide_instructions()

if __name__ == "__main__":
    main()