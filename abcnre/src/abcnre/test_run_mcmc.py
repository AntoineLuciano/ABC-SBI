#!/usr/bin/env python3
"""
Script de test pour la commande run_mcmc
"""

import sys
import os

# Ajouter le répertoire parent au path Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Test d'import
try:
    from abcnre.cli.run_mcmc import setup_run_mcmc_parser

    print("✅ Import du module run_mcmc réussi")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

# Test de configuration du parser
try:
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_run_mcmc_parser(subparsers)
    print("✅ Configuration du parser réussie")
except Exception as e:
    print(f"❌ Erreur de configuration: {e}")
    sys.exit(1)

# Test d'affichage de l'aide
try:
    help_text = parser.format_help()
    print("✅ Génération de l'aide réussie")
    print("\n=== AIDE GÉNÉRÉE ===")
    print(help_text)
except Exception as e:
    print(f"❌ Erreur de génération d'aide: {e}")
    sys.exit(1)

print("\n🎉 Tous les tests sont réussis!")
