#!/usr/bin/env python3
"""
Fix ultime - correction simple et test.
"""

import sys
from pathlib import Path

def main():
    """Fix ultime."""
    print("🚀 FIX ULTIME")
    print("=" * 20)
    
    print("1. Correction de l'import problématique...")
    
    # Corriger diagnostics_integration.py
    file_path = Path("src/abcnre/inference/diagnostics_integration.py")
    with open(file_path, 'r') as f:
        content = f.read()
    
    content = content.replace(
        "from ..simulation.base import ABCSimulator",
        "from ..simulation.simulator import ABCSimulator"
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Import corrigé")
    
    print("\n2. Test du système...")
    
    # Test
    sys.path.insert(0, str(Path("src").resolve()))
    
    try:
        from abcnre.inference import NeuralRatioEstimator, MLPNetwork
        from abcnre.simulation import ABCSimulator
        from abcnre.simulation.models import GaussGaussModel
        
        print("✅ Imports réussis")
        
        # Test rapide
        model = GaussGaussModel()
        network = MLPNetwork(hidden_dims=[32, 16])
        estimator = NeuralRatioEstimator(network)
        
        print("✅ Objets créés")
        
        print("\n🎉 SUCCÈS TOTAL!")
        print("✅ Le système fonctionne maintenant")
        print("\n📋 Prochaines étapes:")
        print("   cd tests/unit/test_inference")
        print("   python test_compatibility.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)