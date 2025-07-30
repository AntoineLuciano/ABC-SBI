#!/usr/bin/env python3
"""
Script de test rapide pour vérifier que les notebooks 2D marginaux peuvent fonctionner.
Test sans exécution complète des notebooks.
"""

import sys
from pathlib import Path
import yaml

# Ajouter le chemin vers le package
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "abcnre" / "src"))


def test_config_loading():
    """Test le chargement de la configuration 2D marginal1"""
    print("=== Test 1: Configuration Loading ===")

    config_path = (
        project_root
        / "abcnre"
        / "examples"
        / "configs"
        / "models"
        / "gauss_gauss_2d_marginal1.yml"
    )
    print(f"Chemin config: {config_path}")
    print(f"Config existe: {config_path.exists()}")

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        print("Configuration chargée:")
        print(f"  - Type: {config['model_type']}")
        print(f"  - Classe: {config['model_class']}")
        print(f"  - Dimension: {config['model_args']['dim']}")
        print(f"  - Marginal d'intérêt: {config['model_args']['marginal_of_interest']}")
        print(f"  - Nombre d'observations: {config['model_args']['n_obs']}")
        return True
    return False


def test_model_import():
    """Test l'import du modèle sans JAX"""
    print("\n=== Test 2: Model Import ===")

    try:
        # Test import direct
        from abcnre.simulation.models.gauss_gauss_1D import GaussGaussMultiDimModel

        print("✅ Import de GaussGaussMultiDimModel réussi")

        # Test des paramètres du constructeur
        import inspect

        sig = inspect.signature(GaussGaussMultiDimModel.__init__)
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        print(f"Paramètres du constructeur: {params}")

        expected_params = [
            "mu0",
            "sigma0",
            "sigma",
            "dim",
            "n_obs",
            "marginal_of_interest",
        ]
        missing = set(expected_params) - set(params)
        if missing:
            print(f"❌ Paramètres manquants: {missing}")
            return False
        else:
            print("✅ Tous les paramètres requis sont présents")
            return True

    except Exception as e:
        print(f"❌ Erreur d'import: {e}")
        return False


def test_notebook_paths():
    """Test l'existence des notebooks"""
    print("\n=== Test 3: Notebook Paths ===")

    notebook_dir = project_root / "abcnre" / "examples" / "gauss_2D" / "notebooks"

    notebooks = ["gauss_gauss_2D-train.ipynb", "gauss_gauss_2D-load.ipynb"]

    all_exist = True
    for notebook in notebooks:
        path = notebook_dir / notebook
        exists = path.exists()
        print(f"  {notebook}: {'✅' if exists else '❌'}")
        if not exists:
            all_exist = False

    return all_exist


def test_directory_structure():
    """Test la structure des répertoires"""
    print("\n=== Test 4: Directory Structure ===")

    base_dir = project_root / "abcnre" / "examples" / "gauss_2D"

    dirs = ["notebooks", "results"]

    for dir_name in dirs:
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            print(f"Création du répertoire: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  {dir_name}/: ✅")

    return True


def main():
    """Exécute tous les tests"""
    print("Tests pour les notebooks 2D marginaux")
    print("=" * 50)

    tests = [
        test_config_loading,
        test_model_import,
        test_notebook_paths,
        test_directory_structure,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Erreur dans {test.__name__}: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("RÉSUMÉ DES TESTS:")
    success_count = sum(results)
    total_count = len(results)

    print(f"Tests réussis: {success_count}/{total_count}")

    if all(results):
        print("✅ Tous les tests sont passés ! Les notebooks devraient fonctionner.")
    else:
        print("❌ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
