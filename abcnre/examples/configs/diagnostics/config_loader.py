# examples/configs/diagnostics/config_loader.py

"""
Utility functions to load SBC configurations from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union
import logging

# Add the src directory to the path to import our modules
import sys

src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from abcnre.diagnostics.sbc_not_ready import SBCConfig, run_abc_sbc

logger = logging.getLogger(__name__)


def load_sbc_config_from_yaml(
    yaml_path: Union[str, Path], variant: str = "default"
) -> SBCConfig:
    """
    Load SBC configuration from YAML file.

    Args:
        yaml_path: Path to the YAML configuration file
        variant: Which configuration variant to load ("quick_test", "default", "heavy")

    Returns:
        SBCConfig object ready to use
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    logger.info(f"Loading SBC configuration from {yaml_path}")
    logger.info(f"Using variant: {variant}")

    with open(yaml_path, "r") as f:
        all_configs = yaml.safe_load(f)

    if variant not in all_configs:
        available_variants = list(all_configs.keys())
        raise ValueError(
            f"Variant '{variant}' not found. Available: {available_variants}"
        )

    config_dict = all_configs[variant]

    # Convert to SBCConfig object
    config = SBCConfig(**config_dict)

    logger.info(f"✅ Configuration loaded successfully:")
    logger.info(f"  Rounds: {config.num_sbc_rounds}")
    logger.info(f"  Posterior samples: {config.num_posterior_samples}")
    logger.info(f"  Correction: {config.use_correction}")

    return config


def run_sbc_with_variant(
    yaml_path: Union[str, Path],
    variant: str = "default",
    estimator_path: Union[str, Path] = None,
) -> Dict[str, Any]:
    """
    Convenience function to load config and run SBC in one step.

    Args:
        yaml_path: Path to the YAML configuration file
        variant: Which configuration variant to use
        estimator_path: Optional override for estimator path

    Returns:
        Dictionary containing all SBC results
    """
    # Load configuration
    config = load_sbc_config_from_yaml(yaml_path, variant)

    # Override estimator path if provided
    if estimator_path is not None:
        config.estimator_path = Path(estimator_path)
        logger.info(f"Overriding estimator path: {config.estimator_path}")

    # Run SBC
    logger.info(f"Running SBC with variant '{variant}'...")
    results = run_abc_sbc(config)

    return results


def list_available_variants(yaml_path: Union[str, Path]) -> list:
    """
    List all available configuration variants in a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        List of available variant names
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        all_configs = yaml.safe_load(f)

    return list(all_configs.keys())


def print_config_summary(yaml_path: Union[str, Path]):
    """
    Print a summary of all configurations in the YAML file.

    Args:
        yaml_path: Path to the YAML configuration file
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        print(f"❌ Configuration file not found: {yaml_path}")
        return

    with open(yaml_path, "r") as f:
        all_configs = yaml.safe_load(f)

    print(f"📋 SBC Configuration Summary: {yaml_path.name}")
    print("=" * 60)

    for variant_name, config in all_configs.items():
        print(f"\n🔧 {variant_name.upper()}:")
        print(f"  Rounds: {config['num_sbc_rounds']}")
        print(f"  Posterior samples: {config['num_posterior_samples']}")
        print(f"  Correction: {config['use_correction']}")
        print(f"  Bounds: {config['initial_bounds']}")
        print(f"  Save results: {config['save_results']}")
        print(f"  Save plots: {config['save_plots']}")

    print("\n" + "=" * 60)
    print("💡 Usage:")
    print("  from config_loader import run_sbc_with_variant")
    print(f"  results = run_sbc_with_variant('{yaml_path.name}', 'default')")


def create_custom_config(
    num_sbc_rounds: int = 500,
    num_posterior_samples: int = 1000,
    use_correction: str = "both",
    initial_bounds: tuple = (0.0, 10.0),
    estimator_path: str = "path/to/estimator.yaml",
    output_dir: str = "results",
    seed: int = 42,
) -> SBCConfig:
    """
    Create a custom SBC configuration programmatically.

    Args:
        num_sbc_rounds: Number of SBC rounds
        num_posterior_samples: Posterior samples per round
        use_correction: "yes", "no", or "both"
        initial_bounds: (min, max) parameter bounds
        estimator_path: Path to estimator YAML
        output_dir: Directory for outputs
        seed: Random seed

    Returns:
        SBCConfig object
    """
    output_dir = Path(output_dir)

    config = SBCConfig(
        num_sbc_rounds=num_sbc_rounds,
        num_posterior_samples=num_posterior_samples,
        use_correction=use_correction,
        initial_bounds=initial_bounds,
        estimator_path=estimator_path,
        output_results_path=output_dir / "sbc_custom.csv",
        output_plots_path=output_dir / "sbc_custom.png",
        save_results=True,
        save_plots=True,
        seed=seed,
    )

    logger.info("✅ Custom configuration created")
    return config


# Convenience functions for common use cases


def quick_test_sbc(
    estimator_path: Union[str, Path], output_dir: str = "quick_results"
) -> Dict[str, Any]:
    """Run a quick SBC test for debugging/validation."""
    config_path = Path(__file__).parent / "calibration.yaml"
    return run_sbc_with_variant(config_path, "quick_test", estimator_path)


def standard_sbc(
    estimator_path: Union[str, Path], output_dir: str = "sbc_results"
) -> Dict[str, Any]:
    """Run standard SBC analysis."""
    config_path = Path(__file__).parent / "calibration.yaml"
    return run_sbc_with_variant(config_path, "default", estimator_path)


def comprehensive_sbc(
    estimator_path: Union[str, Path], output_dir: str = "comprehensive_results"
) -> Dict[str, Any]:
    """Run comprehensive SBC analysis."""
    config_path = Path(__file__).parent / "calibration.yaml"
    return run_sbc_with_variant(config_path, "heavy", estimator_path)


if __name__ == "__main__":
    # Demo usage
    config_file = Path(__file__).parent / "calibration.yaml"

    print("🎯 SBC Configuration Loader Demo")
    print("=" * 40)

    # Show available configurations
    if config_file.exists():
        print_config_summary(config_file)

        # List variants
        variants = list_available_variants(config_file)
        print(f"\n📝 Available variants: {variants}")

        # Load a specific config
        try:
            config = load_sbc_config_from_yaml(config_file, "default")
            print(f"\n✅ Loaded 'default' configuration successfully")
            print(f"   Rounds: {config.num_sbc_rounds}")
            print(f"   Samples: {config.num_posterior_samples}")
        except Exception as e:
            print(f"\n❌ Error loading configuration: {e}")
    else:
        print(f"❌ Configuration file not found: {config_file}")
        print("Please run this script from the diagnostics config directory.")
