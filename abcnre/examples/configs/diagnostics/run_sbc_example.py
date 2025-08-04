# examples/configs/diagnostics/run_sbc_example.py

"""
Example script showing how to use SBCConfig to run SBC.
"""

import sys
from pathlib import Path

# Add the src directory to the path to import our modules
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from abcnre.diagnostics.sbc_not_ready import SBCConfig, run_abc_sbc_with_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def example_quick_sbc():
    """Example of running quick SBC for testing."""

    print("Quick SBC Example")
    print("=================")

    # Create configuration for quick test
    config = SBCConfig(
        num_sbc_rounds=20,  # Small number for quick test
        num_posterior_samples=100,
        use_correction="no",  # Single variant for speed
        initial_bounds=(0.0, 5.0),
        estimator_path="path/to/your/estimator.yaml",  # Update this path
        output_results_path="quick_test_results.csv",
        output_plots_path="quick_test_plots.png",
        save_results=True,
        save_plots=True,
        seed=42,
    )

    print(f"Configuration created:")
    print(f"  SBC rounds: {config.num_sbc_rounds}")
    print(f"  Posterior samples: {config.num_posterior_samples}")
    print(f"  Correction: {config.use_correction}")
    print(f"  Initial bounds: {config.initial_bounds}")
    print(f"  Save results: {config.save_results}")
    print(f"  Save plots: {config.save_plots}")

    # Note: To actually run, you need a valid estimator file
    print("\nTo run this configuration:")
    print("1. Replace 'path/to/your/estimator.yaml' with your actual estimator path")
    print("2. Uncomment the line below:")
    print("# results = run_abc_sbc_with_config(config)")

    return config


def example_comprehensive_sbc():
    """Example of running comprehensive SBC analysis."""

    print("\nComprehensive SBC Example")
    print("=========================")

    # Create configuration for thorough analysis
    config = SBCConfig(
        num_sbc_rounds=500,
        num_posterior_samples=1000,
        use_correction="both",  # Run both corrected and uncorrected
        initial_bounds=(0.0, 10.0),
        estimator_path="path/to/your/estimator.yaml",  # Update this path
        output_results_path="comprehensive_results.csv",
        output_plots_path="comprehensive_plots.png",
        save_results=True,
        save_plots=True,
        seed=42,
    )

    print(f"Configuration created:")
    print(f"  SBC rounds: {config.num_sbc_rounds}")
    print(f"  Posterior samples: {config.num_posterior_samples}")
    print(f"  Correction: {config.use_correction}")
    print(f"  Expected outputs:")
    if config.use_correction == "both":
        print(f"    - comprehensive_results_uncorrected.csv")
        print(f"    - comprehensive_results_corrected.csv")
        print(f"    - comprehensive_plots_uncorrected.png")
        print(f"    - comprehensive_plots_corrected.png")

    return config


def example_yaml_config():
    """Example of using YAML configuration file."""

    print("\nYAML Configuration Example")
    print("==========================")

    yaml_path = Path(__file__).parent / "calibration.yaml"

    if yaml_path.exists():
        print(f"Found configuration file: {yaml_path}")

        # Show how to load different variants
        variants = ["quick_test", "defaults", "heavy"]

        for variant in variants:
            print(f"\nVariant: {variant}")
            try:
                from config_loader import load_sbc_config_from_yaml

                config = load_sbc_config_from_yaml(yaml_path, variant)
                print(f"  Rounds: {config.num_sbc_rounds}")
                print(f"  Samples: {config.num_posterior_samples}")
                print(f"  Correction: {config.use_correction}")

                # Show how to run
                print(f"  To run: run_abc_sbc_with_config(config)")

            except Exception as e:
                print(f"  Error loading variant: {e}")
    else:
        print(f"Configuration file not found: {yaml_path}")
        print("Make sure you're running from the diagnostics config directory")


def example_custom_workflow():
    """Example of a complete workflow with custom settings."""

    print("\nCustom Workflow Example")
    print("=======================")

    # Step 1: Create base configuration
    base_config = SBCConfig(
        num_sbc_rounds=100,
        num_posterior_samples=500,
        use_correction="no",
        initial_bounds=(0.0, 8.0),
        estimator_path="my_trained_estimator.yaml",
        output_results_path="workflow_results.csv",
        save_results=True,
        save_plots=False,  # Skip plots for initial run
        seed=123,
    )

    print("Step 1: Base configuration created")
    print(f"  Quick run with {base_config.num_sbc_rounds} rounds")

    # Step 2: Modify for more thorough analysis
    thorough_config = SBCConfig(
        num_sbc_rounds=500,
        num_posterior_samples=1000,
        use_correction="both",  # Now test both variants
        initial_bounds=base_config.initial_bounds,
        estimator_path=base_config.estimator_path,
        output_results_path="workflow_thorough_results.csv",
        output_plots_path="workflow_thorough_plots.png",
        save_results=True,
        save_plots=True,  # Now save plots
        seed=base_config.seed,
    )

    print("Step 2: Thorough configuration created")
    print(f"  Comprehensive run with {thorough_config.num_sbc_rounds} rounds")
    print(f"  Testing both correction variants")

    # Step 3: Show validation
    print("Step 3: Configuration validation")

    def validate_and_show(config, name):
        print(f"  {name}:")
        print(f"    Valid correction: {config.use_correction in ['yes', 'no', 'both']}")
        print(f"    Positive rounds: {config.num_sbc_rounds > 0}")
        print(f"    Output dir exists: {config.output_results_path.parent.exists()}")

    validate_and_show(base_config, "Base config")
    validate_and_show(thorough_config, "Thorough config")

    return base_config, thorough_config


def run_all_examples():
    """Run all example functions."""

    print("SBC Configuration Examples")
    print("=" * 40)

    try:
        # Example 1: Quick SBC
        quick_config = example_quick_sbc()

        # Example 2: Comprehensive SBC
        comprehensive_config = example_comprehensive_sbc()

        # Example 3: YAML configuration
        example_yaml_config()

        # Example 4: Custom workflow
        base_config, thorough_config = example_custom_workflow()

        print("\n" + "=" * 40)
        print("All examples completed successfully!")
        print("\nTo run SBC with your estimator:")
        print("1. Train your estimator and save to YAML")
        print("2. Update the estimator_path in any config")
        print("3. Call run_abc_sbc_with_config(config)")
        print("4. Check the output files for results")

        return {
            "quick": quick_config,
            "comprehensive": comprehensive_config,
            "base": base_config,
            "thorough": thorough_config,
        }

    except Exception as e:
        print(f"Error in examples: {e}")
        return None


if __name__ == "__main__":
    configs = run_all_examples()
