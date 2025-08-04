""" FILE NOT READY FOR USE YET
"""
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any
import logging

# Type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..inference.estimator import NeuralRatioEstimator

logger = logging.getLogger(__name__)


@dataclass
class SBCConfig:
    """
    Configuration class for Simulation-Based Calibration (SBC).

    Args:
        num_sbc_rounds: Number of SBC rounds to perform
        num_posterior_samples: Number of posterior samples to draw for each round
        use_correction: Whether to use correction ("yes", "no", or "both")
        initial_bounds: Tuple of (min, max) bounds for parameter space
        estimator_path: Path to the estimator YAML file
        output_results_path: Path to save SBC results
        output_plots_path: Optional path to save plots
        save_results: Whether to save results to file
        save_plots: Whether to save plots
        seed: Random seed for reproducibility
    """

    num_sbc_rounds: int
    num_posterior_samples: int
    use_correction: str  # "yes", "no", or "both"
    initial_bounds: Tuple[float, float]
    estimator_path: Union[str, Path]
    output_results_path: Union[str, Path]
    output_plots_path: Optional[Union[str, Path]] = None
    save_results: bool = True
    save_plots: bool = True
    seed: int = 42

    def __post_init__(self):
        """Validation after initialization."""
        if self.use_correction not in ["yes", "no", "both"]:
            raise ValueError("use_correction must be 'yes', 'no', or 'both'")

        if self.num_sbc_rounds <= 0:
            raise ValueError("num_sbc_rounds must be positive")

        if self.num_posterior_samples <= 0:
            raise ValueError("num_posterior_samples must be positive")

        # Convert paths to Path objects
        self.estimator_path = Path(self.estimator_path)
        self.output_results_path = Path(self.output_results_path)
        if self.output_plots_path is not None:
            self.output_plots_path = Path(self.output_plots_path)

        # Ensure output directories exist
        self.output_results_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_plots_path is not None:
            self.output_plots_path.parent.mkdir(parents=True, exist_ok=True)


def compute_sbc_ranks(
    key: jax.random.PRNGKey,
    estimator: "NeuralRatioEstimator",
    num_sbc_rounds: int,
    num_posterior_samples: int,
    correction: bool = False,
    initial_bounds: Tuple[float, float] = (0.0, 10.0),
) -> Dict[str, np.ndarray]:
    """
    Computes SBC ranks using the naive for-loop approach.

    This is the basic implementation that will be accelerated later.
    For now, we keep the straightforward structure with a simple loop.
    """
    # Try relative import first (when used as module)
    try:
        from .calibration import run_abc_sbc
    except ImportError:
        # Fallback for direct execution
        from calibration import run_abc_sbc

    logger.info(f"Computing SBC ranks with {num_sbc_rounds} rounds...")
    logger.info(f"Using correction: {correction}")
    logger.info(f"Posterior samples per round: {num_posterior_samples}")

    # Use the existing run_abc_sbc function
    sbc_results = run_abc_sbc(
        key=key,
        estimator=estimator,
        num_sbc_rounds=num_sbc_rounds,
        num_posterior_samples=num_posterior_samples,
        correction=correction,
        initial_bounds=initial_bounds,
    )

    logger.info("SBC computation completed")
    return sbc_results


def run_abc_sbc_with_config(config: SBCConfig) -> Dict[str, Any]:
    """
    Main function to run ABC-SBC with the given configuration.

    Args:
        config: SBCConfig object containing all parameters

    Returns:
        Dictionary containing all results and metadata
    """
    logger.info("Starting ABC-SBC with configuration:")
    logger.info(f"  Rounds: {config.num_sbc_rounds}")
    logger.info(f"  Posterior samples: {config.num_posterior_samples}")
    logger.info(f"  Correction: {config.use_correction}")
    logger.info(f"  Estimator: {config.estimator_path}")

    # Create random key
    key = jax.random.PRNGKey(config.seed)

    # Load estimator
    logger.info(f"Loading estimator from {config.estimator_path}")
    try:
        # Try relative import first (when used as module)
        try:
            from ..inference.io import load_estimator_from_yaml
        except ImportError:
            # Fallback for direct execution
            import sys
            from pathlib import Path

            parent_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(parent_dir))
            from inference.io import load_estimator_from_yaml

        estimator = load_estimator_from_yaml(config.estimator_path)
        logger.info("Estimator loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load estimator: {e}")
        raise

    # Determine which corrections to run
    corrections_to_run = []
    if config.use_correction == "yes":
        corrections_to_run = [True]
    elif config.use_correction == "no":
        corrections_to_run = [False]
    elif config.use_correction == "both":
        corrections_to_run = [False, True]

    all_results = {}

    # Run SBC for each correction setting
    for correction in corrections_to_run:
        correction_name = "corrected" if correction else "uncorrected"
        logger.info(f"Running SBC with correction={correction} ({correction_name})")

        # Split key for this run
        key, sbc_key = jax.random.split(key)

        # Compute SBC ranks
        sbc_results = compute_sbc_ranks(
            key=sbc_key,
            estimator=estimator,
            num_sbc_rounds=config.num_sbc_rounds,
            num_posterior_samples=config.num_posterior_samples,
            correction=correction,
            initial_bounds=config.initial_bounds,
        )

        # Store results
        all_results[correction_name] = sbc_results

        # Save individual results if requested
        if config.save_results:
            if len(corrections_to_run) > 1:
                # Multiple corrections: save with suffix
                results_path = config.output_results_path.with_name(
                    f"{config.output_results_path.stem}_{correction_name}{config.output_results_path.suffix}"
                )
            else:
                # Single correction: use original path
                results_path = config.output_results_path

            logger.info(f"Saving results to {results_path}")
            try:
                # Try relative import first (when used as module)
                try:
                    from .calibration import save_sbc_results_to_csv
                except ImportError:
                    # Fallback for direct execution
                    from calibration import save_sbc_results_to_csv

                save_sbc_results_to_csv(sbc_results, results_path)
                logger.info(f"Results saved to {results_path}")
            except Exception as e:
                logger.error(f"Failed to save results: {e}")

        # Create and save plots if requested
        if config.save_plots and config.output_plots_path is not None:
            if len(corrections_to_run) > 1:
                # Multiple corrections: save with suffix
                plots_path = config.output_plots_path.with_name(
                    f"{config.output_plots_path.stem}_{correction_name}{config.output_plots_path.suffix}"
                )
            else:
                # Single correction: use original path
                plots_path = config.output_plots_path

            logger.info(f"Creating and saving plots to {plots_path}")
            try:
                # Try relative import first (when used as module)
                try:
                    from .viz import plot_sbc_ranks
                except ImportError:
                    # Fallback for direct execution
                    from viz import plot_sbc_ranks

                plot_sbc_ranks(sbc_results, save_path=plots_path)
                logger.info(f"Plots saved to {plots_path}")
            except Exception as e:
                logger.error(f"Failed to save plots: {e}")
                logger.warning("Continuing without plots...")

    # Create summary results
    summary_results = {
        "config": config,
        "results": all_results,
        "metadata": {
            "num_corrections_run": len(corrections_to_run),
            "corrections_run": [
                ("corrected" if c else "uncorrected") for c in corrections_to_run
            ],
            "total_sbc_rounds": config.num_sbc_rounds * len(corrections_to_run),
            "estimator_path": str(config.estimator_path),
            "seed_used": config.seed,
        },
    }

    logger.info("ABC-SBC completed successfully!")
    logger.info(
        f"Total rounds executed: {summary_results['metadata']['total_sbc_rounds']}"
    )

    return summary_results


def run_sbc_from_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to run SBC directly from a YAML configuration file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Dictionary containing all results and metadata
    """
    import yaml

    yaml_path = Path(yaml_path)
    logger.info(f"Loading SBC configuration from {yaml_path}")

    try:
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        # Create SBCConfig from YAML
        config = SBCConfig(**yaml_config)
        logger.info("Configuration loaded from YAML")

        # Run SBC
        return run_abc_sbc_with_config(config)

    except Exception as e:
        logger.error(f"Failed to load/run from YAML: {e}")
        raise


def validate_config(config: SBCConfig) -> bool:
    """Validates an SBC configuration."""

    try:
        # Check if estimator file exists
        if not config.estimator_path.exists():
            logger.error(f"Estimator file not found: {config.estimator_path}")
            return False

        # Check if output directories are writable
        try:
            config.output_results_path.parent.mkdir(parents=True, exist_ok=True)
            if config.output_plots_path:
                config.output_plots_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Cannot create output directories: {e}")
            return False

        logger.info("Configuration validation passed")
        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def create_example_config(output_dir: Path = Path("sbc_results")) -> SBCConfig:
    """Creates an example configuration for testing."""

    return SBCConfig(
        num_sbc_rounds=100,
        num_posterior_samples=500,
        use_correction="both",
        initial_bounds=(0.0, 5.0),
        estimator_path="path/to/estimator.yaml",
        output_results_path=output_dir / "sbc_results.csv",
        output_plots_path=output_dir / "sbc_plots.png",
        save_results=True,
        save_plots=True,
        seed=42,
    )


if __name__ == "__main__":
    # Example usage when script is run directly
    print("SBC Accelerated Module")
    print("=====================")

    # Create example config
    config = create_example_config()
    print(f"Example config created:")
    print(f"  Rounds: {config.num_sbc_rounds}")
    print(f"  Correction: {config.use_correction}")
    print(f"  Output: {config.output_results_path}")

    print("\nTo run SBC with a real estimator:")
    print("1. Train and save your estimator to YAML")
    print("2. Update config.estimator_path")
    print("3. Call run_abc_sbc_with_config(config)")

    # Validate config structure (without trying to load estimator)
    print(f"\nConfig validation (structure only):")
    print(f"  Valid correction value: {config.use_correction in ['yes', 'no', 'both']}")
    print(f"  Positive rounds: {config.num_sbc_rounds > 0}")
    print(f"  Positive samples: {config.num_posterior_samples > 0}")
    print(f"  Valid bounds: {len(config.initial_bounds) == 2}")
    print(f"  Output directory created: {config.output_results_path.parent.exists()}")

    print("\nModule loaded successfully!")
