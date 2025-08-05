"""
Create Simulator and Estimator Command - Combined workflow using modular functions
"""

import argparse
import yaml
import jax
from pathlib import Path
from typing import Optional
import logging

# Import modular functions from other CLI commands
from .create_simulator import create_simulator_command
from .create_estimator import create_estimator_command

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simulator_and_estimator_command(args):
    """
    Create and configure both ABC simulator and neural ratio estimator using modular approach.

    This function orchestrates the complete pipeline by calling:
    1. create_simulator_command() to create and configure the simulator
    2. create_estimator_command() to create and train the estimator

    Args:
        args: Parsed command line arguments
    """
    logger.info("=== ABC-NRE: Create Simulator and Estimator (Modular) ===")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # === Step 1: Create Simulator ===
    logger.info("--- Step 1: Creating Simulator ---")

    # Prepare arguments for create_simulator_command
    simulator_args = argparse.Namespace()
    simulator_args.output_dir = str(output_dir)
    simulator_args.seed = args.seed

    # Model configuration
    simulator_args.model_path = getattr(args, "model_path", None)
    simulator_args.model_name = getattr(args, "model_name", "gauss_gauss")

    # Summary statistics configuration
    simulator_args.learn_stats = getattr(args, "with_summary_stats", False)
    simulator_args.regressor_config = getattr(args, "regressor_config", None)

    # Observed data configuration
    simulator_args.observed_data = None  # Always generate new data
    simulator_args.true_theta = getattr(args, "true_theta", [0.0, 1.0])

    # Epsilon configuration
    simulator_args.epsilon = getattr(args, "epsilon", None)
    simulator_args.quantile_distance = getattr(args, "quantile_distance", 0.1)

    # Call create_simulator_command
    try:
        simulator = create_simulator_command(simulator_args)
        logger.info("Simulator created successfully")
    except Exception as e:
        logger.error(f"Failed to create simulator: {e}")
        raise

    # === Step 2: Create Estimator ===
    logger.info("--- Step 2: Creating Estimator ---")

    # Prepare arguments for create_estimator_command
    estimator_args = argparse.Namespace()
    estimator_args.output_dir = str(output_dir)
    estimator_args.simulator = str(output_dir / "simulator.yaml")

    # Classifier configuration - map from old argument names to new ones
    if hasattr(args, "network_config") and args.network_config:
        estimator_args.classifier_config = args.network_config
    else:
        # Use default classifier config based on network name
        from pathlib import Path

        templates_dir = Path(__file__).parent / "templates" / "classifier_configs"
        network_name = getattr(args, "network_name", "mlp_default")
        estimator_args.classifier_config = str(templates_dir / f"{network_name}.yaml")

    # Optional epsilon overrides
    estimator_args.epsilon = None  # Use simulator's epsilon
    estimator_args.quantile_distance = None  # Use simulator's epsilon

    # Call create_estimator_command
    try:
        estimator = create_estimator_command(estimator_args)
        logger.info("Estimator created successfully")
    except Exception as e:
        logger.error(f"Failed to create estimator: {e}")
        raise

    # === Step 3: Generate Combined Report ===
    logger.info("--- Step 3: Generating Combined Report ---")

    report_path = output_dir / "combined_report.txt"
    with open(report_path, "w") as f:
        f.write("=== ABC-NRE: Combined Simulator and Estimator Report ===\n\n")
        f.write(f"Command: create_simulator_and_estimator\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Random seed: {args.seed}\n\n")

        f.write("Configuration:\n")
        f.write(f"  Model path: {getattr(args, 'model_path', 'N/A')}\n")
        f.write(f"  Model name: {getattr(args, 'model_name', 'gauss_gauss')}\n")
        f.write(f"  Summary stats: {getattr(args, 'with_summary_stats', False)}\n")
        f.write(f"  Network name: {getattr(args, 'network_name', 'mlp_default')}\n\n")

        f.write("Files created:\n")
        f.write(f"  - simulator.yaml\n")
        f.write(f"  - simulator_report.txt\n")
        f.write(f"  - estimator.yaml\n")
        f.write(f"  - estimator_report.txt\n")
        f.write(f"  - combined_report.txt (this file)\n\n")

        f.write("Next steps:\n")
        f.write(f"  abcnre plot_posterior_comparison {output_dir}/estimator.yaml\n")
        f.write(f"  abcnre compute_metrics {output_dir}/estimator.yaml\n")
        f.write(f"  abcnre run_sbc {output_dir}/estimator.yaml --n_samples 100\n")

    logger.info(f"Combined report saved to: {report_path}")
    logger.info("=== Pipeline completed successfully ===")

    return simulator, estimator


def setup_create_simulator_and_estimator_parser(subparsers):
    """Setup argument parser for create_simulator_and_estimator command."""
    parser = subparsers.add_parser(
        "create_simulator_and_estimator",
        help="Create and train both simulator and estimator in one step",
        description="""
        Create an ABC simulator and neural ratio estimator in a single command.
        This combines the create_simulator and create_estimator workflows for convenience.
        
        Examples:
          # Basic usage with model name
          abcnre create_simulator_and_estimator --model-name gauss_gauss ./output
          
          # With custom model and epsilon
          abcnre create_simulator_and_estimator --model-path model.yaml \\
              --epsilon 0.1 ./output
          
          # With summary statistics learning
          abcnre create_simulator_and_estimator --model-name gauss_gauss_100d \\
              --with-summary-stats --regressor-config templates/regressor_configs/deepset_fast.yaml \\
              ./output
          
          # With custom network
          abcnre create_simulator_and_estimator --model-name gauss_gauss \\
              --network-name mlp_fast ./output
        """,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=40
        ),
    )

    # === Required Arguments ===
    parser.add_argument("output_dir", help="Output directory for generated files")

    # === Model Configuration ===
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-name",
        default="gauss_gauss",
        help="Name of the example model configuration (default: gauss_gauss)",
    )
    model_group.add_argument(
        "--model-path",
        help="Path to custom model configuration YAML file (overrides model_name)",
    )

    # === Simulator Configuration ===
    sim_group = parser.add_argument_group("Simulator Configuration")
    sim_group.add_argument(
        "--epsilon",
        type=float,
        help="ABC tolerance threshold (required if --quantile-distance not provided)",
    )
    sim_group.add_argument(
        "--quantile-distance",
        type=float,
        default=0.1,
        help="Quantile for automatic epsilon calculation (default: 0.1)",
    )

    # === Summary Statistics ===
    stats_group = parser.add_argument_group("Summary Statistics")
    stats_group.add_argument(
        "--with-summary-stats",
        action="store_true",
        help="Enable summary statistics learning",
    )
    stats_group.add_argument(
        "--regressor-config",
        help="Path to regressor configuration YAML file for summary stats",
    )

    # === Network Configuration ===
    network_group = parser.add_argument_group("Network Configuration")
    network_group.add_argument(
        "--network-name",
        default="mlp_default",
        help="Name of the neural network configuration (default: mlp_default)",
    )
    network_group.add_argument(
        "--network-config", help="Path to custom classifier configuration YAML file"
    )

    # === General Options ===
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # === Validation ===
    def validate_args(args):
        """Validate argument combinations."""
        if args.epsilon is None and args.quantile_distance is None:
            # Set default quantile_distance if neither is provided
            args.quantile_distance = 0.1

        if args.epsilon is not None and args.quantile_distance is not None:
            # Prefer epsilon over quantile_distance if both provided
            args.quantile_distance = None
            logger.warning("Both epsilon and quantile_distance provided, using epsilon")

        if args.quantile_distance is not None:
            if not (0 < args.quantile_distance < 1):
                parser.error("--quantile-distance must be between 0 and 1")

    parser.set_defaults(
        func=create_simulator_and_estimator_command, validate=validate_args
    )

    return parser
