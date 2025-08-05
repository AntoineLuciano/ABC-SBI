#!/usr/bin/env python3
"""
Train Full Pipeline Command - Complete ABC-NRE pipeline with modular approach.

This command orchestrates the complete ABC-NRE workflow:
1. Create simulator
2. Train summary statistics (optional)
3. Create estimator
4. Train neural ratio estimator
"""

import argparse
from pathlib import Path
import logging

# Import modular functions from other CLI commands
from .create_simulator import create_simulator_command
from .train_summary_stats import train_summary_stats_command
from .create_estimator import create_estimator_command
from .train_nre import train_nre_command

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_full_pipeline_command(args):
    """
    Execute the complete ABC-NRE pipeline using modular functions.

    This orchestrates:
    1. create_simulator_command() to create and configure the simulator
    2. train_summary_stats_command() to train summary statistics (if enabled)
    3. create_estimator_command() to create the estimator
    4. train_nre_command() to train the neural ratio estimator

    Args:
        args: Parsed command line arguments
    """
    logger.info("=== ABC-NRE: Full Pipeline Training ===")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Define intermediate file paths
    simulator_path = output_dir / "simulator.yaml"
    estimator_path = output_dir / "estimator.yaml"

    # === Step 1: Create Simulator ===
    logger.info("--- Step 1/4: Creating Simulator ---")

    simulator_args = argparse.Namespace()
    simulator_args.output_dir = str(output_dir)
    simulator_args.seed = args.seed

    # Model configuration
    simulator_args.model_path = getattr(args, "model_path", None)
    simulator_args.model_name = getattr(args, "model_name", "gauss_gauss")

    # Observed data configuration
    simulator_args.observed_data = None  # Always generate new data
    simulator_args.true_theta = getattr(args, "true_theta", None)

    # Epsilon configuration
    simulator_args.epsilon = getattr(args, "epsilon", None)
    simulator_args.quantile_distance = getattr(args, "quantile_distance", 0.1)

    # Summary stats - disable for now, will be done in separate step
    simulator_args.learn_stats = False
    simulator_args.regressor_config = None

    try:
        simulator = create_simulator_command(simulator_args)
        logger.info(f"✓ Simulator created and saved to: {simulator_path}")
    except Exception as e:
        logger.error(f"✗ Failed to create simulator: {e}")
        raise

    # === Step 2: Train Summary Statistics (Optional) ===
    if args.with_summary_stats:
        logger.info("--- Step 2/4: Training Summary Statistics ---")

        if not args.regressor_config:
            # Use default regressor config
            templates_dir = Path(__file__).parent / "templates" / "regressor_configs"
            args.regressor_config = str(templates_dir / "deepset_default.yaml")
            logger.info(f"Using default regressor config: {args.regressor_config}")

        stats_args = argparse.Namespace()
        stats_args.simulator_path = str(simulator_path)
        stats_args.nnconfig_path = args.regressor_config
        stats_args.output_path = str(simulator_path)  # Overwrite simulator

        try:
            train_summary_stats_command(stats_args)
            logger.info("✓ Summary statistics trained successfully")
        except Exception as e:
            logger.error(f"✗ Failed to train summary statistics: {e}")
            raise
    else:
        logger.info("--- Step 2/4: Skipping Summary Statistics (disabled) ---")

    # === Step 3: Create Estimator ===
    logger.info("--- Step 3/4: Creating Estimator ---")

    estimator_args = argparse.Namespace()
    estimator_args.output_dir = str(output_dir)
    estimator_args.simulator = str(simulator_path)

    # Classifier configuration
    if args.classifier_config:
        estimator_args.classifier_config = args.classifier_config
    else:
        # Use default classifier config based on network name
        templates_dir = Path(__file__).parent / "templates" / "classifier_configs"
        network_name = getattr(args, "network_name", "mlp_default")
        estimator_args.classifier_config = str(templates_dir / f"{network_name}.yaml")
        logger.info(
            f"Using default classifier config: {estimator_args.classifier_config}"
        )

    # No epsilon overrides - use simulator's values
    estimator_args.epsilon = None
    estimator_args.quantile_distance = None

    try:
        estimator = create_estimator_command(estimator_args)
        logger.info(f"✓ Estimator created and saved to: {estimator_path}")
    except Exception as e:
        logger.error(f"✗ Failed to create estimator: {e}")
        raise

    # === Step 4: Train Neural Ratio Estimator ===
    logger.info("--- Step 4/4: Training Neural Ratio Estimator ---")

    # Use the same classifier config for training
    nre_args = argparse.Namespace()
    nre_args.estimator_path = str(estimator_path)
    nre_args.nnconfig_path = estimator_args.classifier_config
    nre_args.output_path = str(estimator_path)  # Overwrite estimator

    try:
        train_nre_command(nre_args)
        logger.info("✓ Neural ratio estimator trained successfully")
    except Exception as e:
        logger.error(f"✗ Failed to train neural ratio estimator: {e}")
        raise

    # === Step 5: Generate Pipeline Report ===
    logger.info("--- Step 5/4: Generating Pipeline Report ---")

    report_path = output_dir / "pipeline_report.txt"
    with open(report_path, "w") as f:
        f.write("=== ABC-NRE: Full Pipeline Training Report ===\n\n")
        f.write(f"Command: train_full_pipeline\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Random seed: {args.seed}\n\n")

        f.write("Pipeline Configuration:\n")
        f.write(f"  Model path: {getattr(args, 'model_path', 'N/A')}\n")
        f.write(f"  Model name: {getattr(args, 'model_name', 'gauss_gauss')}\n")
        f.write(f"  Summary stats: {args.with_summary_stats}\n")
        if args.with_summary_stats:
            f.write(f"  Regressor config: {args.regressor_config}\n")
        f.write(f"  Network name: {getattr(args, 'network_name', 'mlp_default')}\n")
        if args.classifier_config:
            f.write(f"  Classifier config: {args.classifier_config}\n")
        f.write("\n")

        f.write("Pipeline Steps Completed:\n")
        f.write("  ✓ Step 1: Simulator created\n")
        if args.with_summary_stats:
            f.write("  ✓ Step 2: Summary statistics trained\n")
        else:
            f.write("  - Step 2: Summary statistics skipped\n")
        f.write("  ✓ Step 3: Estimator created\n")
        f.write("  ✓ Step 4: Neural ratio estimator trained\n")
        f.write("\n")

        f.write("Files created:\n")
        f.write(f"  - simulator.yaml (ABC simulator)\n")
        f.write(f"  - simulator_report.txt\n")
        f.write(f"  - estimator.yaml (trained NRE)\n")
        f.write(f"  - estimator_report.txt\n")
        f.write(f"  - pipeline_report.txt (this file)\n")
        f.write("\n")

        f.write("Next steps - Evaluation commands:\n")
        f.write(f"  abcnre plot_posterior_comparison {estimator_path}\n")
        f.write(f"  abcnre compute_metrics {estimator_path}\n")
        f.write(f"  abcnre run_sbc {estimator_path} --n_samples 100\n")

    logger.info(f"Pipeline report saved to: {report_path}")
    logger.info("=== Full Pipeline Training Completed Successfully ===")

    return simulator, estimator


def setup_train_full_pipeline_parser(subparsers):
    """Setup argument parser for train_full_pipeline command."""
    parser = subparsers.add_parser(
        "train_full_pipeline",
        help="Execute complete ABC-NRE pipeline (create simulator + train stats + create estimator + train NRE)",
        description="""
        Execute the complete ABC-NRE training pipeline in a single command.
        This orchestrates all training steps using the modular CLI functions.
        
        Examples:
          # Basic pipeline with model name
          abcnre train_full_pipeline --model-name gauss_gauss ./results
          
          # Pipeline with summary statistics
          abcnre train_full_pipeline --model-name gauss_gauss_100d \\
              --with-summary-stats ./results
          
          # Custom configurations
          abcnre train_full_pipeline --model-path custom_model.yaml \\
              --with-summary-stats \\
              --regressor-config templates/regressor_configs/deepset_fast.yaml \\
              --classifier-config templates/classifier_configs/mlp_fast.yaml \\
              ./results
          
          # Fast prototyping
          abcnre train_full_pipeline --model-name gauss_gauss \\
              --network-name mlp_fast ./results
        """,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=40
        ),
    )

    # === Required Arguments ===
    parser.add_argument("output_dir", help="Output directory for all generated files")

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
        help="ABC tolerance threshold (if not provided, will use quantile-distance)",
    )
    sim_group.add_argument(
        "--quantile-distance",
        type=float,
        default=0.1,
        help="Quantile for automatic epsilon calculation (default: 0.1)",
    )
    sim_group.add_argument(
        "--true-theta",
        nargs="+",
        type=float,
        default=[0.0, 1.0],
        help="True parameter values for data generation (default: [0.0, 1.0])",
    )

    # === Summary Statistics ===
    stats_group = parser.add_argument_group("Summary Statistics")
    stats_group.add_argument(
        "--with-summary-stats",
        action="store_true",
        help="Enable summary statistics learning step",
    )
    stats_group.add_argument(
        "--regressor-config",
        help="Path to regressor configuration YAML file (default: deepset_default.yaml)",
    )

    # === Neural Network Configuration ===
    network_group = parser.add_argument_group("Network Configuration")
    network_group.add_argument(
        "--network-name",
        default="mlp_default",
        help="Name of default classifier configuration (default: mlp_default)",
    )
    network_group.add_argument(
        "--classifier-config",
        help="Path to custom classifier configuration YAML file",
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
        if args.epsilon is not None and args.quantile_distance is not None:
            # Prefer epsilon over quantile_distance if both provided
            args.quantile_distance = None
            logger.warning("Both epsilon and quantile_distance provided, using epsilon")

        if args.quantile_distance is not None:
            if not (0 < args.quantile_distance < 1):
                parser.error("--quantile-distance must be between 0 and 1")

        if args.with_summary_stats and args.regressor_config:
            regressor_path = Path(args.regressor_config)
            if not regressor_path.exists():
                parser.error(f"Regressor config file not found: {regressor_path}")

        if args.classifier_config:
            classifier_path = Path(args.classifier_config)
            if not classifier_path.exists():
                parser.error(f"Classifier config file not found: {classifier_path}")

    parser.set_defaults(func=train_full_pipeline_command, validate=validate_args)

    return parser
