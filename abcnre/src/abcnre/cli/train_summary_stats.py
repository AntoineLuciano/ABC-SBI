#!/usr/bin/env python3
"""
Train Summary Statistics Command - Train summary statistics for an existing simulator.
"""

import argparse
import yaml
import jax
from pathlib import Path
import logging

from abcnre.simulation import load_simulator_from_yaml, save_simulator_to_yaml
from abcnre.training import NNConfig, get_nn_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_summary_stats_command(args):
    """
    Train summary statistics for an existing simulator.

    Args:
        args: Parsed command line arguments
    """
    logger.info("=== ABC-NRE: Train Summary Statistics ===")

    # Setup paths
    simulator_path = Path(args.simulator_path)
    if not simulator_path.exists():
        raise FileNotFoundError(f"Simulator not found: {simulator_path}")

    nnconfig_path = Path(args.nnconfig_path)
    if not nnconfig_path.exists():
        raise FileNotFoundError(f"NN config not found: {nnconfig_path}")

    output_path = Path(args.output_path) if args.output_path else simulator_path

    # Load simulator
    logger.info(f"Loading simulator from: {simulator_path}")
    simulator = load_simulator_from_yaml(simulator_path)
    logger.info(f"Model: {simulator.model.__class__.__name__}")

    # Load regressor configuration
    logger.info(f"Loading regressor config from: {nnconfig_path}")
    regressor_config = load_regressor_config(nnconfig_path)
    logger.info(f"Network: {regressor_config.network.network_type}")
    logger.info(f"Training epochs: {regressor_config.training.num_epochs}")

    # Train summary statistics
    logger.info("Training summary statistics...")
    key = jax.random.PRNGKey(args.seed)
    key, subkey_learn = jax.random.split(key)

    simulator.train_summary_network(subkey_learn, regressor_config)
    logger.info("Summary statistics training completed!")

    # Check correlation if requested
    if args.check_correlation:
        logger.info("Checking summary statistics correlation...")
        key, subkey_check = jax.random.split(key)
        correlation = simulator.check_summary_stats_correlation(
            subkey_check, n_samples=args.correlation_samples
        )
        logger.info(f"Summary stats correlation: {correlation:.4f}")

    # Save updated simulator
    logger.info(f"Saving updated simulator to: {output_path}")
    save_simulator_to_yaml(simulator, output_path, overwrite=args.overwrite)

    # Save training report
    if args.save_report:
        save_summary_stats_report(simulator, regressor_config, output_path.parent, args)

    logger.info("Summary statistics training completed successfully!")


def load_regressor_config(config_path: Path) -> NNConfig:
    """Load regressor configuration."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Check if it's a simplified config or full NNConfig
    if "network_name" in config_dict:
        # Simplified config format
        nn_config = get_nn_config(
            network_name=config_dict["network_name"],
            network_size=config_dict.get("network_size", "default"),
            training_size=config_dict.get("training_size", "default"),
            task_type="regressor",
            lr_scheduler_name=config_dict.get("lr_scheduler_name", "constant"),
            lr_scheduler_variant=config_dict.get("lr_scheduler_variant", "default"),
            stopping_rules_variant=config_dict.get(
                "stopping_rules_variant", "balanced"
            ),
            experiment_name=config_dict.get(
                "experiment_name", "summary_stats_training"
            ),
        )

        # Apply any overrides
        if "overrides" in config_dict:
            for key, value in config_dict["overrides"].items():
                if hasattr(nn_config.training, key):
                    setattr(nn_config.training, key, value)
                    logger.info(f"Applied override: {key} = {value}")

        return nn_config
    else:
        # Full NNConfig YAML format
        return NNConfig.load(config_path)


def save_summary_stats_report(simulator, regressor_config, output_dir: Path, args):
    """Save a human-readable summary of the summary statistics training."""
    report_path = output_dir / "summary_stats_report.txt"

    with open(report_path, "w") as f:
        f.write("ABC-NRE Summary Statistics Training Report\n")
        f.write("=" * 50 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Simulator: {args.simulator_path}\n")
        f.write(f"  NN Config: {args.nnconfig_path}\n")
        f.write(f"  Output: {args.output_path}\n")
        f.write(f"  Seed: {args.seed}\n\n")

        f.write("Model Information:\n")
        f.write(f"  Model Type: {simulator.model.__class__.__name__}\n")
        f.write(
            f"  Data Shape: {simulator.observed_data.shape if simulator.observed_data is not None else 'None'}\n"
        )
        f.write(f"  Epsilon: {simulator.epsilon}\n\n")

        f.write("Network Configuration:\n")
        f.write(f"  Network Type: {regressor_config.network.network_type}\n")
        f.write(f"  Training Epochs: {regressor_config.training.num_epochs}\n")
        f.write(f"  Batch Size: {regressor_config.training.batch_size}\n")
        f.write(f"  Learning Rate: {regressor_config.training.learning_rate}\n")
        f.write(f"  Optimizer: {regressor_config.training.optimizer}\n\n")

        f.write("Summary Statistics:\n")
        f.write(f"  Status: Trained\n")
        f.write(f"  Check Correlation: {args.check_correlation}\n")
        if args.check_correlation:
            f.write(f"  Correlation Samples: {args.correlation_samples}\n")

    logger.info(f"Training report saved to: {report_path}")


def setup_train_summary_stats_parser(subparsers):
    """Setup the train_summary_stats command parser."""
    parser = subparsers.add_parser(
        "train_summary_stats",
        help="Train summary statistics for an existing simulator",
        description="Train summary statistics for an existing simulator using regressor configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "simulator_path", type=str, help="Path to the simulator YAML file"
    )

    parser.add_argument(
        "nnconfig_path",
        type=str,
        help="Path to the regressor NN configuration YAML file",
    )

    # Optional arguments
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for the updated simulator (default: overwrite input)",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for training (default: 42)"
    )

    parser.add_argument(
        "--check-correlation",
        action="store_true",
        help="Check correlation between summary stats and true parameters",
    )

    parser.add_argument(
        "--correlation-samples",
        type=int,
        default=10000,
        help="Number of samples for correlation check (default: 10000)",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )

    parser.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="Save training report (default: True)",
    )

    parser.set_defaults(func=train_summary_stats_command)


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_train_summary_stats_parser(subparsers)
    args = parser.parse_args()
    train_summary_stats_command(args)
