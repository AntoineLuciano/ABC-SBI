#!/usr/bin/env python3
"""
Train NRE Command - Train a neural ratio estimator for an existing estimator.
"""

import argparse
import yaml
import jax
from pathlib import Path
import logging

from abcnre.inference import load_estimator_from_yaml
from abcnre.inference.io import save_estimator_to_yaml
from abcnre.training import NNConfig, get_nn_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_nre_command(args):
    """
    Train (or retrain) a neural ratio estimator.

    Args:
        args: Parsed command line arguments
    """
    logger.info("=== ABC-NRE: Train Neural Ratio Estimator ===")

    # Setup paths
    estimator_path = Path(args.estimator_path)
    if not estimator_path.exists():
        raise FileNotFoundError(f"Estimator not found: {estimator_path}")

    nnconfig_path = Path(args.nnconfig_path)
    if not nnconfig_path.exists():
        raise FileNotFoundError(f"NN config not found: {nnconfig_path}")

    output_path = Path(args.output_path) if args.output_path else estimator_path

    # Load estimator
    logger.info(f"Loading estimator from: {estimator_path}")
    estimator = load_estimator_from_yaml(estimator_path)
    logger.info(f"Model: {estimator.simulator.model.__class__.__name__}")
    logger.info(
        f"Current training status: {'Trained' if estimator.is_trained else 'Not trained'}"
    )

    # Load classifier configuration
    logger.info(f"Loading classifier config from: {nnconfig_path}")
    classifier_config = load_classifier_config(nnconfig_path)
    logger.info(f"Network: {classifier_config.network.network_type}")
    logger.info(f"Training epochs: {classifier_config.training.num_epochs}")

    # Update estimator configuration if different
    if args.update_config:
        logger.info("Updating estimator configuration...")
        estimator.nn_config = classifier_config
        logger.info("Configuration updated")

    # Train the estimator
    logger.info("Training neural ratio estimator...")
    key = jax.random.PRNGKey(args.seed)

    training_result = estimator.train(key, n_phi_to_store=args.n_phi_to_store)
    logger.info("Neural ratio estimator training completed!")

    # Print training summary
    logger.info(f"Training result: {type(training_result).__name__}")
    if hasattr(training_result, "final_loss"):
        logger.info(f"Final loss: {training_result.final_loss}")
    if hasattr(training_result, "training_time"):
        logger.info(f"Training time: {training_result.training_time:.2f}s")

    # Save updated estimator
    logger.info(f"Saving trained estimator to: {output_path}")
    save_estimator_to_yaml(estimator, output_path, overwrite=args.overwrite)

    # Save training report
    if args.save_report:
        save_nre_training_report(
            estimator, training_result, classifier_config, output_path.parent, args
        )

    logger.info("Neural ratio estimator training completed successfully!")

    return estimator, training_result


def load_classifier_config(config_path: Path) -> NNConfig:
    """Load classifier configuration for neural ratio estimation."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Check if it's a simplified config or full NNConfig
    if "network_name" in config_dict:
        # Simplified config format
        nn_config = get_nn_config(
            network_name=config_dict["network_name"],
            network_size=config_dict.get("network_size", "default"),
            training_size=config_dict.get("training_size", "default"),
            task_type="classifier",
            lr_scheduler_name=config_dict.get("lr_scheduler_name", "reduce_on_plateau"),
            lr_scheduler_variant=config_dict.get("lr_scheduler_variant", "default"),
            stopping_rules_variant=config_dict.get(
                "stopping_rules_variant", "balanced"
            ),
            experiment_name=config_dict.get("experiment_name", "nre_training"),
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


def save_nre_training_report(
    estimator, training_result, classifier_config, output_dir: Path, args
):
    """Save a human-readable summary of the NRE training."""
    report_path = output_dir / "nre_training_report.txt"

    with open(report_path, "w") as f:
        f.write("ABC-NRE Neural Ratio Estimator Training Report\n")
        f.write("=" * 55 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Estimator: {args.estimator_path}\n")
        f.write(f"  NN Config: {args.nnconfig_path}\n")
        f.write(f"  Output: {args.output_path}\n")
        f.write(f"  Seed: {args.seed}\n")
        f.write(f"  Phi Storage: {args.n_phi_to_store}\n\n")

        f.write("Model Information:\n")
        f.write(f"  Model Type: {estimator.simulator.model.__class__.__name__}\n")
        f.write(
            f"  Data Shape: {estimator.simulator.observed_data.shape if estimator.simulator.observed_data is not None else 'None'}\n"
        )
        f.write(f"  Epsilon: {estimator.simulator.epsilon}\n\n")

        f.write("Network Configuration:\n")
        f.write(f"  Network Type: {classifier_config.network.network_type}\n")
        f.write(f"  Training Epochs: {classifier_config.training.num_epochs}\n")
        f.write(f"  Batch Size: {classifier_config.training.batch_size}\n")
        f.write(f"  Learning Rate: {classifier_config.training.learning_rate}\n")
        f.write(f"  Optimizer: {classifier_config.training.optimizer}\n\n")

        f.write("Training Results:\n")
        f.write(f"  Status: {'Completed' if estimator.is_trained else 'Failed'}\n")
        f.write(f"  Result Type: {type(training_result).__name__}\n")
        if hasattr(training_result, "final_loss"):
            f.write(f"  Final Loss: {training_result.final_loss}\n")
        if hasattr(training_result, "training_time"):
            f.write(f"  Training Time: {training_result.training_time:.2f}s\n")
        if hasattr(estimator, "stored_phis") and estimator.stored_phis is not None:
            f.write(f"  Stored Phi Samples: {len(estimator.stored_phis)}\n")

    logger.info(f"Training report saved to: {report_path}")


def setup_train_nre_parser(subparsers):
    """Setup the train_nre command parser."""
    parser = subparsers.add_parser(
        "train_nre",
        help="Train neural ratio estimator for an existing estimator",
        description="Train (or retrain) a neural ratio estimator using classifier configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "estimator_path", type=str, help="Path to the estimator YAML file"
    )

    parser.add_argument(
        "nnconfig_path",
        type=str,
        help="Path to the classifier NN configuration YAML file",
    )

    # Optional arguments
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for the trained estimator (default: overwrite input)",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for training (default: 42)"
    )

    parser.add_argument(
        "--n-phi-to-store",
        type=int,
        default=10000,
        help="Number of phi samples to store during training (default: 10000)",
    )

    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update estimator's NN config before training",
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

    parser.set_defaults(func=train_nre_command)


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_train_nre_parser(subparsers)
    args = parser.parse_args()
    train_nre_command(args)
