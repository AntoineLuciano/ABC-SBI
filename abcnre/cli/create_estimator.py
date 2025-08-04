"""
Create Estimator Command - Reproduces Step 3 of gauss_gauss_train.ipynb
"""

import yaml
import jax
from pathlib import Path
from typing import Optional

from abcnre.simulation import load_simulator_from_yaml
from abcnre.training import NNConfig
from abcnre.inference import NeuralRatioEstimator
from abcnre.inference.io import save_estimator_to_yaml


def create_estimator_command(args):
    """
    Create and train a neural ratio estimator.

    Reproduces Step 3 from gauss_gauss_train.ipynb:
    - Step 3.1: Load simulator and create NNConfig
    - Step 3.2: Create NeuralRatioEstimator
    - Step 3.3: Train the estimator
    - Step 3.4: Save the trained estimator
    """
    print("=== ABC-NRE: Create Estimator ===")

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    simulator_path = Path(args.simulator)
    if not simulator_path.exists():
        raise FileNotFoundError(f"Simulator not found: {simulator_path}")

    # === Step 3.0: Load simulator ===
    print("--- Step 3.0: Load simulator ---")
    simulator = load_simulator_from_yaml(simulator_path)
    print(f"Loaded simulator from: {simulator_path}")
    print(f"Model: {simulator.model.__class__.__name__}")
    print(f"Observed data shape: {simulator.observed_data.shape}")
    print(f"Current epsilon: {simulator.epsilon}")

    # Update simulator parameters if provided
    if args.epsilon is not None:
        simulator.epsilon = args.epsilon
        print(f"Updated epsilon to: {args.epsilon}")

    if args.quantile_distance is not None:
        key = jax.random.PRNGKey(42)  # Fixed seed for epsilon calculation
        simulator.set_epsilon_from_quantile(
            key=key, quantile_distance=args.quantile_distance, n_samples=10000
        )
        print(
            f"Updated epsilon from {args.quantile_distance} quantile: {simulator.epsilon}"
        )

    # === Step 3.1: Load classifier configuration ===
    print("--- Step 3.1: Load classifier configuration ---")

    classifier_config_path = Path(args.classifier_config)
    if not classifier_config_path.exists():
        raise FileNotFoundError(
            f"Classifier config not found: {classifier_config_path}"
        )

    nn_config = load_classifier_config(classifier_config_path)
    print(f"Loaded classifier config: {nn_config.experiment_name}")
    print(f"Network: {nn_config.network.network_type}")
    print(f"Training epochs: {nn_config.training.num_epochs}")
    print(f"Batch size: {nn_config.training.batch_size}")
    print(f"Learning rate: {nn_config.training.learning_rate}")

    # === Step 3.2: Create NeuralRatioEstimator ===
    print("--- Step 3.2: Create NeuralRatioEstimator ---")

    estimator = NeuralRatioEstimator(nn_config=nn_config, simulator=simulator)
    print("NeuralRatioEstimator created successfully")

    # === Step 3.3: Train the estimator ===
    print("--- Step 3.3: Train the NeuralRatioEstimator ---")

    # Use a fixed seed for training reproducibility
    key = jax.random.PRNGKey(42)

    print("Starting training...")
    res = estimator.train(key, n_phi_to_store=10000)
    print("Training completed!")

    # Print training summary
    print(f"Training result: {type(res).__name__}")
    if hasattr(res, "final_loss"):
        print(f"Final loss: {res.final_loss}")
    if hasattr(res, "training_time"):
        print(f"Training time: {res.training_time:.2f}s")

    # === Step 3.4: Save the trained estimator ===
    print("--- Step 3.4: Save the trained estimator ---")

    estimator_path = output_dir / "estimator.yaml"
    save_estimator_to_yaml(estimator, estimator_path, overwrite=True)

    print(f"Estimator saved to: {estimator_path}")

    # Save summary report
    save_estimator_report(estimator, res, output_dir, args)

    return estimator, res


def load_classifier_config(config_path: Path) -> NNConfig:
    """Load classifier configuration for neural ratio estimation."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Check if it's a simplified config or full NNConfig
    if "network_name" in config_dict:
        # Simplified config format
        from abcnre.training import get_nn_config

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
            experiment_name=config_dict.get("experiment_name", "classifier_experiment"),
        )

        # Apply any overrides
        if "overrides" in config_dict:
            for key, value in config_dict["overrides"].items():
                if hasattr(nn_config.training, key):
                    setattr(nn_config.training, key, value)
                    print(f"Applied override: {key} = {value}")

        return nn_config
    else:
        # Full NNConfig YAML format
        return NNConfig.load(config_path)


def save_estimator_report(estimator, training_result, output_dir: Path, args):
    """Save a human-readable summary of the estimator training."""
    report_path = output_dir / "estimator_report.txt"

    with open(report_path, "w") as f:
        f.write("=== Neural Ratio Estimator Training Report ===\n\n")
        f.write(f"Created on: {Path().cwd()}\n")
        f.write(f"Command args: {vars(args)}\n\n")

        f.write("Simulator Information:\n")
        f.write(f"  Source: {args.simulator}\n")
        f.write(f"  Model: {estimator.simulator.model.__class__.__name__}\n")
        f.write(f"  Epsilon: {estimator.simulator.epsilon}\n")
        f.write(
            f"  Summary stats enabled: {estimator.simulator.config.get('summary_stats_enabled', False)}\n\n"
        )

        f.write("Network Configuration:\n")
        f.write(f"  Experiment: {estimator.nn_config.experiment_name}\n")
        f.write(f"  Network type: {estimator.nn_config.network.network_type}\n")
        f.write(f"  Task type: {estimator.nn_config.task_type}\n\n")

        f.write("Training Configuration:\n")
        f.write(f"  Epochs: {estimator.nn_config.training.num_epochs}\n")
        f.write(f"  Batch size: {estimator.nn_config.training.batch_size}\n")
        f.write(f"  Learning rate: {estimator.nn_config.training.learning_rate}\n")
        f.write(f"  Optimizer: {estimator.nn_config.training.optimizer}\n")
        f.write(
            f"  LR scheduler: {estimator.nn_config.training.lr_scheduler.schedule_name}\n\n"
        )

        f.write("Training Results:\n")
        f.write(f"  Status: {'Completed' if estimator.is_trained else 'Failed'}\n")
        if hasattr(training_result, "final_loss"):
            f.write(f"  Final loss: {training_result.final_loss}\n")
        if hasattr(training_result, "training_time"):
            f.write(f"  Training time: {training_result.training_time:.2f}s\n")

        f.write("\nFiles created:\n")
        f.write(f"  - estimator.yaml (main configuration)\n")
        f.write(f"  - estimator_report.txt (this file)\n")

        # List other files that might be created
        for file_path in output_dir.glob("estimator_*"):
            if file_path.name not in ["estimator.yaml", "estimator_report.txt"]:
                f.write(f"  - {file_path.name}\n")

    print(f"Report saved to: {report_path}")


def setup_create_estimator_parser(subparsers):
    """Setup argument parser for create_estimator command."""
    parser = subparsers.add_parser(
        "create_estimator",
        help="Create and train a neural ratio estimator",
        description="""
        Create and train a neural ratio estimator from a saved simulator.
        
        This command reproduces Step 3 from gauss_gauss_train.ipynb:
        - Load simulator configuration
        - Create and train neural ratio estimator
        - Save trained estimator with parameters
        """,
    )

    # Required arguments
    parser.add_argument(
        "simulator_path", type=str, help="Path to saved simulator YAML configuration"
    )

    parser.add_argument(
        "output_dir", type=str, help="Output directory for saving trained estimator"
    )

    # Optional arguments
    parser.add_argument(
        "--template",
        type=str,
        default="mlp_default",
        help="Classifier template name (default: mlp_default)",
    )

    parser.add_argument(
        "--epsilon", type=float, help="Update simulator epsilon value before training"
    )

    parser.set_defaults(func=create_estimator_command)

    return parser
