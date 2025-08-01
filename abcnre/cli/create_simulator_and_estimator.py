"""
Create Simulator and Estimator Command - Combined workflow
"""

import argparse
import yaml
import jax
import numpy as np
from pathlib import Path
from typing import Optional

from abcnre.simulation import ABCSimulator
from abcnre.simulation.models import create_model_from_dict, get_example_model_configs
from abcnre.simulation import save_simulator_to_yaml
from abcnre.training import get_nn_config, NNConfig
from abcnre.inference import NeuralRatioEstimator
from abcnre.inference.io import save_estimator_to_yaml


def create_simulator_and_estimator_command(args):
    """
    Create and configure both ABC simulator and neural ratio estimator.

    Combines Steps 1-3 from gauss_gauss_train.ipynb:
    - Step 1: Load model and create simulator
    - Step 1.5: Learn summary stats (optional)
    - Step 2: Sample observed data and set epsilon
    - Step 3: Create and train neural ratio estimator
    """
    print("=== ABC-NRE: Create Simulator and Estimator ===")

    # Setup paths and seed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)
    print(f"Using random seed: {args.seed}")

    # === Step 1: Load model and create simulator ===
    print("--- Step 1: Load model and create simulator ---")

    if args.model_path:
        # Load custom model from path
        with open(args.model_path, "r") as f:
            model_config = yaml.safe_load(f)
        print(f"Loaded custom model from: {args.model_path}")
    else:
        # Load example model by name
        model_config = get_example_model_configs(args.model_name)
        print(f"Loaded example model: {args.model_name}")

    model = create_model_from_dict(model_config)
    simulator = ABCSimulator(model=model)
    print(f"Model loaded: {model}")

    # === Step 1.5: Learn summary stats (if enabled) ===
    if args.with_summary_stats:
        print("--- Step 1.5: Setup summary statistics learning ---")

        if args.regressor_config:
            with open(args.regressor_config, "r") as f:
                regressor_config = yaml.safe_load(f)
            print(f"Loaded regressor config from: {args.regressor_config}")
        else:
            regressor_config = get_nn_config("deepset_default")
            print("Using default regressor config")

        # Configure summary statistics learning
        simulator.config["summary_stats_enabled"] = True
        simulator.config["regressor_config"] = regressor_config
        print("Summary statistics learning enabled")

    # === Step 2: Sample observed data and set epsilon ===
    print("--- Step 2: Sample observed data and set epsilon ---")

    # Sample true parameters and generate observed data
    key, theta_key = jax.random.split(key)
    true_theta = model.get_prior_sample(theta_key)
    print(f"True theta: {true_theta}")

    key, data_key = jax.random.split(key)
    observed_data = model.simulate_data(data_key, true_theta)
    print(f"Observed data shape: {observed_data.shape}")

    # Set epsilon
    if args.quantile_distance is not None:
        print(f"Computing epsilon from quantile distance: {args.quantile_distance}")
        key, epsilon_key = jax.random.split(key)

        # Generate reference samples for epsilon calculation
        n_ref_samples = args.n_reference_samples
        print(f"Generating {n_ref_samples} reference samples...")

        distances = []
        for i in range(n_ref_samples):
            key, ref_key = jax.random.split(key)
            ref_theta = model.get_prior_sample(ref_key)
            ref_data = model.simulate_data(ref_key, ref_theta)
            distance = model.discrepancy_fn(observed_data, ref_data)
            distances.append(distance)

        distances = np.array(distances)
        epsilon = np.quantile(distances, args.quantile_distance)
        print(f"Computed epsilon: {epsilon}")
    else:
        epsilon = args.epsilon
        print(f"Using provided epsilon: {epsilon}")

    # Update simulator with observed data and epsilon
    simulator.observed_data = observed_data
    simulator.epsilon = epsilon
    simulator.config["true_theta"] = true_theta.tolist()
    print(f"Simulator configured with epsilon = {epsilon}")

    # Save simulator
    simulator_path = output_dir / "simulator.yaml"
    save_simulator_to_yaml(simulator, simulator_path, overwrite=True)
    print(f"Simulator saved to: {simulator_path}")

    # === Step 3: Create and train neural ratio estimator ===
    print("--- Step 3: Create and train neural ratio estimator ---")

    # Create network configuration
    if args.network_config:
        with open(args.network_config, "r") as f:
            nn_config_dict = yaml.safe_load(f)
        nn_config = NNConfig.from_dict(nn_config_dict)
        print(f"Loaded network config from: {args.network_config}")
    else:
        nn_config_dict = get_nn_config(args.network_name)
        nn_config = NNConfig.from_dict(nn_config_dict)
        print(f"Using default network config: {args.network_name}")

    # Override experiment name
    nn_config.experiment_name = (
        args.experiment_name or f"combined_{args.model_name}_{args.network_name}"
    )
    print(f"Experiment name: {nn_config.experiment_name}")

    # Create estimator
    estimator = NeuralRatioEstimator(nn_config=nn_config, simulator=simulator)
    print(f"Created estimator with network: {nn_config.network.network_type}")

    # Train estimator
    print("Training estimator...")
    key, train_key = jax.random.split(key)

    train_config = {
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "validation_fraction": args.validation_fraction,
        "verbose": True,
    }

    print(f"Training configuration: {train_config}")

    estimator.train(
        key=train_key,
        n_samples=train_config["n_samples"],
        batch_size=train_config["batch_size"],
        max_epochs=train_config["max_epochs"],
        learning_rate=train_config["learning_rate"],
        patience=train_config["patience"],
        validation_fraction=train_config["validation_fraction"],
        verbose=train_config["verbose"],
    )

    print("Training completed successfully!")

    # Save estimator
    estimator_path = output_dir / "estimator.yaml"
    save_estimator_to_yaml(estimator, estimator_path, overwrite=True)
    print(f"Estimator saved to: {estimator_path}")

    # === Summary ===
    print("--- Summary ---")
    print(f"✅ Model: {model.__class__.__name__}")
    print(f"✅ True theta: {true_theta}")
    print(f"✅ Observed data shape: {observed_data.shape}")
    print(f"✅ Epsilon: {epsilon}")
    print(f"✅ Network: {nn_config.network.network_type}")
    print(f"✅ Experiment: {nn_config.experiment_name}")
    print(f"✅ Simulator saved to: {simulator_path}")
    print(f"✅ Estimator saved to: {estimator_path}")

    if args.with_summary_stats:
        print(f"✅ Summary statistics: Enabled")

    print(f"\n🎉 Combined setup completed successfully!")
    print(f"📁 Output directory: {output_dir}")

    # Next steps suggestion
    print("\n--- Next Steps ---")
    print("You can now use these commands for evaluation:")
    print(f"  abcnre plot_posterior_comparison {estimator_path}")
    print(f"  abcnre compute_metrics {estimator_path}")
    print(f"  abcnre run_sbc {estimator_path} --n_samples 100")


def setup_create_simulator_and_estimator_parser(subparsers):
    """Setup argument parser for create_simulator_and_estimator command."""
    parser = subparsers.add_parser(
        "create_simulator_and_estimator",
        help="Create and train both simulator and estimator in one step",
        description="""
        Create an ABC simulator and neural ratio estimator in a single command.
        This combines the create_simulator and create_estimator workflows for convenience.
        
        Examples:
          # Basic usage with defaults
          abcnre create_simulator_and_estimator gauss_gauss_1d_default ./output
          
          # With custom epsilon and network
          abcnre create_simulator_and_estimator gauss_gauss_2d_default ./output \\
              --epsilon 0.1 --network-name mlp_small
          
          # With summary statistics learning
          abcnre create_simulator_and_estimator gauss_gauss_100d_default ./output \\
              --with-summary-stats --quantile-distance 0.1
          
          # Full customization
          abcnre create_simulator_and_estimator gauss_gauss_3d_correlated ./output \\
              --epsilon 0.05 --n-samples 10000 --max-epochs 100 \\
              --experiment-name my_experiment
        """,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=40
        ),
    )

    # === Required Arguments ===
    parser.add_argument(
        "model_name", help="Name of the example model configuration to use"
    )
    parser.add_argument("output_dir", help="Output directory for generated files")

    # === Model Configuration ===
    model_group = parser.add_argument_group("Model Configuration")
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
        help="Quantile for automatic epsilon calculation (alternative to --epsilon)",
    )
    sim_group.add_argument(
        "--n-reference-samples",
        type=int,
        default=1000,
        help="Number of reference samples for epsilon calculation (default: 1000)",
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
        "--network-config", help="Path to custom network configuration YAML file"
    )
    network_group.add_argument(
        "--experiment-name", help="Name for the experiment (default: auto-generated)"
    )

    # === Training Configuration ===
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of training samples (default: 10000)",
    )
    train_group.add_argument(
        "--batch-size", type=int, default=512, help="Training batch size (default: 512)"
    )
    train_group.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (default: 50)",
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    train_group.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (default: 10)"
    )
    train_group.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
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
            parser.error("Must provide either --epsilon or --quantile-distance")

        if args.epsilon is not None and args.quantile_distance is not None:
            parser.error("Cannot provide both --epsilon and --quantile-distance")

        if args.quantile_distance is not None:
            if not (0 < args.quantile_distance < 1):
                parser.error("--quantile-distance must be between 0 and 1")

    parser.set_defaults(
        func=create_simulator_and_estimator_command, validate=validate_args
    )

    return parser
