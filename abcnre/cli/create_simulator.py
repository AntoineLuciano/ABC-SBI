"""
Create Simulator Command - Reproduces Steps 1-2 of gauss_gauss_train.ipynb
"""

import yaml
import jax
import numpy as np
from pathlib import Path
from typing import Optional

from abcnre.simulation import ABCSimulator
from abcnre.simulation.models import create_model_from_dict, get_example_model_configs
from abcnre.simulation import save_simulator_to_yaml
from abcnre.training import get_nn_config


def create_simulator_command(args):
    """
    Create and configure an ABC simulator.

    Reproduces Steps 1-2 from gauss_gauss_train.ipynb:
    - Step 1: Load model and create simulator
    - Step 1.5: Learn summary stats (optional)
    - Step 2: Sample observed data and set epsilon
    """
    print("=== ABC-NRE: Create Simulator ===")

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

    # === Step 1.5: Learn summary stats (optional) ===
    if args.learn_stats:
        if not args.regressor_config:
            raise ValueError(
                "--regressor-config is required when --learn-stats is enabled"
            )

        print("--- Step 1.5: Learn Summary Stats ---")

        # Load regressor configuration
        regressor_config = load_regressor_config(args.regressor_config)

        key, subkey_learn = jax.random.split(key)
        simulator.train_summary_network(subkey_learn, regressor_config)

        # Check correlation
        key, subkey_check = jax.random.split(key)
        correlation = simulator.check_summary_stats_correlation(
            subkey_check, n_samples=10000
        )
        print(f"Summary stats correlation: {correlation}")

    # === Step 2.1: Handle observed data ===
    print("--- Step 2.1: Set observed data ---")

    if args.observed_data:
        # Load pre-computed observed data
        x_obs = np.load(args.observed_data)
        print(f"Loaded observed data from: {args.observed_data}")
    else:
        # Sample new observed data from true_theta
        true_theta = args.true_theta
        key, subkey_sample = jax.random.split(key)
        x_obs = simulator.model.simulate_data(subkey_sample, true_theta)
        print(f"Sampled new observed data with true_theta={true_theta}")

    simulator.update_observed_data(x_obs)
    print(f"Observation x_obs shape: {x_obs.shape}")

    # === Step 2.2: Set epsilon ===
    print("--- Step 2.2: Set epsilon (ABC tolerance) ---")

    if args.epsilon is not None:
        # Manual epsilon
        simulator.epsilon = args.epsilon
        print(f"Set manual epsilon: {args.epsilon}")
    else:
        # Automatic epsilon from quantile
        key, subkey_epsilon = jax.random.split(key)
        simulator.set_epsilon_from_quantile(
            key=subkey_epsilon,
            quantile_distance=args.quantile_distance,
            n_samples=10000,
        )
        print(
            f"Set epsilon from {args.quantile_distance} quantile: {simulator.epsilon}"
        )

    # === Step 2.3: Save simulator ===
    print("--- Step 2.3: Save simulator ---")

    simulator_path = output_dir / "simulator.yaml"
    save_simulator_to_yaml(simulator, simulator_path, overwrite=True)

    print(f"✅ Simulator saved to: {simulator_path}")

    # Save summary report
    save_simulator_report(simulator, output_dir, args)

    return simulator


def load_regressor_config(config_path: str):
    """Load regressor configuration for summary statistics learning."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Regressor config not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Convert to NNConfig if it's a raw config dict
    if "network_name" in config_dict:
        # It's a simplified config, use get_nn_config
        nn_config = get_nn_config(
            network_name=config_dict["network_name"],
            network_size=config_dict.get("network_size", "default"),
            training_size=config_dict.get("training_size", "default"),
            task_type="summary_learner",
            lr_scheduler_name=config_dict.get("lr_scheduler_name", "cosine"),
            experiment_name=config_dict.get("experiment_name", "summary_learner"),
        )

        # Apply any overrides
        if "overrides" in config_dict:
            for key, value in config_dict["overrides"].items():
                if hasattr(nn_config.training, key):
                    setattr(nn_config.training, key, value)

        return nn_config
    else:
        # Assume it's a full NNConfig YAML
        from abcnre.training import NNConfig

        return NNConfig.load(config_path)


def save_simulator_report(simulator, output_dir: Path, args):
    """Save a human-readable summary of the simulator configuration."""
    report_path = output_dir / "simulator_report.txt"

    with open(report_path, "w") as f:
        f.write("=== ABC Simulator Configuration Report ===\n\n")
        f.write(f"Created on: {Path().cwd()}\n")
        f.write(f"Command args: {vars(args)}\n\n")

        f.write("Model Configuration:\n")
        f.write(f"  Model type: {simulator.model.__class__.__name__}\n")
        f.write(
            f"  Parameter dimension: {getattr(simulator.model, 'parameter_space_dim', 'Unknown')}\n\n"
        )

        f.write("Observed Data:\n")
        f.write(
            f"  Shape: {simulator.observed_data.shape if simulator.observed_data is not None else 'None'}\n"
        )
        f.write(f"  Epsilon: {simulator.epsilon}\n\n")

        f.write("Summary Statistics:\n")
        f.write(f"  Enabled: {simulator.config.get('summary_stats_enabled', False)}\n")
        if simulator.config.get("summary_stats_enabled", False):
            f.write(
                f"  Network available: {hasattr(simulator, '_summary_network') and simulator._summary_network is not None}\n"
            )

        f.write("\nFiles created:\n")
        f.write(f"  - simulator.yaml (main configuration)\n")
        f.write(f"  - simulator_report.txt (this file)\n")

    print(f"Report saved to: {report_path}")


def setup_create_simulator_parser(subparsers):
    """Setup argument parser for create_simulator command."""
    parser = subparsers.add_parser(
        "create_simulator",
        help="Create and configure an ABC simulator",
        description="""
        Create and configure an ABC simulator from a model configuration.
        
        This command reproduces Steps 1-2 from gauss_gauss_train.ipynb:
        - Load model and create simulator
        - Optionally learn summary statistics
        - Sample observed data and set epsilon
        """,
    )

    # Required arguments
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of example model (e.g., gauss_gauss_1d_default)",
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for saving simulator configuration",
    )

    # Optional arguments
    parser.add_argument(
        "--with-summary-stats",
        action="store_true",
        help="Learn summary statistics using a regressor network",
    )

    parser.add_argument(
        "--regressor-template",
        type=str,
        default="deepset_default",
        help="Regressor template name (default: deepset_default)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility (default: 123)",
    )

    parser.add_argument(
        "--true-theta",
        type=float,
        default=2.5,
        help="True parameter value for generating observed data (default: 2.5)",
    )

    parser.add_argument(
        "--epsilon", type=float, help="Manual epsilon value for ABC tolerance"
    )

    parser.add_argument(
        "--quantile-distance",
        type=float,
        default=1.0,
        help="Quantile for automatic epsilon determination (default: 1.0)",
    )

    parser.set_defaults(func=create_simulator_command)

    return parser
