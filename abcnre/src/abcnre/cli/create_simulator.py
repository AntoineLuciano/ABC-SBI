"""
Create Simulator Command - Reproduces Steps 1-2 of gauss_gauss_train.ipynb
"""

import yaml
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Optional

from abcnre.simulation import ABCSimulator
from abcnre.simulation.models import create_model_from_dict, get_example_model_configs
from abcnre.simulation import save_simulator_to_yaml
from abcnre.cli.utils import add_boolean_flag, get_default_filename
from logging import getLogger

logger = getLogger(__name__)


template_config_path = (
    Path(__file__).parent.parent / "cli" / "templates" / "regressor_configs"
)


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
    logger.info(f"Using random seed: {args.seed}")

    # === Step 1: Load model and create simulator ===
    logger.info("--- Step 1: Load model and create simulator ---")

    if args.model_path:
        # Load custom model from path
        with open(args.model_path, "r") as f:
            model_config = yaml.safe_load(f)
        logger.info(f"Loaded custom model from: {args.model_path}")
    else:
        # Load example model by name
        model_config = get_example_model_configs(args.model_name)
        logger.info(f"Loaded example model: {args.model_name}")

    model = create_model_from_dict(model_config)

    # Override marginal_of_interest if specified
    if args.marginal_of_interest is not None:
        logger.info(f"Overriding marginal_of_interest to: {args.marginal_of_interest}")

        if hasattr(model, "marginal_of_interest"):
            if args.marginal_of_interest == -1:
                # Set to "all parameters" mode
                if hasattr(model, "parameter_of_interest"):
                    model.parameter_of_interest = "all"
                model.marginal_of_interest = -1
                model.phi_dim = model.parameter_dim
                logger.info("Set inference mode to 'all parameters' (phi = theta)")
            else:
                # Set to specific parameter
                model.marginal_of_interest = args.marginal_of_interest
                model.phi_dim = 1

                # Update parameter_of_interest for better logging
                if hasattr(model, "param_names") and args.marginal_of_interest < len(
                    model.param_names
                ):
                    param_name = model.param_names[args.marginal_of_interest]
                    model.parameter_of_interest = param_name
                    logger.info(
                        f"Set inference mode to parameter '{param_name}' (index {args.marginal_of_interest})"
                    )
                else:
                    logger.info(
                        f"Set inference mode to parameter index {args.marginal_of_interest}"
                    )
                    
        else:
            logger.warning(
                f"Model {type(model).__name__} does not support marginal_of_interest override"
            )
    simulator = ABCSimulator(model=model)

    # === Step 1.5: Learn summary stats (optional) ===
    if args.learn_summary_stats:
        logger.info("--- Step 1.5: Learn Summary Stats ---")

        if args.regressor_config_path is not None:
            from abcnre.training import NNConfig

            regressor_config = NNConfig.load(args.regressor_config_path)
        elif args.regressor_config_template_name is not None:
            from abcnre.training import TemplateConfig, get_nn_config_from_template

            template_path = (
                Path(template_config_path)
                / f"{args.regressor_config_template_name}.yaml"
            )
            print(template_path.resolve())
            template_config = TemplateConfig.load(template_path)
            regressor_config = get_nn_config_from_template(template_config)
        else:
            raise ValueError("No valid regressor configuration provided.")
      
        
        key, subkey_learn = jax.random.split(key)
        simulator.train_summary_network(subkey_learn, regressor_config)
        

        # Check correlation
        key, subkey_check = jax.random.split(key)
        n_samples_check_summary = 10000
        correlation = simulator.check_summary_stats_correlation(
            subkey_check, n_samples=n_samples_check_summary
        )
        logger.info(f"Summary stats correlation: {correlation}")

    # === Step 2.1: Handle observed data ===
    logger.info("--- Step 2.1: Set observed data ---")

    if args.observed_data_path is not None:
        # Load pre-computed observed data
        x_obs = np.load(args.observed_data_path)
        logger.info(f"Loaded observed data from: {args.observed_data_path}")
    elif args.true_theta is not None:
        # Use true_theta to generate observed data
        key, subkey_sample = jax.random.split(key)
        x_obs = simulator.model.sample_x(subkey_sample, jnp.array(args.true_theta))
        logger.info(f"Generated observed data from true_theta: {args.true_theta}")

    else:
        key, subkey_sample = jax.random.split(key)
        true_theta = simulator.model.sample_theta(subkey_sample)
        key, subkey_sample = jax.random.split(key)
        logger.info(f"Sampling x_obs : theta.shape: {true_theta.shape}")
        x_obs = simulator.model.sample_x(subkey_sample, true_theta)
        logger.info(f"Sampled new observed data with true_theta={true_theta}")

    simulator.update_observed_data(x_obs)
    logger.info(f"Observation x_obs shape: {x_obs.shape}")
    logger.info(f'Summary stats s(x_obs) shape: {simulator.observed_summary_stats.shape}')

    # === Step 2.2: Set epsilon ===
    logger.info("--- Step 2.2: Set epsilon (ABC tolerance) ---")

    if args.epsilon is not None:
        # Manual epsilon
        simulator.epsilon = args.epsilon
        logger.info(f"Set manual epsilon: {args.epsilon}")
    elif args.quantile_distance is not None:
        # Automatic epsilon from quantile
        key, subkey_epsilon = jax.random.split(key)
        simulator.set_epsilon_from_quantile(
            key=subkey_epsilon,
            quantile_distance=args.quantile_distance,
            n_samples=10000,
        )
        logger.info(
            f"Set epsilon from {args.quantile_distance} quantile: {simulator.epsilon}"
        )
    else:
        # Default epsilon = inf
        logger.info(f"Set default epsilon: {simulator.epsilon}")

    if args.save:
        # === Step 2.3: Save simulator ===
        logger.info("--- Step 2.3: Save simulator ---")

        simulator_path = output_dir / get_default_filename("simulator")
        save_simulator_to_yaml(simulator, simulator_path, overwrite=True)

        logger.info(f"Simulator saved to: {simulator_path}")

    # Save summary report
    save_simulator_report(simulator, output_dir, args)

    return simulator


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

    logger.info(f"Report saved to: {report_path}")


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
        "--model_path",
        help="Path to custom model configuration YAML file (overrides model_name)",
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for saving simulator configuration",
    )

    # Optional arguments
    parser.add_argument(
        "--learn_summary_stats",
        action="store_true",
        help="Learn summary statistics using a regressor network",
    )

    parser.add_argument(
        "--regressor_config_template_name",
        type=str,
        default="deepset_default",
        help="Regressor template name (default: deepset_default)",
    )

    parser.add_argument(
        "--regressor_config_path",
        type=str,
        default=None,
        help="Path to regressor configuration file (default: deepset_default.yaml)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility (default: 123)",
    )

    parser.add_argument(
        "--marginal_of_interest",
        type=int,
        default=None,
        help="Index of marginal parameter for inference (-1 for all parameters, 0+ for specific parameter index). "
        "For G&K: 0=A, 1=B, 2=g, 3=k, -1=all. For GaussGauss: 0=mu_1, 1=mu_2, etc., -1=all.",
    )

    parser.add_argument(
        "--true_theta",
        type=float,
        nargs="+",
        default=None,
        help="True parameter values for generating observed data (space-separated list of floats)",
    )

    parser.add_argument(
        "--observed_data_path",
        type=str,
        default=None,
        help="Path to observed data file (default: None)",
    )

    parser.add_argument(
        "--epsilon", type=float, help="Manual epsilon value for ABC tolerance"
    )

    parser.add_argument(
        "--quantile_distance",
        type=float,
        default=1.0,
        help="Quantile for automatic epsilon determination (default: 1.0)",
    )

    # Add standardized boolean flags
    add_boolean_flag(
        parser, "save", default=True, help_text="Save simulator configuration files"
    )

    parser.set_defaults(func=create_simulator_command)

    return parser
