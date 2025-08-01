"""
Interactive utilities for creating ABC-SBI configurations.

This module provides guided workflows for creating models, simulators, networks,
and estimators through command-line prompts with validation and default values.
"""

import os
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Import existing configurations and utilities
from ..simulation.models import get_example_model_configs, get_available_models
from ..training import get_nn_config
from ..training.config import NNConfig


class InteractiveSession:
    """Manages an interactive configuration session with navigation and validation."""

    def __init__(self):
        self.history = []
        self.current_step = 0

    def add_step(self, step_name: str, data: Dict[str, Any]):
        """Add a completed step to history."""
        self.history.append(
            {"step": step_name, "data": data, "timestamp": datetime.now().isoformat()}
        )
        self.current_step += 1

    def go_back(self) -> bool:
        """Go back to previous step if possible."""
        if self.current_step > 0:
            self.current_step -= 1
            self.history.pop()
            return True
        return False

    def get_current_data(self) -> Dict[str, Any]:
        """Get data from all completed steps."""
        result = {}
        for entry in self.history:
            result.update(entry["data"])
        return result


def prompt_with_default(
    prompt: str,
    default_value: Any = None,
    value_type: type = str,
    validator: callable = None,
    choices: List[str] = None,
) -> Any:
    """
    Prompt user for input with default value, type conversion, and validation.

    Args:
        prompt: Question to ask user
        default_value: Default value if user enters nothing
        value_type: Type to convert input to (str, int, float, bool)
        validator: Optional validation function
        choices: Optional list of valid choices

    Returns:
        Validated user input or default value
    """
    while True:
        # Display prompt with default
        if default_value is not None:
            display_prompt = f"{prompt} [{default_value}]: "
        else:
            display_prompt = f"{prompt}: "

        if choices:
            print(f"Available choices: {', '.join(choices)}")

        user_input = input(display_prompt).strip()

        # Use default if no input
        if not user_input and default_value is not None:
            # For choices, ensure default is valid
            if choices and str(default_value) not in choices:
                print(
                    f"Default value '{default_value}' not in choices: {', '.join(choices)}"
                )
                continue
            return default_value

        # Validate choices
        if choices and user_input not in choices:
            print(f"Invalid choice. Please select from: {', '.join(choices)}")
            continue

        # Type conversion
        try:
            if value_type == bool:
                if user_input.lower() in ["true", "t", "yes", "y", "1"]:
                    converted_value = True
                elif user_input.lower() in ["false", "f", "no", "n", "0"]:
                    converted_value = False
                else:
                    print("Please enter true/false, yes/no, or 1/0")
                    continue
            else:
                converted_value = value_type(user_input)
        except ValueError:
            print(f"Invalid input. Expected {value_type.__name__}")
            continue

        # Custom validation
        if validator:
            validation_result = validator(converted_value)
            if validation_result is not True:
                print(f"Validation error: {validation_result}")
                continue

        return converted_value


def validate_positive_number(value: Union[int, float]) -> Union[bool, str]:
    """Validate that a number is positive."""
    if value <= 0:
        return "Value must be positive"
    return True


def validate_probability(value: float) -> Union[bool, str]:
    """Validate that a value is a valid probability (0 <= p <= 1)."""
    if not (0 <= value <= 1):
        return "Value must be between 0 and 1"
    return True


def validate_dimension(value: int) -> Union[bool, str]:
    """Validate dimension parameter."""
    if value < 1:
        return "Dimension must be at least 1"
    if value > 1000:
        return "Dimension seems too large (>1000). Are you sure?"
    return True


def create_workflow_directory(name: str, base_dir: str = ".") -> Path:
    """
    Create a timestamped workflow directory.

    Args:
        name: Base name for the workflow
        base_dir: Base directory to create workflow in

    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{name}_{timestamp}"
    workflow_dir = Path(base_dir) / dir_name
    workflow_dir.mkdir(parents=True, exist_ok=True)
    return workflow_dir


def list_available_templates(config_type: str) -> List[str]:
    """
    List available configuration templates.

    Args:
        config_type: Type of config ('models', 'networks', 'training', 'lr_schedulers')

    Returns:
        List of available template names
    """
    if config_type == "models":
        # Get actual example config names, not registry keys
        try:
            example_configs = get_example_model_configs()
            if isinstance(example_configs, dict):
                return list(example_configs.keys())
            else:
                # If it returns a list or other format
                return example_configs
        except:
            # Fallback to known config names
            return [
                "gauss_gauss_1d_default",
                "gauss_gauss_2d_default",
                "gauss_gauss_100d_default",
            ]
    elif config_type == "networks":
        # Available network types based on the config system
        return ["mlp", "deepset", "conditioned_deepset"]
    elif config_type == "training":
        return ["training"]  # Only one training template with variants
    elif config_type == "lr_schedulers":
        return ["cosine", "exponential", "reduce_on_plateau", "constant"]
    else:
        return []


def load_config_template(config_type: str, template_name: str) -> Dict[str, Any]:
    """
    Load a configuration template from the examples/configs directory.

    Args:
        config_type: Type of config ('training', 'lr_schedulers')
        template_name: Name of the template file

    Returns:
        Loaded configuration dictionary
    """
    config_path = (
        Path(__file__).parent.parent.parent.parent
        / "examples"
        / "configs"
        / config_type
        / f"{template_name}.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(f"Template not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_template_variants(config_template: Dict[str, Any]) -> List[str]:
    """
    Get available variants from a configuration template.

    Args:
        config_template: Loaded template configuration

    Returns:
        List of available variant names
    """
    variants = config_template.get("variants", {})
    return list(variants.keys())


def build_config_from_template(
    config_template: Dict[str, Any], variant: str
) -> Dict[str, Any]:
    """
    Build a complete configuration from template and variant.

    Args:
        config_template: Loaded template configuration
        variant: Selected variant name

    Returns:
        Complete configuration dictionary
    """
    base_config = config_template.get("base", {}).copy()
    variant_config = config_template.get("variants", {}).get(variant, {})

    # Merge variant config into base config
    base_config.update(variant_config)
    return base_config


def interactive_create_model(output_dir: Optional[str] = None) -> str:
    """
    Interactive guide for creating a model configuration.

    Args:
        output_dir: Directory to save configuration (created if None)

    Returns:
        Path to created model configuration file
    """
    print("=" * 60)
    print("INTERACTIVE MODEL CONFIGURATION")
    print("=" * 60)

    session = InteractiveSession()

    # Step 1: Choose creation method
    print("\nStep 1: Model Creation Method")
    creation_method = prompt_with_default(
        "Create model from template or from scratch?",
        default_value="template",
        choices=["template", "scratch"],
    )

    if creation_method == "template":
        # Step 2: Choose template
        print("\nStep 2: Choose Model Template")
        available_models = list_available_templates("models")
        print("Available model templates:")
        for i, model in enumerate(available_models, 1):
            print(f"  {i}. {model}")

        template_choice = prompt_with_default(
            "Select template name",
            default_value=available_models[0] if available_models else None,
            choices=available_models,
        )

        # Load template configuration
        template_config = get_example_model_configs(template_choice)
        model_config = template_config.copy()

        print(f"\nLoaded template: {template_choice}")
        print("Current configuration:")
        print(yaml.dump(model_config, default_flow_style=False))

        # Step 3: Modify template parameters
        print("\nStep 3: Modify Parameters")
        modify = prompt_with_default(
            "Do you want to modify any parameters?", default_value="no", value_type=bool
        )

        if modify:
            model_config = _modify_model_parameters(model_config)

    else:
        # Create from scratch
        model_config = _create_model_from_scratch()

    # Step 4: Create output directory and save
    if output_dir is None:
        workflow_name = prompt_with_default(
            "Enter workflow name", default_value="my_model"
        )
        output_dir = create_workflow_directory(workflow_name)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "model_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(model_config, f, default_flow_style=False, sort_keys=False)

    print(f"\nModel configuration saved to: {config_path}")
    return str(config_path)


def _modify_model_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """Interactively modify model parameters."""
    model_args = config.get("model_args", {})

    print("\nAvailable parameters to modify:")
    for key, value in model_args.items():
        print(f"  {key}: {value}")

    while True:
        param_name = prompt_with_default(
            "Parameter to modify (or 'done' to finish)", default_value="done"
        )

        if param_name == "done":
            break

        if param_name not in model_args:
            print(f"Parameter '{param_name}' not found")
            continue

        current_value = model_args[param_name]
        current_type = type(current_value)

        # Specific validation based on parameter name
        validator = None
        if param_name in ["dim", "n_obs", "marginal_of_interest"]:
            validator = validate_positive_number
        elif param_name in ["sigma0", "sigma"]:
            validator = validate_positive_number

        new_value = prompt_with_default(
            f"New value for {param_name}",
            default_value=current_value,
            value_type=current_type,
            validator=validator,
        )

        model_args[param_name] = new_value
        print(f"Updated {param_name}: {current_value} -> {new_value}")

    config["model_args"] = model_args
    return config


def _create_model_from_scratch() -> Dict[str, Any]:
    """Create a model configuration from scratch."""
    print("\nCreating model from scratch...")
    print("Currently only GaussGaussMultiDimModel is supported")

    # Basic model structure
    config = {
        "model_type": "GaussGaussMultiDimModel",
        "model_class": "GaussGaussMultiDimModel",
        "model_args": {},
    }

    # Get model parameters
    args = config["model_args"]

    args["mu0"] = prompt_with_default(
        "Prior mean value", default_value=0.0, value_type=float
    )

    args["sigma0"] = prompt_with_default(
        "Prior standard deviation",
        default_value=1.0,
        value_type=float,
        validator=validate_positive_number,
    )

    args["sigma"] = prompt_with_default(
        "Model noise standard deviation",
        default_value=1.0,
        value_type=float,
        validator=validate_positive_number,
    )

    args["dim"] = prompt_with_default(
        "Parameter dimension",
        default_value=2,
        value_type=int,
        validator=validate_dimension,
    )

    args["n_obs"] = prompt_with_default(
        "Number of observations per sample",
        default_value=100,
        value_type=int,
        validator=validate_positive_number,
    )

    args["marginal_of_interest"] = prompt_with_default(
        "Index of marginal parameter of interest",
        default_value=0,
        value_type=int,
        validator=lambda x: (
            True if 0 <= x < args["dim"] else f"Must be between 0 and {args['dim']-1}"
        ),
    )

    return config


def interactive_create_network(output_dir: Optional[str] = None) -> str:
    """
    Interactive guide for creating a network configuration.

    Args:
        output_dir: Directory to save configuration

    Returns:
        Path to created network configuration file
    """
    print("=" * 60)
    print("INTERACTIVE NETWORK CONFIGURATION")
    print("=" * 60)

    # Step 1: Choose configuration level
    print("\nStep 1: Configuration Level")
    config_level = prompt_with_default(
        "Use simple configuration or advanced?",
        default_value="simple",
        choices=["simple", "advanced"],
    )

    if config_level == "simple":
        # Use existing templates
        print("\nStep 2: Choose Network Template")
        available_networks = list_available_templates("networks")
        print("Available network templates:")
        for i, network in enumerate(available_networks, 1):
            print(f"  {i}. {network}")

        network_choice = prompt_with_default(
            "Select network template",
            default_value=(
                available_networks[0] if available_networks else "mlp_default"
            ),
            choices=available_networks,
        )

        # Load and potentially modify template
        network_config = get_nn_config(network_choice)

        # Ask for experiment name
        experiment_name = prompt_with_default(
            "Experiment name", default_value="interactive_experiment"
        )

        # Create NNConfig
        nn_config = NNConfig.from_dict(network_config)
        nn_config.experiment_name = experiment_name

    else:
        # Advanced configuration
        nn_config = _create_advanced_network_config()

    # Save configuration
    if output_dir is None:
        workflow_name = prompt_with_default(
            "Enter workflow name", default_value="my_network"
        )
        output_dir = create_workflow_directory(workflow_name)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "network_config.yml"
    nn_config.save(config_path)

    print(f"\nNetwork configuration saved to: {config_path}")
    return str(config_path)


def _create_advanced_network_config() -> NNConfig:
    """Create advanced network configuration."""
    print("\nAdvanced Network Configuration")

    # Experiment name
    experiment_name = prompt_with_default(
        "Experiment name", default_value="advanced_experiment"
    )

    # Network type
    network_type = prompt_with_default(
        "Network type",
        default_value="mlp",
        choices=["mlp", "deepset", "conditioned_deepset"],
    )

    # Network arguments based on type
    network_args = {}

    if network_type == "mlp":
        # Get hidden dimensions
        hidden_dims_str = prompt_with_default(
            "Hidden dimensions (comma-separated)", default_value="64,32"
        )
        hidden_dims = [int(x.strip()) for x in hidden_dims_str.split(",")]
        network_args["hidden_dims"] = hidden_dims

        network_args["activation"] = prompt_with_default(
            "Activation function",
            default_value="relu",
            choices=["relu", "tanh", "gelu"],
        )

        network_args["use_layer_norm"] = prompt_with_default(
            "Use layer normalization?", default_value=True, value_type=bool
        )

        network_args["dropout_rate"] = prompt_with_default(
            "Dropout rate",
            default_value=0.0,
            value_type=float,
            validator=validate_probability,
        )

    # Task type
    task_type = prompt_with_default(
        "Task type",
        default_value="classifier",
        choices=["classifier", "summary_learner"],
    )

    # Create network config dict
    from ..training.config import NetworkConfig

    network_config = NetworkConfig(network_type=network_type, network_args=network_args)

    # Create NNConfig
    nn_config = NNConfig(
        experiment_name=experiment_name, network=network_config, task_type=task_type
    )

    return nn_config


def interactive_create_full_workflow(
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Complete interactive workflow for ABC-SBI pipeline.

    Full workflow steps:
    1. Create model configuration
    2. Optional: Learn summary statistics (create summary network + training)
    3. Simulate true data (specify path or true theta)
    4. Create simulator with epsilon value
    5. Create estimator with classifier network
    6. Training configuration
    7. Diagnostics (plots, metrics, SBC)

    Args:
        output_dir: Base directory for workflow

    Returns:
        Dictionary with paths to all created configurations
    """
    print("=" * 80)
    print("INTERACTIVE FULL ABC-SBI WORKFLOW")
    print("=" * 80)
    print("This workflow will guide you through the complete ABC-SBI pipeline:")
    print("1. Model creation")
    print("2. Optional summary statistics learning")
    print("3. True data simulation")
    print("4. Simulator creation")
    print("5. Estimator creation")
    print("6. Training configuration")
    print("7. Diagnostics setup")
    print("=" * 80)

    # Create workflow directory
    if output_dir is None:
        workflow_name = prompt_with_default(
            "Enter workflow name", default_value="abc_sbi_workflow"
        )
        workflow_dir = create_workflow_directory(workflow_name)
    else:
        workflow_dir = Path(output_dir)
        workflow_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating workflow in: {workflow_dir}")

    session = InteractiveSession()
    results = {}

    # STEP 1: Create model
    print("\n" + "=" * 60)
    print("STEP 1: MODEL CONFIGURATION")
    print("=" * 60)
    print("Define your statistical model (prior and likelihood)")

    results["model_config"] = interactive_create_model(str(workflow_dir))
    session.add_step("model", {"model_config": results["model_config"]})

    # Save checkpoint
    save_checkpoint = prompt_with_default(
        "Save checkpoint and continue?", default_value=True, value_type=bool
    )
    if not save_checkpoint:
        return results

    # STEP 2: Optional summary statistics learning
    print("\n" + "=" * 60)
    print("STEP 2: SUMMARY STATISTICS LEARNING (OPTIONAL)")
    print("=" * 60)
    print("You can learn summary statistics to reduce data dimensionality")

    learn_summary = prompt_with_default(
        "Do you want to learn summary statistics?", default_value=False, value_type=bool
    )

    if learn_summary:
        print("\nSTEP 2A: Summary Statistics Network")
        print("Creating network constrained to summary_learner task...")

        # Force summary_learner for this step
        results["summary_network_config"] = _create_summary_network(str(workflow_dir))
        session.add_step(
            "summary_network",
            {"summary_network_config": results["summary_network_config"]},
        )

        print("\nSTEP 2B: Summary Statistics Training")
        results["summary_training_config"] = _create_summary_training_config(
            model_path=results["model_config"],
            network_path=results["summary_network_config"],
            output_dir=str(workflow_dir),
        )
        session.add_step(
            "summary_training",
            {"summary_training_config": results["summary_training_config"]},
        )

        # Save checkpoint
        save_checkpoint = prompt_with_default(
            "Save checkpoint and continue?", default_value=True, value_type=bool
        )
        if not save_checkpoint:
            return results

    # STEP 3: True data simulation
    print("\n" + "=" * 60)
    print("STEP 3: TRUE DATA SIMULATION")
    print("=" * 60)
    print("Specify the true data for inference")

    results["true_data_config"] = _create_true_data_config(
        model_path=results["model_config"], output_dir=str(workflow_dir)
    )
    session.add_step("true_data", {"true_data_config": results["true_data_config"]})

    # Save checkpoint
    save_checkpoint = prompt_with_default(
        "Save checkpoint and continue?", default_value=True, value_type=bool
    )
    if not save_checkpoint:
        return results

    # STEP 4: Simulator creation
    print("\n" + "=" * 60)
    print("STEP 4: SIMULATOR CONFIGURATION")
    print("=" * 60)
    print("Configure the ABC simulator with epsilon tolerance")

    results["simulator_config"] = _create_simulator_with_epsilon(
        model_path=results["model_config"],
        summary_network_path=results.get("summary_network_config"),
        output_dir=str(workflow_dir),
    )
    session.add_step("simulator", {"simulator_config": results["simulator_config"]})

    # STEP 5: Estimator creation
    print("\n" + "=" * 60)
    print("STEP 5: ESTIMATOR CONFIGURATION")
    print("=" * 60)
    print("Create classifier network for likelihood ratio estimation")

    results["estimator_config"] = _create_estimator_with_classifier(
        simulator_path=results["simulator_config"], output_dir=str(workflow_dir)
    )
    session.add_step("estimator", {"estimator_config": results["estimator_config"]})

    # STEP 6: Training configuration
    print("\n" + "=" * 60)
    print("STEP 6: TRAINING CONFIGURATION")
    print("=" * 60)
    print("Configure training parameters for the classifier")

    results["training_config"] = _create_training_config(
        estimator_path=results["estimator_config"], output_dir=str(workflow_dir)
    )
    session.add_step("training", {"training_config": results["training_config"]})

    # Save checkpoint
    save_checkpoint = prompt_with_default(
        "Save checkpoint and continue?", default_value=True, value_type=bool
    )
    if not save_checkpoint:
        return results

    # STEP 7: Diagnostics setup
    print("\n" + "=" * 60)
    print("STEP 7: DIAGNOSTICS CONFIGURATION")
    print("=" * 60)
    print("Configure diagnostic tools (plots, metrics, SBC)")

    results["diagnostics_config"] = _create_diagnostics_config(
        estimator_path=results["estimator_config"],
        true_data_path=results["true_data_config"],
        output_dir=str(workflow_dir),
    )
    session.add_step(
        "diagnostics", {"diagnostics_config": results["diagnostics_config"]}
    )

    # Final summary
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"Workflow directory: {workflow_dir}")
    print(f"Completed {len(session.history)} steps")
    print("\nGenerated configurations:")
    for config_type, path in results.items():
        print(f"  {config_type}: {Path(path).name}")

    # Create execution script
    execution_script = _create_execution_script(results, workflow_dir)
    print(f"\nExecution script created: {execution_script}")

    return results


def _create_summary_network(output_dir: str) -> str:
    """Create a network configuration constrained to summary_learner task."""
    print("Creating summary statistics network (task_type: summary_learner)")

    # Force summary_learner task
    network_type = prompt_with_default(
        "Network type for summary statistics",
        default_value="mlp",
        choices=["mlp", "deepset", "conditioned_deepset"],
    )

    # Get network parameters
    network_args = {}

    if network_type == "mlp":
        hidden_dims_str = prompt_with_default(
            "Hidden dimensions for summary network (comma-separated)",
            default_value="128,64,32",
        )
        hidden_dims = [int(x.strip()) for x in hidden_dims_str.split(",")]
        network_args["hidden_dims"] = hidden_dims

        network_args["activation"] = prompt_with_default(
            "Activation function",
            default_value="relu",
            choices=["relu", "tanh", "gelu"],
        )

    # Create network config
    from ..training.config import NetworkConfig, NNConfig

    network_config = NetworkConfig(network_type=network_type, network_args=network_args)

    nn_config = NNConfig(
        experiment_name="summary_statistics_learning",
        network=network_config,
        task_type="summary_learner",  # Force summary_learner
    )

    # Save configuration
    config_path = Path(output_dir) / "summary_network_config.yml"
    nn_config.save(config_path)

    print(f"Summary network configuration saved to: {config_path}")
    return str(config_path)


def _create_summary_training_config(
    model_path: str, network_path: str, output_dir: str
) -> str:
    """Create training configuration for summary statistics learning using YAML templates."""
    print("Configuring summary statistics training...")

    # Load training template
    training_template = load_config_template("training", "training")
    available_variants = get_template_variants(training_template)

    print(f"Available training variants: {', '.join(available_variants)}")

    # Choose training variant
    training_variant = prompt_with_default(
        "Select training variant", default_value="fast", choices=available_variants
    )

    # Build training config from template
    training_config = build_config_from_template(training_template, training_variant)

    # Choose LR scheduler
    lr_scheduler_name = training_config.get("lr_scheduler_name", "cosine")
    available_schedulers = list_available_templates("lr_schedulers")

    lr_scheduler_choice = prompt_with_default(
        "Select learning rate scheduler",
        default_value=lr_scheduler_name,
        choices=available_schedulers,
    )

    # Load LR scheduler template
    scheduler_template = load_config_template("lr_schedulers", lr_scheduler_choice)
    scheduler_variants = get_template_variants(scheduler_template)

    print(
        f"Available {lr_scheduler_choice} scheduler variants: {', '.join(scheduler_variants)}"
    )

    scheduler_variant = prompt_with_default(
        f"Select {lr_scheduler_choice} scheduler variant",
        default_value="default",
        choices=scheduler_variants,
    )

    # Build scheduler config
    scheduler_config = build_config_from_template(scheduler_template, scheduler_variant)

    # Combine everything
    complete_config = {
        "model_config": model_path,
        "network_config": network_path,
        "task_type": "summary_learner",
        "training": training_config,
        "lr_scheduler": scheduler_config,
    }

    # Allow user to override specific parameters
    override_params = prompt_with_default(
        "Do you want to override any training parameters?",
        default_value=False,
        value_type=bool,
    )

    if override_params:
        print("Current training configuration:")
        print(yaml.dump(training_config, default_flow_style=False))

        # Override specific parameters
        if "learning_rate" in training_config:
            new_lr = prompt_with_default(
                "Override learning rate?",
                default_value=training_config["learning_rate"],
                value_type=float,
                validator=validate_positive_number,
            )
            complete_config["training"]["learning_rate"] = new_lr

        if "num_epochs" in training_config:
            new_epochs = prompt_with_default(
                "Override number of epochs?",
                default_value=training_config["num_epochs"],
                value_type=int,
                validator=validate_positive_number,
            )
            complete_config["training"]["num_epochs"] = new_epochs

    # Save configuration
    config_path = Path(output_dir) / "summary_training_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(complete_config, f, default_flow_style=False, sort_keys=False)

    print(f"Summary training configuration saved to: {config_path}")
    print(f"Using training variant: {training_variant}")
    print(f"Using LR scheduler: {lr_scheduler_choice} ({scheduler_variant})")
    return str(config_path)


def _create_true_data_config(model_path: str, output_dir: str) -> str:
    """Create configuration for true data simulation."""
    print("Configuring true data simulation...")

    # Data source choice
    data_source = prompt_with_default(
        "True data source",
        default_value="simulate",
        choices=["simulate", "file", "specify_theta"],
    )

    true_data_config = {}

    if data_source == "simulate":
        print("Simulating new true data from the model...")

        # Get true theta if needed
        specify_theta = prompt_with_default(
            "Do you want to specify the true theta parameter?",
            default_value=False,
            value_type=bool,
        )

        if specify_theta:
            # Load model to get dimension info
            with open(model_path, "r") as f:
                model_config = yaml.safe_load(f)

            dim = model_config.get("model_args", {}).get("dim", 2)

            if dim == 1:
                true_theta = prompt_with_default(
                    "True theta value", default_value=0.0, value_type=float
                )
            else:
                theta_str = prompt_with_default(
                    f"True theta values (comma-separated, {dim} values)",
                    default_value=",".join(["0.0"] * dim),
                )
                true_theta = [float(x.strip()) for x in theta_str.split(",")]

                if len(true_theta) != dim:
                    print(f"Warning: Expected {dim} values, got {len(true_theta)}")

            true_data_config["true_theta"] = true_theta

        # Number of observations
        n_obs = prompt_with_default(
            "Number of observations in true dataset",
            default_value=1000,
            value_type=int,
            validator=validate_positive_number,
        )
        true_data_config["n_obs"] = n_obs

    elif data_source == "file":
        data_path = prompt_with_default("Path to true data file", default_value="")
        true_data_config["data_path"] = data_path

    elif data_source == "specify_theta":
        # Load model config to get dimension
        with open(model_path, "r") as f:
            model_config = yaml.safe_load(f)

        dim = model_config.get("model_args", {}).get("dim", 2)

        if dim == 1:
            true_theta = prompt_with_default(
                "True theta value", default_value=0.0, value_type=float
            )
        else:
            theta_str = prompt_with_default(
                f"True theta values (comma-separated, {dim} values)",
                default_value=",".join(["0.0"] * dim),
            )
            true_theta = [float(x.strip()) for x in theta_str.split(",")]

        true_data_config["true_theta"] = true_theta

        # Still need to simulate observations
        n_obs = prompt_with_default(
            "Number of observations to simulate",
            default_value=1000,
            value_type=int,
            validator=validate_positive_number,
        )
        true_data_config["n_obs"] = n_obs

    true_data_config["model_config"] = model_path
    true_data_config["data_source"] = data_source

    # Save configuration
    config_path = Path(output_dir) / "true_data_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(true_data_config, f, default_flow_style=False, sort_keys=False)

    print(f"True data configuration saved to: {config_path}")
    return str(config_path)


def _create_simulator_with_epsilon(
    model_path: str, summary_network_path: Optional[str], output_dir: str
) -> str:
    """Create simulator configuration with epsilon tolerance."""
    print("Configuring ABC simulator...")

    # Epsilon value
    epsilon = prompt_with_default(
        "Epsilon tolerance value for ABC",
        default_value=0.1,
        value_type=float,
        validator=validate_positive_number,
    )

    # Number of simulations
    num_simulations = prompt_with_default(
        "Number of simulations to generate",
        default_value=100000,
        value_type=int,
        validator=validate_positive_number,
    )

    # Distance function
    distance_function = prompt_with_default(
        "Distance function",
        default_value="euclidean",
        choices=["euclidean", "manhattan", "cosine"],
    )

    # Create simulator config
    simulator_config = {
        "model_config": model_path,
        "epsilon": epsilon,
        "num_simulations": num_simulations,
        "distance_function": distance_function,
    }

    if summary_network_path:
        simulator_config["summary_network"] = summary_network_path
        print("Using learned summary statistics")

    # Save configuration
    config_path = Path(output_dir) / "simulator_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(simulator_config, f, default_flow_style=False, sort_keys=False)

    print(f"Simulator configuration saved to: {config_path}")
    return str(config_path)


def _create_estimator_with_classifier(simulator_path: str, output_dir: str) -> str:
    """Create estimator with classifier network."""
    print("Creating classifier network for likelihood ratio estimation...")

    # Network type for classifier
    network_type = prompt_with_default(
        "Network type for classifier",
        default_value="mlp",
        choices=["mlp", "deepset", "conditioned_deepset"],
    )

    # Network parameters
    network_args = {}

    if network_type == "mlp":
        hidden_dims_str = prompt_with_default(
            "Hidden dimensions for classifier (comma-separated)",
            default_value="256,128,64",
        )
        hidden_dims = [int(x.strip()) for x in hidden_dims_str.split(",")]
        network_args["hidden_dims"] = hidden_dims

        network_args["activation"] = prompt_with_default(
            "Activation function",
            default_value="relu",
            choices=["relu", "tanh", "gelu"],
        )

        network_args["dropout_rate"] = prompt_with_default(
            "Dropout rate",
            default_value=0.1,
            value_type=float,
            validator=validate_probability,
        )

    # Create network config
    from ..training.config import NetworkConfig, NNConfig

    network_config = NetworkConfig(network_type=network_type, network_args=network_args)

    nn_config = NNConfig(
        experiment_name="likelihood_ratio_estimation",
        network=network_config,
        task_type="classifier",  # Force classifier
    )

    # Save network configuration
    network_config_path = Path(output_dir) / "classifier_network_config.yml"
    nn_config.save(network_config_path)

    # Create estimator config
    estimator_config = {
        "simulator_config": simulator_path,
        "network_config": str(network_config_path),
        "task_type": "classifier",
    }

    # Save estimator configuration
    config_path = Path(output_dir) / "estimator_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(estimator_config, f, default_flow_style=False, sort_keys=False)

    print(f"Estimator configuration saved to: {config_path}")
    return str(config_path)


def _create_training_config(estimator_path: str, output_dir: str) -> str:
    """Create training configuration for the estimator using YAML templates."""
    print("Configuring classifier training...")

    # Load training template
    training_template = load_config_template("training", "training")
    available_variants = get_template_variants(training_template)

    print(f"Available training variants: {', '.join(available_variants)}")

    # Choose training variant
    training_variant = prompt_with_default(
        "Select training variant", default_value="default", choices=available_variants
    )

    # Build training config from template
    training_config = build_config_from_template(training_template, training_variant)

    # Choose LR scheduler
    lr_scheduler_name = training_config.get("lr_scheduler_name", "cosine")
    available_schedulers = list_available_templates("lr_schedulers")

    lr_scheduler_choice = prompt_with_default(
        "Select learning rate scheduler",
        default_value=lr_scheduler_name,
        choices=available_schedulers,
    )

    # Load LR scheduler template
    scheduler_template = load_config_template("lr_schedulers", lr_scheduler_choice)
    scheduler_variants = get_template_variants(scheduler_template)

    print(
        f"Available {lr_scheduler_choice} scheduler variants: {', '.join(scheduler_variants)}"
    )

    scheduler_variant = prompt_with_default(
        f"Select {lr_scheduler_choice} scheduler variant",
        default_value="default",
        choices=scheduler_variants,
    )

    # Build scheduler config
    scheduler_config = build_config_from_template(scheduler_template, scheduler_variant)

    # Combine everything
    complete_config = {
        "estimator_config": estimator_path,
        "training": training_config,
        "lr_scheduler": scheduler_config,
    }

    # Allow user to override specific parameters
    override_params = prompt_with_default(
        "Do you want to override any training parameters?",
        default_value=False,
        value_type=bool,
    )

    if override_params:
        print("Current training configuration:")
        print(yaml.dump(training_config, default_flow_style=False))

        # Override specific parameters
        if "learning_rate" in training_config:
            new_lr = prompt_with_default(
                "Override learning rate?",
                default_value=training_config["learning_rate"],
                value_type=float,
                validator=validate_positive_number,
            )
            complete_config["training"]["learning_rate"] = new_lr

        if "num_epochs" in training_config:
            new_epochs = prompt_with_default(
                "Override number of epochs?",
                default_value=training_config["num_epochs"],
                value_type=int,
                validator=validate_positive_number,
            )
            complete_config["training"]["num_epochs"] = new_epochs

        if "early_stopping_patience" in training_config:
            new_patience = prompt_with_default(
                "Override early stopping patience?",
                default_value=training_config["early_stopping_patience"],
                value_type=int,
                validator=validate_positive_number,
            )
            complete_config["training"]["early_stopping_patience"] = new_patience

    # Save configuration
    config_path = Path(output_dir) / "training_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(complete_config, f, default_flow_style=False, sort_keys=False)

    print(f"Training configuration saved to: {config_path}")
    print(f"Using training variant: {training_variant}")
    print(f"Using LR scheduler: {lr_scheduler_choice} ({scheduler_variant})")
    return str(config_path)


def _create_diagnostics_config(
    estimator_path: str, true_data_path: str, output_dir: str
) -> str:
    """Create diagnostics configuration."""
    print("Configuring diagnostics...")

    # Which diagnostics to run
    print("Select diagnostics to include:")

    run_plots = prompt_with_default(
        "Generate posterior comparison plots?", default_value=True, value_type=bool
    )

    run_metrics = prompt_with_default(
        "Compute evaluation metrics?", default_value=True, value_type=bool
    )

    run_sbc = prompt_with_default(
        "Run Simulation-Based Calibration (SBC)?", default_value=True, value_type=bool
    )

    # SBC configuration
    sbc_config = None
    if run_sbc:
        num_sbc_samples = prompt_with_default(
            "Number of SBC samples",
            default_value=100,
            value_type=int,
            validator=validate_positive_number,
        )

        sbc_config = {"num_samples": num_sbc_samples, "bins": 10}

    # Create diagnostics config
    diagnostics_config = {
        "estimator_config": estimator_path,
        "true_data_config": true_data_path,
        "diagnostics": {"plots": run_plots, "metrics": run_metrics, "sbc": run_sbc},
    }

    if sbc_config:
        diagnostics_config["diagnostics"]["sbc_config"] = sbc_config

    # Save configuration
    config_path = Path(output_dir) / "diagnostics_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(diagnostics_config, f, default_flow_style=False, sort_keys=False)

    print(f"Diagnostics configuration saved to: {config_path}")
    return str(config_path)


def _create_execution_script(results: Dict[str, str], workflow_dir: Path) -> str:
    """Create a script to execute the full workflow."""
    script_content = f"""#!/bin/bash
# ABC-SBI Workflow Execution Script
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Workflow directory: {workflow_dir}

set -e  # Exit on any error

echo "Starting ABC-SBI workflow execution..."
echo "Workflow directory: {workflow_dir}"
echo ""

"""

    # Add execution steps based on what was configured
    step_num = 1

    if "summary_training_config" in results:
        script_content += f"""
echo "Step {step_num}: Training summary statistics network..."
abcnre train --config {Path(results['summary_training_config']).name}
echo "Summary statistics training completed."
echo ""
"""
        step_num += 1

    if "true_data_config" in results:
        script_content += f"""
echo "Step {step_num}: Generating true data..."
abcnre simulate-true-data --config {Path(results['true_data_config']).name}
echo "True data generation completed."
echo ""
"""
        step_num += 1

    if "simulator_config" in results:
        script_content += f"""
echo "Step {step_num}: Running ABC simulations..."
abcnre create_simulator --config {Path(results['simulator_config']).name}
echo "ABC simulations completed."
echo ""
"""
        step_num += 1

    if "training_config" in results:
        script_content += f"""
echo "Step {step_num}: Training classifier network..."
abcnre train --config {Path(results['training_config']).name}
echo "Classifier training completed."
echo ""
"""
        step_num += 1

    if "diagnostics_config" in results:
        script_content += f"""
echo "Step {step_num}: Running diagnostics..."
abcnre run_diagnostics --config {Path(results['diagnostics_config']).name}
echo "Diagnostics completed."
echo ""
"""
        step_num += 1

    script_content += """
echo "Workflow execution completed successfully!"
echo "Check the results in the workflow directory."
"""

    # Save script
    script_path = workflow_dir / "run_workflow.sh"
    with open(script_path, "w") as f:
        f.write(script_content)

    # Make executable
    script_path.chmod(0o755)


def interactive_create_simulator(
    model_path: Optional[str] = None, output_dir: Optional[str] = None
) -> str:
    """
    Interactive guide for creating a simulator configuration.

    Args:
        model_path: Path to model configuration
        output_dir: Directory to save configuration

    Returns:
        Path to created simulator configuration file
    """
    print("=" * 60)
    print("INTERACTIVE SIMULATOR CONFIGURATION")
    print("=" * 60)

    if output_dir is None:
        workflow_name = prompt_with_default(
            "Enter workflow name", default_value="my_simulator"
        )
        output_dir = create_workflow_directory(workflow_name)

    # Use the detailed simulator creation function
    return _create_simulator_with_epsilon(
        model_path=model_path or "", summary_network_path=None, output_dir=output_dir
    )


def interactive_create_estimator(
    simulator_path: Optional[str] = None,
    network_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Interactive guide for creating an estimator configuration.

    Args:
        simulator_path: Path to simulator configuration
        network_path: Path to network configuration (ignored, creates classifier)
        output_dir: Directory to save configuration

    Returns:
        Path to created estimator configuration file
    """
    print("=" * 60)
    print("INTERACTIVE ESTIMATOR CONFIGURATION")
    print("=" * 60)

    if output_dir is None:
        workflow_name = prompt_with_default(
            "Enter workflow name", default_value="my_estimator"
        )
        output_dir = create_workflow_directory(workflow_name)

    # Use the detailed estimator creation function
    return _create_estimator_with_classifier(
        simulator_path=simulator_path or "", output_dir=output_dir
    )


# Export main functions
__all__ = [
    "interactive_create_model",
    "interactive_create_network",
    "interactive_create_simulator",
    "interactive_create_estimator",
    "interactive_create_full_workflow",
]
