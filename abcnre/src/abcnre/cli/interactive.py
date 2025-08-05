#!/usr/bin/env python3
"""
Interactive command for the ABC-NRE CLI.

Provides guided workflows for creating ABC-SBI configurations interactively.
"""

import argparse
from pathlib import Path
import sys
import os

# Add the src directory to the path to access abcnre modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abcnre.utils.interactive import (
    interactive_create_model,
    interactive_create_network,
    interactive_create_simulator,
    interactive_create_estimator,
    interactive_create_full_workflow,
)


def interactive_model_command(args):
    """Run interactive model creation."""
    output_dir = args.output_dir if args.output_dir else None
    config_path = interactive_create_model(output_dir)
    print(f"\nSuccess! Model configuration created at: {config_path}")


def interactive_network_command(args):
    """Run interactive network creation."""
    output_dir = args.output_dir if args.output_dir else None
    config_path = interactive_create_network(output_dir)
    print(f"\nSuccess! Network configuration created at: {config_path}")


def interactive_simulator_command(args):
    """Run interactive simulator creation."""
    output_dir = args.output_dir if args.output_dir else None
    model_path = args.model_config if hasattr(args, "model_config") else None
    config_path = interactive_create_simulator(model_path, output_dir)
    print(f"\nSuccess! Simulator configuration created at: {config_path}")


def interactive_estimator_command(args):
    """Run interactive estimator creation."""
    output_dir = args.output_dir if args.output_dir else None
    simulator_path = (
        args.simulator_config if hasattr(args, "simulator_config") else None
    )
    network_path = args.network_config if hasattr(args, "network_config") else None
    config_path = interactive_create_estimator(simulator_path, network_path, output_dir)
    print(f"\nSuccess! Estimator configuration created at: {config_path}")


def interactive_full_workflow_command(args):
    """Run interactive full workflow creation."""
    output_dir = args.output_dir if args.output_dir else None
    results = interactive_create_full_workflow(output_dir)
    print(f"\nSuccess! Full workflow created with {len(results)} configurations")


def setup_interactive_parsers(subparsers):
    """Setup all interactive command parsers."""

    # Interactive model
    model_parser = subparsers.add_parser(
        "interactive-model",
        help="Interactively create model configuration",
        description="Guided workflow for creating model configurations with validation and templates",
    )
    model_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output directory for configuration (default: creates timestamped directory)",
    )
    model_parser.set_defaults(
        func=interactive_model_command, command="interactive-model"
    )

    # Interactive network
    network_parser = subparsers.add_parser(
        "interactive-network",
        help="Interactively create network configuration",
        description="Guided workflow for creating network configurations with templates",
    )
    network_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output directory for configuration (default: creates timestamped directory)",
    )
    network_parser.set_defaults(
        func=interactive_network_command, command="interactive-network"
    )

    # Interactive simulator
    simulator_parser = subparsers.add_parser(
        "interactive-simulator",
        help="Interactively create simulator configuration",
        description="Guided workflow for creating simulator configurations",
    )
    simulator_parser.add_argument(
        "-m", "--model-config", type=str, help="Path to model configuration file"
    )
    simulator_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output directory for configuration (default: creates timestamped directory)",
    )
    simulator_parser.set_defaults(
        func=interactive_simulator_command, command="interactive-simulator"
    )

    # Interactive estimator
    estimator_parser = subparsers.add_parser(
        "interactive-estimator",
        help="Interactively create estimator configuration",
        description="Guided workflow for creating estimator configurations",
    )
    estimator_parser.add_argument(
        "-s",
        "--simulator-config",
        type=str,
        help="Path to simulator configuration file",
    )
    estimator_parser.add_argument(
        "-n", "--network-config", type=str, help="Path to network configuration file"
    )
    estimator_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output directory for configuration (default: creates timestamped directory)",
    )
    estimator_parser.set_defaults(
        func=interactive_estimator_command, command="interactive-estimator"
    )

    # Interactive full workflow
    workflow_parser = subparsers.add_parser(
        "interactive-workflow",
        help="Interactively create complete workflow",
        description="Complete guided workflow for creating all configurations (model, network, simulator, estimator)",
    )
    workflow_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Base output directory for workflow (default: creates timestamped directory)",
    )
    workflow_parser.set_defaults(
        func=interactive_full_workflow_command, command="interactive-workflow"
    )


# For standalone usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive ABC-SBI Configuration Tools"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available interactive commands"
    )

    setup_interactive_parsers(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    else:
        args.func(args)
