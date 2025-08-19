#!/usr/bin/env python3
"""
CLI utilities for handling common patterns across CLI commands.
"""

from pathlib import Path
from typing import Union, Optional


# Standard default filenames for CLI commands
DEFAULT_FILENAMES = {
    "simulator": "simulator.yaml",
    "estimator": "estimator.yaml",
    "mcmc": "mcmc.npz",
    "sbc": "sbc_results.npz",
    "report": "report.txt",
    "plot_1d": "comparison_1d.png",
    "plot_2d": "comparison_2d.png",
    "mcmc_output": "mcmc_output.png",
    "posterior_comparison": "posterior_comparison.png",
}


def handle_output_path(
    output_path: Union[str, Path], default_filename: str, create_parent: bool = True
) -> Path:
    """
    Handle output path resolution with automatic filename completion.

    Args:
        output_path: User-provided output path
        default_filename: Default filename to use if only directory is provided
        create_parent: Whether to create parent directories

    Returns:
        Resolved output path

    Examples:
        handle_output_path("/path/to/dir", "simulator.yaml") -> "/path/to/dir/simulator.yaml"
        handle_output_path("/path/to/file.yaml", "simulator.yaml") -> "/path/to/file.yaml"
    """
    output_path = Path(output_path)

    # If the path has no suffix (extension), treat it as a directory
    # and append the default filename
    if not output_path.suffix:
        output_path = output_path / default_filename

    # Create parent directories if requested
    if create_parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    return output_path


def get_default_filename(file_type: str) -> str:
    """
    Get the standard default filename for a given file type.

    Args:
        file_type: Type of file (e.g., 'simulator', 'estimator', 'mcmc')

    Returns:
        Default filename for that type

    Raises:
        KeyError: If file_type is not recognized

    Examples:
        get_default_filename("simulator") -> "simulator.yaml"
        get_default_filename("mcmc") -> "mcmc.npz"
    """
    if file_type not in DEFAULT_FILENAMES:
        raise KeyError(
            f"Unknown file type '{file_type}'. Available types: {list(DEFAULT_FILENAMES.keys())}"
        )

    return DEFAULT_FILENAMES[file_type]


def add_boolean_flag(parser, flag_name: str, default: bool = True, help_text: str = ""):
    """
    Add standardized boolean flags with --flag/--no-flag pattern.

    Args:
        parser: ArgumentParser to add flags to
        flag_name: Base name of the flag (e.g., 'save', 'show')
        default: Default value
        help_text: Base help text
    """
    group = parser.add_mutually_exclusive_group()

    # Positive flag
    group.add_argument(
        f"--{flag_name}",
        dest=flag_name,
        action="store_true",
        default=default,
        help=(
            f"{help_text} (default: {default})" if help_text else f"Enable {flag_name}"
        ),
    )

    # Negative flag
    group.add_argument(
        f"--no-{flag_name}",
        dest=flag_name,
        action="store_false",
        help=f"Disable {flag_name}",
    )
