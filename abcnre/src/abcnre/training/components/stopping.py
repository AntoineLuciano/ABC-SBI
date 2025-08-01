"""
Stopping rules management - extracted from train.py

This module handles all stopping rules using the proper StoppingRulesConfig structure
from config.py with comprehensive coverage of all stopping criteria.
"""

from typing import Tuple, Optional
import logging

# Import the proper config classes
from ..config import (
    StoppingRulesConfig,
)

logger = logging.getLogger(__name__)


class StoppingRulesManager:
    """
    Manages all stopping rules using the proper StoppingRulesConfig structure.

    This class properly handles the dataclass structure from config.py instead of
    treating stopping rules as generic dicts or objects.
    """

    def __init__(self, training_config):
        """
        Initialize stopping rules manager.

        Args:
            training_config: TrainingConfig instance with stopping_rules attribute
        """
        self.training_config = training_config

        # Properly handle the stopping rules configuration
        self.stopping_rules = self._normalize_stopping_rules(
            training_config.stopping_rules
        )

        # Initialize tracking state for different stopping criteria
        self._convergence_history = []
        self._plateau_history = []
        self._early_stopping_history = []
        self._best_val_loss = float("inf")
        self._epochs_since_improvement = 0
        self._start_time = None

        if self.stopping_rules:
            active_criteria = self.stopping_rules.get_active_stopping_criteria()
            logger.debug(f"Initialized stopping rules with criteria: {active_criteria}")
        else:
            logger.debug("No stopping rules configured")

    def _normalize_stopping_rules(
        self, stopping_rules
    ) -> Optional[StoppingRulesConfig]:
        """
        Normalize stopping rules to proper StoppingRulesConfig instance.

        Args:
            stopping_rules: Can be dict, StoppingRulesConfig instance, or None

        Returns:
            StoppingRulesConfig instance or None
        """
        if stopping_rules is None:
            return None

        if isinstance(stopping_rules, StoppingRulesConfig):
            # Already properly configured
            return stopping_rules

        if isinstance(stopping_rules, dict):
            # Convert dict to proper dataclass structure
            try:
                return StoppingRulesConfig.from_dict(stopping_rules)
            except Exception as e:
                logger.warning(
                    f"Failed to convert stopping rules dict to StoppingRulesConfig: {e}"
                )
                logger.warning("Using default stopping rules")
                return StoppingRulesConfig()  # Use defaults

        logger.warning(f"Unexpected stopping rules type: {type(stopping_rules)}")
        return StoppingRulesConfig()  # Use defaults

    def check_before_epoch(
        self, epoch: int, n_samples_per_epoch: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if training should stop BEFORE starting the epoch.

        Uses the proper config structure to check sample stopping rules.

        Args:
            epoch: Current epoch number (0-based)
            n_samples_per_epoch: Number of samples per epoch

        Returns:
            (should_stop, reason)
        """
        if not self.stopping_rules:
            return False, None

        # Check sample stopping rule using proper dataclass
        if (
            self.stopping_rules.sample_stopping.enabled
            and self.stopping_rules.sample_stopping.max_samples is not None
        ):

            next_epoch_total_samples = (epoch + 1) * n_samples_per_epoch
            if (
                next_epoch_total_samples
                > self.stopping_rules.sample_stopping.max_samples
            ):
                reason = (
                    f"Sample stopping before epoch {epoch + 1}: "
                    f"Next epoch would use {next_epoch_total_samples} samples, "
                    f"exceeding max_samples {self.stopping_rules.sample_stopping.max_samples}"
                )
                return True, reason

        return False, None

    def check_after_epoch(
        self,
        epoch: int,
        total_samples: int,
        total_simulations: int,
        n_samples_per_epoch: int,
        current_loss: float,
        val_loss: float,
        current_lr: float,
        elapsed_time: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if training should stop AFTER completing the epoch.

        Uses proper config structure to check all stopping criteria.

        Args:
            epoch: Current epoch number (0-based)
            total_samples: Total samples processed so far
            total_simulations: Total simulations run so far
            n_samples_per_epoch: Samples per epoch
            current_loss: Current epoch training loss
            val_loss: Current epoch validation loss
            current_lr: Current learning rate
            elapsed_time: Elapsed training time in seconds

        Returns:
            (should_stop, reason)
        """
        if not self.stopping_rules:
            return False, None

        # Check max epochs (always active)
        if epoch + 1 >= self.stopping_rules.max_epochs:
            return True, f"Max epochs reached: {self.stopping_rules.max_epochs}"

        # Check sample stopping rule
        if (
            self.stopping_rules.sample_stopping.enabled
            and self.stopping_rules.sample_stopping.max_samples is not None
        ):
            if total_samples >= self.stopping_rules.sample_stopping.max_samples:
                reason = (
                    f"Sample stopping at epoch {epoch + 1}: "
                    f"Total samples {total_samples} >= max_samples {self.stopping_rules.sample_stopping.max_samples}"
                )
                return True, reason

        # Check simulation stopping rule
        if (
            self.stopping_rules.simulation_stopping.enabled
            and self.stopping_rules.simulation_stopping.max_simulations is not None
        ):
            if (
                total_simulations
                >= self.stopping_rules.simulation_stopping.max_simulations
            ):
                reason = (
                    f"Simulation stopping at epoch {epoch + 1}: "
                    f"Total simulations {total_simulations} >= max_simulations {self.stopping_rules.simulation_stopping.max_simulations}"
                )
                return True, reason

        # Check time stopping rule
        if (
            self.stopping_rules.time_stopping.enabled
            and self.stopping_rules.time_stopping.max_time_hours is not None
        ):
            elapsed_hours = elapsed_time / 3600
            if elapsed_hours >= self.stopping_rules.time_stopping.max_time_hours:
                reason = (
                    f"Time stopping at epoch {epoch + 1}: "
                    f"Elapsed time {elapsed_hours:.2f}h >= max_time {self.stopping_rules.time_stopping.max_time_hours}h"
                )
                return True, reason

        # Check LR stopping rule
        if (
            self.stopping_rules.lr_stopping.enabled
            and self.stopping_rules.lr_stopping.min_lr is not None
        ):
            if current_lr <= self.stopping_rules.lr_stopping.min_lr:
                reason = (
                    f"LR stopping at epoch {epoch + 1}: "
                    f"Current LR {current_lr:.2e} <= min_lr {self.stopping_rules.lr_stopping.min_lr:.2e}"
                )
                return True, reason

        # Check convergence stopping rule
        should_stop, reason = self._check_convergence_stopping(epoch, current_loss)
        if should_stop:
            return True, reason

        # Check plateau stopping rule
        should_stop, reason = self._check_plateau_stopping(epoch, val_loss)
        if should_stop:
            return True, reason

        # Check early stopping rule (validation-based)
        should_stop, reason = self._check_early_stopping(epoch, val_loss)
        if should_stop:
            return True, reason

        return False, None

    def _check_convergence_stopping(
        self, epoch: int, current_loss: float
    ) -> Tuple[bool, Optional[str]]:
        """Check convergence-based stopping using proper config."""
        if not self.stopping_rules.convergence_stopping.enabled:
            return False, None

        tolerance = self.stopping_rules.convergence_stopping.tolerance
        patience = self.stopping_rules.convergence_stopping.patience

        self._convergence_history.append(current_loss)

        # Keep only recent history
        if len(self._convergence_history) > patience + 1:
            self._convergence_history = self._convergence_history[-(patience + 1) :]

        # Need enough history to check convergence
        if len(self._convergence_history) < patience + 1:
            return False, None

        # Check if loss has converged (small changes over patience epochs)
        recent_losses = self._convergence_history[-patience:]
        loss_changes = [
            abs(recent_losses[i] - recent_losses[i - 1])
            for i in range(1, len(recent_losses))
        ]

        if all(change < tolerance for change in loss_changes):
            reason = (
                f"Convergence stopping at epoch {epoch + 1}: "
                f"Loss changes all below tolerance {tolerance} for {patience} epochs"
            )
            return True, reason

        return False, None

    def _check_plateau_stopping(
        self, epoch: int, val_loss: float
    ) -> Tuple[bool, Optional[str]]:
        """Check plateau-based stopping using proper config."""
        if not self.stopping_rules.plateau_stopping.enabled:
            return False, None

        patience = self.stopping_rules.plateau_stopping.patience
        threshold = self.stopping_rules.plateau_stopping.threshold

        self._plateau_history.append(val_loss)

        # Keep only recent history
        if len(self._plateau_history) > patience + 1:
            self._plateau_history = self._plateau_history[-(patience + 1) :]

        # Need enough history to check plateau
        if len(self._plateau_history) < patience + 1:
            return False, None

        # Check if we're on a plateau (no significant improvement over patience epochs)
        if len(self._plateau_history) <= patience:
            # Not enough history to determine plateau
            return False, None

        best_recent = min(self._plateau_history[-patience:])
        best_before = min(self._plateau_history[:-patience])

        if best_before - best_recent < threshold:
            reason = (
                f"Plateau stopping at epoch {epoch + 1}: "
                f"No improvement > {threshold} over {patience} epochs "
                f"(best recent: {best_recent:.6f}, best before: {best_before:.6f})"
            )
            return True, reason

        return False, None

    def _check_early_stopping(
        self, epoch: int, val_loss: float
    ) -> Tuple[bool, Optional[str]]:
        """Check early stopping based on validation loss."""
        if not self.stopping_rules.early_stopping.enabled:
            return False, None

        patience = self.stopping_rules.early_stopping.patience
        min_delta = self.stopping_rules.early_stopping.min_delta

        # Track validation loss history
        self._early_stopping_history.append(val_loss)

        # Check if this is the best validation loss we've seen
        if val_loss < self._best_val_loss - min_delta:
            self._best_val_loss = val_loss
            self._epochs_since_improvement = 0
        else:
            self._epochs_since_improvement += 1

        # Check if we should stop due to lack of improvement
        if self._epochs_since_improvement >= patience:
            reason = (
                f"Early stopping at epoch {epoch + 1}: "
                f"No improvement in validation loss for {patience} epochs "
                f"(best: {self._best_val_loss:.6f}, current: {val_loss:.6f})"
            )
            return True, reason

        return False, None

    def get_active_criteria(self) -> list[str]:
        """Get list of active stopping criteria for logging."""
        if not self.stopping_rules:
            return ["max_epochs"]

        return self.stopping_rules.get_active_stopping_criteria()

    def get_stopping_info(self) -> dict:
        """Get detailed info about stopping configuration for debugging."""
        if not self.stopping_rules:
            return {"enabled": False}

        return {
            "enabled": True,
            "max_epochs": self.stopping_rules.max_epochs,
            "active_criteria": self.get_active_criteria(),
            "config": self.stopping_rules.to_dict(),
        }

    def reset_state(self):
        """Reset internal state (useful for multiple training runs)."""
        self._convergence_history = []
        self._plateau_history = []
        self._early_stopping_history = []
        self._best_val_loss = float("inf")
        self._epochs_since_improvement = 0
        self._start_time = None
        logger.debug("Reset stopping rules state")


# Utility functions for stopping rules analysis
def get_stopping_summary(stopping_manager: StoppingRulesManager) -> dict:
    """
    Get a summary of stopping rules configuration and current state.

    Args:
        stopping_manager: StoppingRulesManager instance

    Returns:
        Dictionary with stopping rules summary
    """
    if not stopping_manager.stopping_rules:
        return {"enabled": False, "criteria": []}

    return {
        "enabled": True,
        "max_epochs": stopping_manager.stopping_rules.max_epochs,
        "active_criteria": stopping_manager.get_active_criteria(),
        "convergence_history_length": len(stopping_manager._convergence_history),
        "plateau_history_length": len(stopping_manager._plateau_history),
        "early_stopping_history_length": len(stopping_manager._early_stopping_history),
        "best_val_loss": stopping_manager._best_val_loss,
        "epochs_since_improvement": stopping_manager._epochs_since_improvement,
    }


def validate_stopping_rules_consistency(
    stopping_rules: StoppingRulesConfig,
) -> list[str]:
    """
    Validate stopping rules for potential conflicts or issues.

    Args:
        stopping_rules: StoppingRulesConfig instance

    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []

    if not stopping_rules:
        return warnings

    # Check for very low max_epochs with other criteria
    if stopping_rules.max_epochs < 10:
        active_criteria = stopping_rules.get_active_stopping_criteria()
        if len(active_criteria) > 1:
            warnings.append(
                f"Very low max_epochs ({stopping_rules.max_epochs}) with other stopping criteria active"
            )

    # Check convergence vs plateau patience
    if (
        stopping_rules.convergence_stopping.enabled
        and stopping_rules.plateau_stopping.enabled
    ):
        conv_patience = stopping_rules.convergence_stopping.patience
        plateau_patience = stopping_rules.plateau_stopping.patience
        if conv_patience >= plateau_patience:
            warnings.append(
                f"Convergence patience ({conv_patience}) >= plateau patience ({plateau_patience})"
            )

    # Check time stopping with very short times
    if (
        stopping_rules.time_stopping.enabled
        and stopping_rules.time_stopping.max_time_hours is not None
    ):
        if stopping_rules.time_stopping.max_time_hours < 0.1:  # Less than 6 minutes
            warnings.append(
                f"Very short max_time_hours: {stopping_rules.time_stopping.max_time_hours}"
            )

    return warnings
