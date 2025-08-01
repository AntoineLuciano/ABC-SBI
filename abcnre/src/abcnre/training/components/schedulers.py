"""
Learning rate scheduling management - unified approach including reduce-on-plateau.

This module handles all types of learning rate scheduling using the proper
LRSchedulerConfig structure from config.py with clean, professional logging.
"""

from typing import Tuple, Dict, Any
import logging

# Import proper config classes
from ..config import LRSchedulerConfig
from ..optimization import create_optimizer

logger = logging.getLogger(__name__)


class ReduceOnPlateauManager:
    """
    Manages reduce-on-plateau learning rate scheduling.
    
    Uses proper LRSchedulerConfig structure from config.py with robust
    parameter validation and state management.
    """
    
    def __init__(self, lr_scheduler_config: LRSchedulerConfig, base_learning_rate: float):
        """
        Initialize plateau manager from proper config structure.
        
        Args:
            lr_scheduler_config: LRSchedulerConfig instance from config.py
            base_learning_rate: Base learning rate from training config
        """
        if lr_scheduler_config.schedule_name != "reduce_on_plateau":
            raise ValueError(f"Expected reduce_on_plateau scheduler, got {lr_scheduler_config.schedule_name}")
        
        self.lr_scheduler_config = lr_scheduler_config
        self.best_loss = float("inf")
        self.plateau_counter = 0
        self.current_lr = base_learning_rate
        
        # Extract plateau parameters from proper config structure
        self.patience = lr_scheduler_config.schedule_args.get("patience", 5)
        self.factor = lr_scheduler_config.schedule_args.get("factor", 0.5)
        self.min_lr = lr_scheduler_config.schedule_args.get("min_lr", 1e-8)
        
        # Validate parameters
        self._validate_config()
        
        logger.debug(f"Initialized reduce-on-plateau scheduler: patience={self.patience}, factor={self.factor}, min_lr={self.min_lr}")
    
    def _validate_config(self):
        """Validate plateau configuration parameters."""
        if self.patience <= 0:
            raise ValueError("reduce_on_plateau patience must be positive")
        if not (0 < self.factor < 1):
            raise ValueError("reduce_on_plateau factor must be between 0 and 1")
        if self.min_lr <= 0:
            raise ValueError("reduce_on_plateau min_lr must be positive")
        if self.min_lr >= self.current_lr:
            logger.warning(f"min_lr ({self.min_lr}) >= base_lr ({self.current_lr})")
    
    def update(self, current_loss: float) -> Tuple[str, float, bool]:
        """
        Update plateau manager with current loss.
        
        Args:
            current_loss: Current epoch loss value
            
        Returns:
            (action, new_lr, should_stop)
            action: "continue", "reduce_lr", "early_stop"
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.plateau_counter = 0
            return "continue", self.current_lr, False
        else:
            self.plateau_counter += 1
            
            if self.plateau_counter >= self.patience:
                old_lr = self.current_lr
                new_lr = self.current_lr * self.factor
                
                if new_lr < self.min_lr:
                    if self.current_lr <= self.min_lr * 1.001:
                        # Already at minimum LR - early stopping
                        logger.info(f"Learning rate reached minimum: {self.current_lr:.2e}")
                        return "early_stop", self.current_lr, True
                    else:
                        self.current_lr = self.min_lr
                else:
                    self.current_lr = new_lr
                
                self.plateau_counter = 0
                logger.info(f"Learning rate reduced: {old_lr:.2e} -> {self.current_lr:.2e}")
                return "reduce_lr", self.current_lr, False
            
            return "continue", self.current_lr, False
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for debugging/logging."""
        return {
            "current_lr": self.current_lr,
            "best_loss": self.best_loss,
            "plateau_counter": self.plateau_counter,
            "patience": self.patience,
            "factor": self.factor,
            "min_lr": self.min_lr
        }


def compute_effective_learning_rate(
    lr_schedule, 
    lr_scheduler_config: LRSchedulerConfig, 
    step_counter: int, 
    current_lr: float
) -> float:
    """
    Compute the effective learning rate using proper config structure.
    
    Args:
        lr_schedule: The learning rate schedule (can be None)
        lr_scheduler_config: LRSchedulerConfig instance from config.py
        step_counter: Current step number
        current_lr: Current learning rate (for plateau scheduling)
        
    Returns:
        Effective learning rate for this step
    """
    if (lr_schedule is not None and 
        lr_scheduler_config.schedule_name != "reduce_on_plateau"):
        return float(lr_schedule(step_counter))
    else:
        return current_lr


def handle_lr_scheduling(
    lr_scheduler_config: LRSchedulerConfig,
    optimizer_type: str,
    weight_decay: float,
    current_loss: float,
    plateau_manager: ReduceOnPlateauManager,
    training_components: Dict[str, Any],
    training_logger
) -> Tuple[bool, Dict[str, Any]]:
    """
    Handle all learning rate scheduling logic using proper config structure.
    
    Args:
        lr_scheduler_config: LRSchedulerConfig instance from config.py
        optimizer_type: Type of optimizer
        weight_decay: Weight decay parameter
        current_loss: Current epoch loss
        plateau_manager: Plateau manager instance
        training_components: Dictionary of training components
        training_logger: Logger instance
        
    Returns:
        (should_stop_early, updated_training_components)
    """
    # Only handle reduce-on-plateau here, other schedulers are handled by optax
    if lr_scheduler_config.schedule_name != "reduce_on_plateau":
        return False, training_components
    
    # Update plateau manager
    action, new_lr, should_stop = plateau_manager.update(current_loss)
    
    if action == "early_stop":
        training_logger.log_early_stopping(-1, "Learning rate reached minimum")
        return True, training_components
    
    elif action == "reduce_lr":
        old_lr = training_components.get("current_lr", new_lr)
        training_logger.log_lr_reduction(old_lr, new_lr)
        
        # Recreate optimizer with new LR
        new_optimizer = create_optimizer(
            learning_rate=new_lr,
            optimizer_type=optimizer_type,
            weight_decay=weight_decay,
        )
        new_opt_state = new_optimizer.init(training_components["params"])
        
        # Create new training step function
        from .setup import create_training_step_function
        new_train_step = create_training_step_function(new_optimizer, training_components["loss_fn"])
        
        # Update components
        training_components.update({
            "optimizer": new_optimizer,
            "opt_state": new_opt_state,
            "train_step": new_train_step,
            "current_lr": new_lr
        })
    
    return False, training_components


def initialize_lr_scheduling(
    lr_scheduler_config: LRSchedulerConfig, 
    base_learning_rate: float
) -> Tuple[ReduceOnPlateauManager, Dict[str, Any]]:
    """
    Initialize learning rate scheduling components using proper config structure.
    
    Args:
        lr_scheduler_config: LRSchedulerConfig instance from config.py
        base_learning_rate: Base learning rate from training config
        
    Returns:
        (plateau_manager, scheduling_state)
    """
    plateau_manager = None
    scheduling_state = {}
    
    if lr_scheduler_config.schedule_name == "reduce_on_plateau":
        plateau_manager = ReduceOnPlateauManager(lr_scheduler_config, base_learning_rate)
        scheduling_state["current_lr"] = base_learning_rate
        logger.info("Initialized reduce-on-plateau scheduler")
    
    return plateau_manager, scheduling_state


def get_scheduler_info(lr_scheduler_config: LRSchedulerConfig, base_lr: float) -> Dict[str, Any]:
    """
    Get information about the scheduler configuration using proper config structure.
    
    Args:
        lr_scheduler_config: LRSchedulerConfig instance from config.py
        base_lr: Base learning rate
        
    Returns:
        Dictionary with scheduler information for logging/debugging
    """
    scheduler_info = {
        "type": lr_scheduler_config.schedule_name,
        "base_lr": base_lr,
        "args": lr_scheduler_config.schedule_args
    }
    
    if lr_scheduler_config.schedule_name == "reduce_on_plateau":
        scheduler_info.update({
            "patience": lr_scheduler_config.schedule_args.get("patience", 5),
            "factor": lr_scheduler_config.schedule_args.get("factor", 0.5),
            "min_lr": lr_scheduler_config.schedule_args.get("min_lr", 1e-8),
            "stateful": True
        })
    else:
        scheduler_info["stateful"] = False
    
    return scheduler_info


def is_reduce_on_plateau(lr_scheduler_config: LRSchedulerConfig) -> bool:
    """Check if using reduce-on-plateau scheduler."""
    return lr_scheduler_config.schedule_name == "reduce_on_plateau"


def validate_scheduler_config(lr_scheduler_config: LRSchedulerConfig, base_learning_rate: float):
    """
    Validate scheduler configuration using proper config structure.
    
    Args:
        lr_scheduler_config: LRSchedulerConfig instance from config.py
        base_learning_rate: Base learning rate from training config
        
    Raises:
        ValueError: If configuration is invalid
    """
    schedule_name = lr_scheduler_config.schedule_name
    schedule_args = lr_scheduler_config.schedule_args
    
    if base_learning_rate <= 0:
        raise ValueError("Base learning rate must be positive")
    
    if schedule_name == "reduce_on_plateau":
        patience = schedule_args.get("patience", 5)
        factor = schedule_args.get("factor", 0.5)
        min_lr = schedule_args.get("min_lr", 1e-8)
        
        if patience <= 0:
            raise ValueError("reduce_on_plateau patience must be positive")
        if not (0 < factor < 1):
            raise ValueError("reduce_on_plateau factor must be between 0 and 1")
        if min_lr <= 0:
            raise ValueError("reduce_on_plateau min_lr must be positive")
        if min_lr >= base_learning_rate:
            logger.warning(f"min_lr ({min_lr}) >= base_lr ({base_learning_rate})")
            
        logger.debug(f"Validated reduce-on-plateau: patience={patience}, factor={factor}, min_lr={min_lr}")
    
    elif schedule_name in ["cosine", "exponential", "constant"]:
        # These are handled by optax, basic validation only
        if schedule_name == "cosine":
            alpha = schedule_args.get("alpha", 0.0)
            if not (0 <= alpha <= 1):
                raise ValueError("cosine alpha must be between 0 and 1")
        
        elif schedule_name == "exponential":
            decay_rate = schedule_args.get("decay_rate", 0.9)
            if not (0 < decay_rate < 1):
                raise ValueError("exponential decay_rate must be between 0 and 1")
            
            decay_steps = schedule_args.get("decay_steps")
            if decay_steps is not None and decay_steps <= 0:
                raise ValueError("exponential decay_steps must be positive")
    
    else:
        logger.warning(f"Unknown scheduler type: {schedule_name}")


def create_lr_schedule_from_config(
    lr_scheduler_config: LRSchedulerConfig,
    base_learning_rate: float,
    num_epochs: int,
    num_steps_per_epoch: int
):
    """
    Create learning rate schedule from proper config structure.
    
    This replaces the create_learning_rate_schedule function to use proper config structure.
    
    Args:
        lr_scheduler_config: LRSchedulerConfig instance from config.py
        base_learning_rate: Base learning rate
        num_epochs: Number of training epochs
        num_steps_per_epoch: Steps per epoch
        
    Returns:
        Learning rate schedule or None for reduce_on_plateau
    """
    # Import here to avoid circular imports
    from ..train_old import create_learning_rate_schedule
    
    return create_learning_rate_schedule(
        schedule_name=lr_scheduler_config.schedule_name,
        base_learning_rate=base_learning_rate,
        num_epochs=num_epochs,
        num_steps_per_epoch=num_steps_per_epoch,
        **lr_scheduler_config.schedule_args
    )


# Utility functions for scheduler analysis
def get_lr_reduction_history(plateau_manager: ReduceOnPlateauManager) -> Dict[str, Any]:
    """
    Get history of learning rate reductions for analysis.
    
    Args:
        plateau_manager: ReduceOnPlateauManager instance
        
    Returns:
        Dictionary with reduction history information
    """
    if plateau_manager is None:
        return {}
    
    state = plateau_manager.get_state()
    return {
        "current_lr": state["current_lr"],
        "original_lr": state["current_lr"] / (state["factor"] ** max(0, state["plateau_counter"] - state["patience"])),
        "reductions_made": max(0, state["plateau_counter"] - state["patience"]),
        "plateau_counter": state["plateau_counter"],
        "best_loss_so_far": state["best_loss"]
    }


def should_continue_training(
    plateau_manager: ReduceOnPlateauManager,
    current_loss: float,
    min_improvement_threshold: float = 1e-6
) -> bool:
    """
    Determine if training should continue based on learning rate and improvement.
    
    Args:
        plateau_manager: ReduceOnPlateauManager instance
        current_loss: Current loss value
        min_improvement_threshold: Minimum improvement to consider significant
        
    Returns:
        True if training should continue, False if should stop
    """
    if plateau_manager is None:
        return True
    
    state = plateau_manager.get_state()
    
    # Stop if at minimum learning rate and no recent improvement
    if state["current_lr"] <= state["min_lr"] * 1.001:
        improvement = state["best_loss"] - current_loss
        if improvement < min_improvement_threshold:
            return False
    
    return True