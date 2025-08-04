"""
Metrics and logging management - extracted from train.py

This module handles all metrics collection, storage, and logging functionality
with clear separation of concerns.
"""

import jax
import jax.numpy as jnp
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TrainingMetrics:
    """
    Manages training metrics collection and storage.

    Handles both classifier and summary learner metrics with automatic
    validation and clean API for adding/retrieving metrics.
    """

    def __init__(self, task_type: str):
        """
        Initialize metrics manager.

        Args:
            task_type: "classifier" or "regressor"
        """
        if task_type not in ["classifier", "regressor"]:
            raise ValueError(f"Unknown task_type: {task_type}")

        self.task_type = task_type

        # Base metrics for all tasks
        self.metrics = {
            "learning_rate": [],
            "total_simulations": [],
            "train_loss": [],
            "val_loss": [],
        }

        # Task-specific metrics
        if task_type == "classifier":
            self.metrics["train_accuracy"] = []
            self.metrics["val_accuracy"] = []

        self._epoch_count = 0

        logger.debug(f"Initialized TrainingMetrics for {task_type}")

    def add_epoch_metrics(self, epoch_data: Dict[str, Any]):
        """
        Add metrics for a complete epoch.

        Args:
            epoch_data: Dictionary with epoch metrics
                Required keys: train_loss, val_loss, learning_rate, total_simulations
                Optional keys: train_accuracy, val_accuracy (for classifier)
        """
        # Validate required keys
        required_keys = ["train_loss", "val_loss", "learning_rate", "total_simulations"]
        for key in required_keys:
            if key not in epoch_data:
                raise ValueError(f"Missing required metric: {key}")

        # Add base metrics
        self.metrics["train_loss"].append(float(epoch_data["train_loss"]))
        self.metrics["val_loss"].append(float(epoch_data["val_loss"]))
        self.metrics["learning_rate"].append(float(epoch_data["learning_rate"]))
        self.metrics["total_simulations"].append(int(epoch_data["total_simulations"]))

        # Add task-specific metrics
        if self.task_type == "classifier":
            if "val_accuracy" in epoch_data:
                self.metrics["val_accuracy"].append(float(epoch_data["val_accuracy"]))
            if "train_accuracy" in epoch_data:
                self.metrics["train_accuracy"].append(
                    float(epoch_data["train_accuracy"])
                )

        self._epoch_count += 1

        # Log metrics periodically
        if self._epoch_count % 50 == 0:
            logger.debug(f"Stored metrics for {self._epoch_count} epochs")

    def get_latest(self, metric_name: str) -> Optional[float]:
        """
        Get latest value for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Latest value or None if metric doesn't exist/is empty
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return self.metrics[metric_name][-1]

    def get_history(self, metric_name: str) -> list:
        """
        Get complete history for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            List of all values for this metric
        """
        return self.metrics.get(metric_name, [])

    def get_summary_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get summary statistics for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary with min, max, mean, std
        """
        history = self.get_history(metric_name)
        if not history:
            return {}

        import numpy as np

        values = np.array(history)

        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "count": len(values),
        }

    def get_final_summary(self, total_time: float) -> Dict[str, Any]:
        """
        Get final training summary for TrainingResult.

        Args:
            total_time: Total training time in seconds

        Returns:
            Dictionary with final training summary
        """
        final_loss = self.get_latest("train_loss")
        final_simulations = self.get_latest("total_simulations")

        summary = {
            "final_loss": final_loss if final_loss is not None else float("inf"),
            "total_simulations": (
                final_simulations if final_simulations is not None else 0
            ),
            "total_time": total_time,
            "epochs_completed": self._epoch_count,
        }

        # Add classifier-specific final metrics
        if self.task_type == "classifier":
            final_val_acc = self.get_latest("val_accuracy")
            if final_val_acc is not None:
                summary["final_val_accuracy"] = final_val_acc

        return summary

    def has_improved(self, metric_name: str, mode: str = "min") -> bool:
        """
        Check if metric has improved in the latest epoch.

        Args:
            metric_name: Name of the metric to check
            mode: "min" for metrics that should decrease, "max" for metrics that should increase

        Returns:
            True if metric improved, False otherwise
        """
        history = self.get_history(metric_name)
        if len(history) < 2:
            return False

        current = history[-1]
        previous = history[-2]

        if mode == "min":
            return current < previous
        else:  # mode == "max"
            return current > previous

    def get_best_epoch(self, metric_name: str, mode: str = "min") -> Dict[str, Any]:
        """
        Get epoch information when metric was at its best.

        Args:
            metric_name: Name of the metric
            mode: "min" for metrics that should be minimized, "max" for maximized

        Returns:
            Dictionary with best epoch info
        """
        history = self.get_history(metric_name)
        if not history:
            return {}

        import numpy as np

        values = np.array(history)

        if mode == "min":
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        return {
            "epoch": best_idx,
            "value": float(values[best_idx]),
            "improvement_from_start": (
                float(abs(values[best_idx] - values[0])) if len(values) > 0 else 0
            ),
        }


class TrainingLogger:
    """
    Manages training progress logging with appropriate verbosity levels.

    Handles formatted logging of training progress, metrics, and important events
    with clean separation from metrics storage.
    """

    def __init__(self, verbose: bool, task_type: str):
        """
        Initialize training logger.

        Args:
            verbose: Whether to enable verbose logging
            task_type: "classifier" or "regressor"
        """
        self.verbose = verbose
        self.task_type = task_type

        if not verbose:
            # Reduce JAX logging noise when not verbose
            logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

    def log_training_start(
        self, task_type: str, num_epochs: int, batch_size: int, n_batch: int
    ):
        """Log training start information."""
        if self.verbose:
            logger.info("=" * 60)
            logger.info(f"Starting {task_type} training")
            logger.info(
                f"Epochs: {num_epochs}, Batch size: {batch_size}, Batches/epoch: {n_batch}"
            )
            logger.info("=" * 60)

    def log_epoch_progress(
        self,
        epoch: int,
        num_epochs: int,
        metrics: TrainingMetrics,
        elapsed_time: float,
        total_samples: int,
    ):
        """
        Log progress every N epochs.

        Args:
            epoch: Current epoch number
            num_epochs: Total number of epochs
            metrics: TrainingMetrics instance
            elapsed_time: Elapsed time in seconds
            total_samples: Total samples processed
        """
        if not self.verbose or epoch % 10 != 0:
            return

        train_loss = metrics.get_latest("train_loss")
        val_loss = metrics.get_latest("val_loss")
        learning_rate = metrics.get_latest("learning_rate")
        total_simulations = metrics.get_latest("total_simulations")

        # Progress percentage
        progress = (epoch + 1) / num_epochs * 100

        log_msg = (
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Loss: train = {train_loss:.6f} val = {val_loss:.6f} | "
        )

        # Add accuracy for classifier
        if self.task_type == "classifier":
            train_acc = metrics.get_latest("train_accuracy")
            val_acc = metrics.get_latest("val_accuracy")
            if train_acc is not None and val_acc is not None:
                log_msg += f"Acc: train = {train_acc:.2%} val = {val_acc:.2%} | "

        log_msg += f"LR: {learning_rate:.2e} | " f"Samples: {total_samples:,} | "
        if total_simulations != total_samples:
            log_msg += f"Sims: {total_simulations:,} | "

        log_msg += f"Time: {elapsed_time:.1f}s"

        logger.info(log_msg)

    def log_lr_reduction(self, old_lr: float, new_lr: float):
        """Log learning rate reduction."""
        if self.verbose:
            reduction_factor = new_lr / old_lr
            logger.info(
                f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e} ({reduction_factor:.2f}x)"
            )

    def log_early_stopping(self, epoch: int, reason: str):
        """Log early stopping event."""
        if self.verbose:
            logger.info(f"Early stopping at epoch {epoch + 1}: {reason}")

    def log_training_completion(self, metrics: TrainingMetrics, total_time: float):
        """Log final training statistics."""
        if not self.verbose:
            return

        final_loss = metrics.get_latest("train_loss")
        final_val_loss = metrics.get_latest("val_loss")
        total_simulations = metrics.get_latest("total_simulations")

        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        logger.info(f"Epochs completed: {metrics._epoch_count}")

        if final_loss is not None:
            logger.info(f"Final train loss: {final_loss:.6f}")

        if final_val_loss is not None:
            logger.info(f"Final val loss: {final_val_loss:.6f}")

        if self.task_type == "classifier":
            final_val_acc = metrics.get_latest("val_accuracy")
            if final_val_acc is not None:
                logger.info(f"Final val accuracy: {final_val_acc:.1%}")

        if total_simulations is not None:
            logger.info(f"Total simulations: {total_simulations:,}")

        # Show best performance
        best_val_loss = metrics.get_best_epoch("val_loss", mode="min")
        if best_val_loss:
            logger.info(
                f"Best val loss: {best_val_loss['value']:.6f} (epoch {best_val_loss['epoch']})"
            )

        logger.info("=" * 60)

    def log_metrics_summary(self, metrics: TrainingMetrics):
        """Log summary of all collected metrics."""
        if not self.verbose:
            return

        logger.info("Training Metrics Summary:")

        for metric_name in ["train_loss", "val_loss"]:
            stats = metrics.get_summary_stats(metric_name)
            if stats:
                logger.info(
                    f"   {metric_name}: min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}"
                )

        if self.task_type == "classifier":
            for metric_name in ["train_accuracy", "val_accuracy"]:
                stats = metrics.get_summary_stats(metric_name)
                if stats:
                    logger.info(
                        f"   {metric_name}: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}"
                    )


def evaluate_batch_metrics(
    task_type: str, params, batch_data, network, loss_fn
) -> Dict[str, float]:
    """
    Evaluate metrics on a single batch.

    Args:
        task_type: "classifier" or "regressor"
        params: Network parameters
        batch_data: Batch data dictionary
        network: Network instance
        loss_fn: Loss function

    Returns:
        Dictionary with computed metrics
    """
    if task_type != "classifier":
        return {}

    # Compute accuracy for classifier
    batch_input = batch_data["input"]
    batch_output = batch_data["output"]

    try:
        logits = network.apply(params, batch_input, training=False)
        probs = jax.nn.sigmoid(logits.squeeze(-1))
        predictions = (probs > 0.5).astype(jnp.float32)
        accuracy = jnp.mean(predictions == batch_output)

        return {"accuracy": float(accuracy)}

    except Exception as e:
        logger.warning(f"Failed to compute batch metrics: {e}")
        return {"accuracy": 0.0}


def log_config_summary(config, verbose: bool = True):
    """
    Log a summary of the training configuration.

    Args:
        config: NNConfig instance
        verbose: Whether to log the summary
    """
    if not verbose:
        return

    logger.info("Configuration Summary:")
    logger.info(f"   Task: {config.task_type}")
    logger.info(f"   Network: {config.network.network_type}")
    logger.info(f"   Optimizer: {config.training.optimizer}")
    logger.info(f"   Learning rate: {config.training.learning_rate}")
    logger.info(f"   LR scheduler: {config.training.lr_scheduler.schedule_name}")
    logger.info(f"   Batch size: {config.training.batch_size}")
    logger.info(f"   Samples/epoch: {config.training.n_samples_per_epoch}")
    logger.info(f"   Max epochs: {config.training.num_epochs}")

    # Stopping rules info
    if hasattr(config.training, "stopping_rules") and config.training.stopping_rules:
        logger.info("   Stopping rules: Enabled")
    else:
        logger.info("   Stopping rules: Disabled")

    # Phi storage info
    n_phi_to_store = getattr(config.training, "n_phi_to_store", 0)
    if n_phi_to_store > 0:
        logger.info(f"   Phi storage: {n_phi_to_store} values")


# Utility functions for metrics analysis
def compute_metrics_delta(
    metrics: TrainingMetrics, metric_name: str, window: int = 10
) -> float:
    """
    Compute the change in a metric over a recent window.

    Args:
        metrics: TrainingMetrics instance
        metric_name: Name of the metric
        window: Number of recent epochs to consider

    Returns:
        Change in metric (positive = increasing, negative = decreasing)
    """
    history = metrics.get_history(metric_name)
    if len(history) < window:
        return 0.0

    recent_avg = sum(history[-window:]) / window
    older_avg = (
        sum(history[-2 * window : -window]) / window
        if len(history) >= 2 * window
        else history[0]
    )

    return recent_avg - older_avg


def is_training_stagnant(
    metrics: TrainingMetrics, patience: int = 20, threshold: float = 1e-6
) -> bool:
    """
    Check if training appears to be stagnant.

    Args:
        metrics: TrainingMetrics instance
        patience: Number of epochs to check
        threshold: Minimum change to consider as progress

    Returns:
        True if training appears stagnant
    """
    val_loss_delta = abs(compute_metrics_delta(metrics, "val_loss", patience))
    return val_loss_delta < threshold
