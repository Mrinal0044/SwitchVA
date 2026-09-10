"""Checkpoint management for saving and resuming model states."""

import os
import logging
from typing import Any, Dict, Optional
import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Handles saving, loading, and tracking top checkpoints based on validation metrics."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        monitor_metric: str = "val_ccc_mean",
        mode: str = "max",
        save_top_k: int = 3,
        best_filename: str = "best_model.pt",
        last_filename: str = "last_model.pt",
    ):
        """
        Args:
            checkpoint_dir: Directory to store model checkpoints.
            monitor_metric: Metric key to monitor.
            mode: "max" or "min".
            save_top_k: Number of best checkpoints to retain.
            best_filename: Filename for the best checkpoint.
            last_filename: Filename for the most recent checkpoint.
        """
        self.checkpoint_dir = checkpoint_dir
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.save_top_k = save_top_k
        self.best_filename = best_filename
        self.last_filename = last_filename
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.checkpoints = []

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def is_better(self, current: float, best: float) -> bool:
        """Check if current metric value is better than best."""
        if self.mode == "max":
            return current > best
        return current < best

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Save a checkpoint to disk.

        Args:
            model: PyTorch model.
            optimizer: Optimizer state.
            scheduler: LR scheduler state.
            epoch: Current epoch.
            metrics: Evaluation metrics dictionary.
            filename: Explicit checkpoint filename (optional).

        Returns:
            Saved file path.
        """
        if metrics is None:
            metrics = {}

        if filename is None:
            metric_val = metrics.get(self.monitor_metric, 0.0)
            filename = f"checkpoint_epoch_{epoch}_{self.monitor_metric}_{metric_val:.4f}.pt"

        save_path = os.path.join(self.checkpoint_dir, filename)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(state, save_path)
        logger.info("Saved checkpoint to %s", save_path)

        # Update latest checkpoint
        if self.last_filename:
            last_path = os.path.join(self.checkpoint_dir, self.last_filename)
            torch.save(state, last_path)

        # Check if this is the overall best
        current_metric = metrics.get(self.monitor_metric, None)
        if current_metric is not None and self.is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            best_path = os.path.join(self.checkpoint_dir, self.best_filename)
            torch.save(state, best_path)
            logger.info("Updated best model (%s = %.4f) at %s", self.monitor_metric, current_metric, best_path)

        return save_path

    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load state into model, optimizer, and scheduler.

        Args:
            checkpoint_path: Path to checkpoint file.
            model: PyTorch model.
            optimizer: Optional optimizer.
            scheduler: Optional scheduler.
            device: Target device.

        Returns:
            Checkpoint dictionary.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        map_location = device if device is not None else torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info("Successfully loaded checkpoint from %s (epoch %d)", checkpoint_path, checkpoint.get("epoch", 0))
        return checkpoint
