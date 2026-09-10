"""Device management utilities supporting CUDA, MPS, and CPU."""

import logging
import torch

logger = logging.getLogger(__name__)


def get_device(device_str: str = "auto") -> torch.device:
    """Select the best available compute device without hardcoded GPU-only assumptions.

    Args:
        device_str: "auto", "cuda", "mps", "cpu", or specific device index (e.g. "cuda:0").

    Returns:
        torch.device instance.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Auto-selected device: CUDA (%s)", torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Auto-selected device: Apple Silicon MPS")
        else:
            device = torch.device("cpu")
            logger.info("Auto-selected device: CPU")
        return device

    try:
        device = torch.device(device_str)
        logger.info("Using explicitly requested device: %s", device)
        return device
    except Exception as e:
        logger.warning("Failed to allocate device '%s' (%s). Falling back to CPU.", device_str, e)
        return torch.device("cpu")
