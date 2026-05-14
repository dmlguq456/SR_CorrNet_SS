"""SR_CorrNet SS -- Speech Separation inference package."""
try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required but not installed. "
        "Install with an accelerator extra:\n"
        "  uv sync --extra cu126    # CUDA 12.6\n"
        "  uv sync --extra cpu      # CPU-only\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu126  # pip alternative"
    ) from None

from sr_corrnet.inference import SSInference

__all__ = ["SSInference"]
