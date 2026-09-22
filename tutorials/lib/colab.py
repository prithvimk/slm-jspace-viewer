"""Small, explicit Google Colab runtime checks for the tutorial notebooks."""
from __future__ import annotations

from typing import Any


def report_runtime(*, require_gpu: bool = False) -> dict[str, Any]:
    """Print the active PyTorch runtime and return its device capabilities."""
    import torch

    cuda_available = torch.cuda.is_available()
    summary: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "device": "cpu",
        "capability": None,
        "recommended_dtype": "float32",
    }
    if cuda_available:
        capability = torch.cuda.get_device_capability()
        summary.update(
            device=torch.cuda.get_device_name(),
            capability=capability,
            # Turing (for example, Colab's T4) has FP16 Tensor Cores. BF16
            # Tensor Core support begins with Ampere (compute capability 8).
            recommended_dtype="bfloat16" if capability[0] >= 8 else "float16",
        )

    print(
        f"PyTorch {summary['torch']} | device: {summary['device']} | "
        f"recommended model dtype: {summary['recommended_dtype']}"
    )
    if require_gpu and not cuda_available:
        raise RuntimeError(
            "This notebook's micro-fit needs a GPU runtime. In Colab, choose "
            "Runtime > Change runtime type > T4 GPU, then rerun this setup cell."
        )
    if not cuda_available:
        print("CPU profile selected: notebooks 1–3 work, but generation will be slower.")
    return summary
