"""Export Colab requirements while retaining Colab's PyTorch/CUDA runtime."""
from __future__ import annotations

import subprocess
from pathlib import Path

if __name__ == "__main__":
    destination = Path("/tmp/slm-jspace-colab-requirements.txt")
    subprocess.run(
        [
            "uv",
            "export",
            "--locked",
            "--no-hashes",
            "--no-dev",
            "--group",
            "tutorial",
            "--no-emit-project",
            "--output-file",
            str(destination),
        ],
        check=True,
    )
    # Colab already supplies a PyTorch build matched to its NVIDIA driver.
    # The project lock targets general environments and therefore includes a
    # complete alternate CUDA stack; installing it in Colab is slow and can
    # replace the working runtime. Keep all tutorial dependencies but omit the
    # PyTorch distribution and its bundled CUDA wheels.
    excluded_prefixes = ("torch", "triton", "cuda-", "nvidia-")
    filtered = [
        line
        for line in destination.read_text(encoding="utf-8").splitlines()
        if not line.lower().startswith(excluded_prefixes)
    ]
    destination.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    print(destination)
