"""Write the small dependency set needed by the public Colab notebooks."""
from __future__ import annotations

from pathlib import Path

if __name__ == "__main__":
    destination = Path("/tmp/slm-jspace-colab-requirements.txt")
    # Do not export uv.lock here. It pins every transitive package, including
    # Colab-owned PyTorch, NumPy, Jupyter, and Google integration packages.
    # Replacing those in a live kernel causes ABI and resolver conflicts.
    requirements = """\
transformers>=5.5
huggingface-hub>=0.30
safetensors>=0.5
plotly>=6.0
jlens @ git+https://github.com/anthropics/jacobian-lens.git@581d398613e5602a5af361e1c34d3a92ea82ba8e
"""
    destination.write_text(requirements, encoding="utf-8")
    print(destination)
