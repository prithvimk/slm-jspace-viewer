"""Export the locked runtime requirements for the active Google Colab kernel."""
from __future__ import annotations

import subprocess
from pathlib import Path

if __name__ == "__main__":
    destination = Path("/tmp/slm-jspace-colab-requirements.txt")
    subprocess.run(
        ["uv", "export", "--locked", "--no-hashes", "--no-dev", "--group", "tutorial", "--output-file", str(destination)],
        check=True,
    )
    print(destination)
