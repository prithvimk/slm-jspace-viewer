"""Command line entry points."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from .runner import fit_lens, run_experiment, smoke_test

app = typer.Typer(no_args_is_help=True, help="Gemma Jacobian-lens experiment workflows.")


@app.command("smoke-test")
def smoke_test_command(model_config: Path) -> None:
    """Verify model forward pass, residual capture, and jlens adapter."""
    typer.echo(smoke_test(model_config))


@app.command("fit-lens")
def fit_lens_command(config: Path) -> None:
    """Fit and save an Anthropic-reference Jacobian lens."""
    typer.echo(fit_lens(config))


@app.command("run-experiment")
def run_experiment_command(config: Path) -> None:
    """Generate first, capture once, and export a complete artifact."""
    typer.echo(run_experiment(config))


@app.command("view")
def view_command(artifact: Path) -> None:
    """Launch the offline Streamlit viewer for one artifact."""
    completed = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("viewer.py")), "--", str(artifact)],
        check=False,
    )
    raise typer.Exit(completed.returncode)
