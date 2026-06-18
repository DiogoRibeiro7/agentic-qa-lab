"""Command line interface for the project."""

from __future__ import annotations

import typer
from rich.console import Console

from .config import RuntimeSettings

app = typer.Typer(help="Portfolio project command line interface.")
console = Console()


@app.command()
def info() -> None:
    """Print validated runtime settings."""
    settings = RuntimeSettings()
    console.print(settings.model_dump())


if __name__ == "__main__":
    app()
