"""
Main entry point for yctl CLI.
"""

import typer
from typing import Optional
from yctl.commands import init_command, inspect_command, doctor_command, think_command
from yctl.utils import console

app = typer.Typer(
    name="yctl",
    help="Personal AI Engineer CLI Tool - Streamline your AI development workflow",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command("init")
def init(
    project_type: str = typer.Argument(..., help="Project type: nlp | cv | ml | research"),
    project_name: str = typer.Argument(..., help="Project name"),
    skip_venv: bool = typer.Option(False, "--skip-venv", help="Skip virtual environment creation"),
) -> None:
    """
    Initialize a new AI project.
    
    Creates a complete project structure with best practices, virtual environment,
    and starter code for your chosen project type.
    
    Examples:
        yctl init nlp sentiment-analyzer
        yctl init cv image-classifier
        yctl init ml house-price-prediction
        yctl init research novel-architecture
    """
    init_command(project_type, project_name, skip_venv)


@app.command("inspect")
def inspect(
    dataset_path: str = typer.Argument(..., help="Path to dataset file"),
    show_sample: bool = typer.Option(False, "--sample", help="Show sample rows"),
) -> None:
    """
    Inspect and analyze a dataset.
    
    Provides comprehensive statistics, detects data quality issues,
    suggests preprocessing steps, and recommends suitable models.
    
    Supported formats: CSV, Excel, JSON, Parquet
    
    Examples:
        yctl inspect data/train.csv
        yctl inspect data/dataset.parquet --sample
    """
    inspect_command(dataset_path, show_sample)


@app.command("doctor")
def doctor() -> None:
    """
    Check system health for AI development.
    
    Verifies:
        - Python version
        - pip and venv
        - GPU availability
        - CUDA status
        - Common development tools
    
    Provides fix suggestions for any issues found.
    
    Example:
        yctl doctor
    """
    doctor_command()


@app.command("think")
def think(
    idea: str = typer.Argument(..., help="Your AI project idea"),
) -> None:
    """
    Analyze an AI project idea and get recommendations.
    
    Provides:
        - Feasibility assessment
        - Complexity rating
        - Step-by-step roadmap
        - Suggested datasets
        - Model architectures
        - Tools and libraries
        - Potential challenges
        - Learning resources
    
    Examples:
        yctl think "sentiment analysis for customer reviews"
        yctl think "object detection for autonomous driving"
        yctl think "time series forecasting for stock prices"
    """
    think_command(idea)


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        is_eager=True,
    ),
) -> None:
    """
    yctl - Personal AI Engineer CLI Tool
    
    A powerful command-line tool for AI engineers working on Ubuntu.
    Streamlines project initialization, dataset inspection, system diagnostics,
    and AI ideation.
    """
    if version:
        from yctl import __version__
        console.print(f"yctl version {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
