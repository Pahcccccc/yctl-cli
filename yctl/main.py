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
    project_type: str = typer.Argument(
        ..., 
        help="Project type: nlp (NLP), cv (Computer Vision), ml (Machine Learning), or research"
    ),
    project_name: str = typer.Argument(..., help="Name for your new project"),
    skip_venv: bool = typer.Option(
        False, 
        "--skip-venv", 
        help="Skip virtual environment creation (not recommended)"
    ),
) -> None:
    """
    🚀 Initialize a new AI/ML project with best practices.
    
    Creates a complete project structure with virtual environment,
    curated dependencies, starter code, and comprehensive documentation.
    
    Examples:
    
        $ yctl init nlp sentiment-analyzer
        
        $ yctl init cv image-classifier
        
        $ yctl init ml house-price-prediction
        
        $ yctl init research novel-architecture
    """
    init_command(project_type, project_name, skip_venv)


@app.command("inspect")
def inspect(
    dataset_path: str = typer.Argument(..., help="Path to your dataset file"),
    show_sample: bool = typer.Option(False, "--sample", help="Display sample rows from the dataset"),
) -> None:
    """
    🔍 Inspect and analyze a dataset comprehensively.
    
    Provides detailed statistics, detects data quality issues,
    suggests preprocessing steps, and recommends suitable ML models.
    
    Supported formats: CSV, Excel (.xlsx, .xls), JSON, Parquet
    
    Examples:
    
        $ yctl inspect data/train.csv
        
        $ yctl inspect data/dataset.parquet --sample
    """
    inspect_command(dataset_path, show_sample)


@app.command("doctor")
def doctor() -> None:
    """
    🏥 Check system health for AI/ML development.
    
    Verifies your development environment is properly configured:
    
    • Python version (3.10+ recommended)
    
    • pip and venv availability
    
    • GPU detection (NVIDIA)
    
    • CUDA status and version
    
    • Common development tools
    
    Provides actionable fix suggestions for any issues found.
    
    Example:
    
        $ yctl doctor
    """
    doctor_command()


@app.command("think")
def think(
    idea: str = typer.Argument(..., help="Your AI/ML project idea or concept"),
) -> None:
    """
    💡 Analyze an AI project idea and get expert recommendations.
    
    Provides comprehensive guidance including:
    
    • Feasibility assessment
    
    • Complexity rating
    
    • Step-by-step roadmap
    
    • Suggested datasets and sources
    
    • Recommended model architectures
    
    • Required tools and libraries
    
    • Potential challenges
    
    • Learning resources
    
    Examples:
    
        $ yctl think "sentiment analysis for customer reviews"
        
        $ yctl think "object detection for autonomous driving"
        
        $ yctl think "time series forecasting for stock prices"
    """
    think_command(idea)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
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
    
    # If no subcommand is provided, show help
    if ctx.invoked_subcommand is None and not version:
        console.print(ctx.get_help())
        raise typer.Exit()


if __name__ == "__main__":
    app()
