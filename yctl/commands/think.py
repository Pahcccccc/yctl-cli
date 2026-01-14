"""
yctl think command - Analyze AI project ideas.
"""

import typer
from rich.panel import Panel
from rich.markdown import Markdown
from yctl.core.ai_advisor import analyze_ai_idea
from yctl.utils import console, print_header, create_table


def think_command(
    idea: str = typer.Argument(..., help="Your AI project idea"),
) -> None:
    """
    Analyze an AI project idea and get recommendations.
    
    Provides roadmap, datasets, models, tools, and resources.
    """
    print_header("AI Idea Analysis")
    
    with console.status("[bold cyan]Analyzing your idea...", spinner="dots"):
        recommendation = analyze_ai_idea(idea)
    
    # Display idea
    console.print(Panel(
        f"[bold]{recommendation.idea}[/bold]",
        title="💡 Your Idea",
        border_style="cyan"
    ))
    console.print()
    
    # Display assessment
    console.print("[bold cyan]📊 Assessment[/bold cyan]")
    assessment_table = create_table("", ["Metric", "Value"])
    
    # Color code feasibility
    feasibility_color = {
        'high': 'green',
        'medium': 'yellow',
        'low': 'red'
    }
    feasibility_str = f"[{feasibility_color[recommendation.feasibility]}]{recommendation.feasibility.upper()}[/{feasibility_color[recommendation.feasibility]}]"
    
    # Color code complexity
    complexity_color = {
        'beginner': 'green',
        'intermediate': 'yellow',
        'advanced': 'orange1',
        'research': 'red'
    }
    complexity_str = f"[{complexity_color.get(recommendation.complexity, 'white')}]{recommendation.complexity.upper()}[/{complexity_color.get(recommendation.complexity, 'white')}]"
    
    assessment_table.add_row("Feasibility", feasibility_str)
    assessment_table.add_row("Complexity", complexity_str)
    
    console.print(assessment_table)
    console.print()
    
    # Display roadmap
    console.print("[bold cyan]🗺️  Recommended Roadmap[/bold cyan]")
    for step in recommendation.roadmap:
        console.print(step)
    console.print()
    
    # Display datasets
    console.print("[bold cyan]📚 Suggested Datasets[/bold cyan]")
    dataset_table = create_table("", ["Dataset", "Source"])
    for dataset in recommendation.datasets:
        dataset_table.add_row(dataset['name'], dataset['source'])
    console.print(dataset_table)
    console.print()
    
    # Display model architectures
    console.print("[bold cyan]🏗️  Model Architectures[/bold cyan]")
    for i, model in enumerate(recommendation.model_architectures, 1):
        console.print(f"  {i}. {model}")
    console.print()
    
    # Display tools and libraries
    console.print("[bold cyan]🛠️  Tools & Libraries[/bold cyan]")
    # Group into rows of 4
    tools = recommendation.tools_libraries
    for i in range(0, len(tools), 4):
        row = tools[i:i+4]
        console.print("  " + "  •  ".join(row))
    console.print()
    
    # Display challenges
    console.print("[bold yellow]⚠️  Potential Challenges[/bold yellow]")
    for i, challenge in enumerate(recommendation.challenges, 1):
        console.print(f"  {i}. {challenge}")
    console.print()
    
    # Display resources
    console.print("[bold cyan]📖 Learning Resources[/bold cyan]")
    for i, resource in enumerate(recommendation.resources, 1):
        console.print(f"  {i}. {resource}")
    console.print()
    
    # Next steps
    console.print(Panel(
        "[bold]Next Steps:[/bold]\n"
        "1. Research the suggested datasets and models\n"
        "2. Create a project: [cyan]yctl init <type> <name>[/cyan]\n"
        "3. Start with a simple baseline\n"
        "4. Iterate and improve",
        title="🚀 Getting Started",
        border_style="green"
    ))
    console.print()
