"""
yctl init command - Initialize new AI projects.
"""

import typer
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn
from yctl.core.templates import get_template, TEMPLATES
from yctl.utils import (
    console,
    print_header,
    print_success,
    print_error,
    print_info,
    run_command,
    write_file,
)


def init_command(
    project_type: str = typer.Argument(..., help="Project type: nlp, cv, ml, research"),
    project_name: str = typer.Argument(..., help="Project name"),
    skip_venv: bool = typer.Option(False, "--skip-venv", help="Skip virtual environment creation"),
) -> None:
    """
    Initialize a new AI project with best practices.
    
    Creates project structure, virtual environment, and starter files.
    """
    print_header(f"Initializing {project_type.upper()} Project: {project_name}")
    
    # Validate project type
    if project_type not in TEMPLATES:
        print_error(f"Unknown project type: {project_type}")
        print_info(f"Available types: {', '.join(TEMPLATES.keys())}")
        raise typer.Exit(1)
    
    # Create project directory
    project_path = Path.cwd() / project_name
    
    if project_path.exists():
        print_error(f"Directory '{project_name}' already exists!")
        raise typer.Exit(1)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Create project structure
            task = progress.add_task("Creating project structure...", total=None)
            template = get_template(project_type, project_name, project_path)
            template.create_structure()
            progress.update(task, completed=True)
            
            # Create requirements.txt
            task = progress.add_task("Generating requirements.txt...", total=None)
            requirements = template.get_requirements()
            write_file(project_path / "requirements.txt", requirements)
            progress.update(task, completed=True)
            
            # Create README.md
            task = progress.add_task("Generating README.md...", total=None)
            readme = template.get_readme()
            write_file(project_path / "README.md", readme)
            progress.update(task, completed=True)
            
            # Create .gitignore
            task = progress.add_task("Creating .gitignore...", total=None)
            gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Data
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Models
outputs/models/*
!outputs/models/.gitkeep
*.pth
*.pt
*.h5
*.pkl

# Logs
outputs/logs/*
!outputs/logs/.gitkeep
*.log
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
            write_file(project_path / ".gitignore", gitignore)
            
            # Create .gitkeep files for empty directories
            for gitkeep_dir in ["data/raw", "data/processed", "outputs/models", "outputs/logs"]:
                gitkeep_path = project_path / gitkeep_dir / ".gitkeep"
                if gitkeep_path.parent.exists():
                    write_file(gitkeep_path, "")
            
            progress.update(task, completed=True)
            
            # Create virtual environment
            if not skip_venv:
                task = progress.add_task("Creating virtual environment...", total=None)
                exit_code, stdout, stderr = run_command(
                    f"python3 -m venv venv",
                    cwd=project_path
                )
                
                if exit_code != 0:
                    print_error(f"Failed to create virtual environment: {stderr}")
                    raise typer.Exit(1)
                
                progress.update(task, completed=True)
                
                # Upgrade pip
                task = progress.add_task("Upgrading pip...", total=None)
                exit_code, stdout, stderr = run_command(
                    f"venv/bin/pip install --upgrade pip",
                    cwd=project_path
                )
                progress.update(task, completed=True)
        
        # Success message
        console.print()
        print_success(f"Project '{project_name}' created successfully!")
        console.print()
        
        # Next steps
        console.print("[bold cyan]Next Steps:[/bold cyan]")
        console.print(f"  1. cd {project_name}")
        if not skip_venv:
            console.print("  2. source venv/bin/activate")
            console.print("  3. pip install -r requirements.txt")
        else:
            console.print("  2. pip install -r requirements.txt")
        console.print(f"  4. Start coding in src/")
        console.print()
        
        # Project-specific tips
        console.print("[bold cyan]Quick Tips:[/bold cyan]")
        if project_type == "nlp":
            console.print("  • Download spaCy model: python -m spacy download en_core_web_sm")
            console.print("  • Check out Hugging Face models: https://huggingface.co/models")
        elif project_type == "cv":
            console.print("  • Organize images in class-based folders under data/raw/")
            console.print("  • Use albumentations for data augmentation")
        elif project_type == "ml":
            console.print("  • Start with EDA in notebooks/")
            console.print("  • Use yctl inspect to analyze your dataset")
        elif project_type == "research":
            console.print("  • Use the experiment template in experiments/template.md")
            console.print("  • Track experiments with wandb or tensorboard")
        console.print()
        
    except Exception as e:
        print_error(f"Failed to create project: {str(e)}")
        raise typer.Exit(1)
