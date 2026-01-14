"""
yctl doctor command - Check system health for AI development.
"""

import typer
from rich.table import Table
from yctl.core.diagnostics import run_diagnostics
from yctl.utils import console, print_header, print_success, print_info


def doctor_command() -> None:
    """
    Check system health for AI development.
    
    Verifies Python, pip, venv, GPU, CUDA, and common tools.
    """
    print_header("System Health Check")
    
    with console.status("[bold cyan]Running diagnostics...", spinner="dots"):
        results = run_diagnostics()
    
    # Create results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=20)
    table.add_column("Status", width=10)
    table.add_column("Details", width=50)
    
    # Count statuses
    ok_count = 0
    warning_count = 0
    error_count = 0
    
    for result in results:
        # Determine status symbol and color
        if result.status == "ok":
            status = "[green]✓ OK[/green]"
            ok_count += 1
        elif result.status == "warning":
            status = "[yellow]⚠ WARNING[/yellow]"
            warning_count += 1
        else:
            status = "[red]✗ ERROR[/red]"
            error_count += 1
        
        # Add row to table
        details = result.message
        if result.details:
            details += f"\n{result.details}"
        
        table.add_row(result.name, status, details)
    
    console.print()
    console.print(table)
    console.print()
    
    # Display summary
    console.print("[bold cyan]Summary:[/bold cyan]")
    console.print(f"  ✓ OK: [green]{ok_count}[/green]")
    console.print(f"  ⚠ Warnings: [yellow]{warning_count}[/yellow]")
    console.print(f"  ✗ Errors: [red]{error_count}[/red]")
    console.print()
    
    # Display fix suggestions for warnings and errors
    issues_with_fixes = [r for r in results if r.status in ["warning", "error"] and r.fix_suggestion]
    
    if issues_with_fixes:
        console.print("[bold yellow]Suggested Fixes:[/bold yellow]")
        for result in issues_with_fixes:
            console.print(f"\n[bold]{result.name}:[/bold]")
            console.print(f"  {result.fix_suggestion}")
        console.print()
    
    # Overall status
    if error_count == 0 and warning_count == 0:
        print_success("All checks passed! Your system is ready for AI development.")
    elif error_count == 0:
        print_info("System is functional but has some warnings. Consider addressing them.")
    else:
        console.print("[bold red]⚠ Some critical issues detected. Please fix errors before proceeding.[/bold red]")
    
    console.print()
