"""
Common utility functions.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def run_command(cmd: str, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """
    Run a shell command and return exit code, stdout, stderr.
    
    Args:
        cmd: Command to run
        cwd: Working directory
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


def get_python_version() -> str:
    """Get the current Python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def check_command_exists(command: str) -> bool:
    """Check if a command exists in PATH."""
    return subprocess.run(
        f"command -v {command}",
        shell=True,
        capture_output=True
    ).returncode == 0


def create_directory(path: Path, exist_ok: bool = True) -> None:
    """Create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=exist_ok)


def write_file(path: Path, content: str) -> None:
    """Write content to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def get_venv_python(project_path: Path) -> Path:
    """Get the path to the Python executable in a venv."""
    return project_path / "venv" / "bin" / "python"


def get_venv_pip(project_path: Path) -> Path:
    """Get the path to pip in a venv."""
    return project_path / "venv" / "bin" / "pip"


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
