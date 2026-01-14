"""Tests for CLI commands."""

import subprocess
import tempfile
from pathlib import Path


def test_cli_help():
    """Test that yctl --help works."""
    result = subprocess.run(
        ["yctl", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "yctl" in result.stdout.lower()
    assert "command" in result.stdout.lower()


def test_cli_doctor():
    """Test that yctl doctor runs without errors."""
    result = subprocess.run(
        ["yctl", "doctor"],
        capture_output=True,
        text=True
    )
    # Doctor may return warnings but should not crash
    assert result.returncode == 0
    assert "Python" in result.stdout


def test_cli_init_help():
    """Test that yctl init --help works."""
    result = subprocess.run(
        ["yctl", "init", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "project" in result.stdout.lower()


def test_cli_inspect_help():
    """Test that yctl inspect --help works."""
    result = subprocess.run(
        ["yctl", "inspect", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "dataset" in result.stdout.lower()


def test_cli_think_help():
    """Test that yctl think --help works."""
    result = subprocess.run(
        ["yctl", "think", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "idea" in result.stdout.lower()


def test_cli_init_creates_project():
    """Test that yctl init creates a project structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_name = "test-project"
        result = subprocess.run(
            ["yctl", "init", "nlp", project_name, "--skip-venv"],
            cwd=tmpdir,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        
        project_path = Path(tmpdir) / project_name
        assert project_path.exists()
        assert (project_path / "README.md").exists()
        assert (project_path / "requirements.txt").exists()
        assert (project_path / "src").exists()


def test_cli_inspect_csv():
    """Test that yctl inspect works with a CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("id,name,value\n")
        f.write("1,Alice,100\n")
        f.write("2,Bob,200\n")
        f.flush()
        
        result = subprocess.run(
            ["yctl", "inspect", f.name],
            capture_output=True,
            text=True
        )
        
        Path(f.name).unlink()
        
        assert result.returncode == 0
        assert "Dataset Overview" in result.stdout or "dataset" in result.stdout.lower()


def test_cli_think_analyzes_idea():
    """Test that yctl think analyzes an idea."""
    result = subprocess.run(
        ["yctl", "think", "sentiment analysis"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "sentiment" in result.stdout.lower() or "idea" in result.stdout.lower()
