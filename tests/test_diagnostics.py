"""Tests for diagnostics module."""

from yctl.core.diagnostics import SystemDiagnostics, run_diagnostics


def test_system_diagnostics_runs():
    """Test that system diagnostics runs without errors."""
    diagnostics = SystemDiagnostics()
    results = diagnostics.run_all_checks()
    
    assert len(results) > 0
    assert all(hasattr(r, 'name') for r in results)
    assert all(hasattr(r, 'status') for r in results)
    assert all(r.status in ['ok', 'warning', 'error'] for r in results)


def test_run_diagnostics_function():
    """Test run_diagnostics convenience function."""
    results = run_diagnostics()
    
    assert len(results) > 0
    # Should at least check Python
    assert any('python' in r.name.lower() for r in results)


def test_diagnostics_checks_python():
    """Test that diagnostics checks Python version."""
    diagnostics = SystemDiagnostics()
    diagnostics.check_python_version()
    
    assert len(diagnostics.results) > 0
    python_check = diagnostics.results[0]
    assert 'python' in python_check.name.lower()
