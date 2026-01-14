"""Tests for core analyzers module."""

import tempfile
from pathlib import Path
import pandas as pd

from yctl.core.analyzers import DatasetAnalyzer, analyze_dataset


def test_dataset_analyzer_csv():
    """Test DatasetAnalyzer with CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("id,name,age,salary\n")
        f.write("1,Alice,25,50000\n")
        f.write("2,Bob,30,60000\n")
        f.write("3,Charlie,35,\n")  # Missing value
        f.flush()
        
        analyzer = DatasetAnalyzer(Path(f.name))
        stats = analyzer.analyze()
        
        Path(f.name).unlink()
        
        assert stats.num_rows == 3
        assert stats.num_cols == 4
        assert "salary" in stats.missing_values
        assert stats.missing_values["salary"] > 0


def test_analyze_dataset_function():
    """Test analyze_dataset convenience function."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("id,value\n")
        f.write("1,100\n")
        f.write("2,200\n")
        f.flush()
        
        stats, preprocessing, models = analyze_dataset(Path(f.name))
        
        Path(f.name).unlink()
        
        assert stats.num_rows == 2
        assert len(preprocessing) > 0
        assert len(models) > 0


def test_dataset_analyzer_detects_issues():
    """Test that analyzer detects data quality issues."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create dataset with issues
        f.write("id,value\n")
        f.write("1,100\n")
        f.write("1,100\n")  # Duplicate
        f.flush()
        
        analyzer = DatasetAnalyzer(Path(f.name))
        stats = analyzer.analyze()
        
        Path(f.name).unlink()
        
        assert stats.duplicates > 0
        assert len(stats.potential_issues) > 0
