"""Tests for AI advisor module."""

from yctl.core.ai_advisor import AIAdvisor, analyze_ai_idea


def test_ai_advisor_sentiment_analysis():
    """Test AI advisor with sentiment analysis idea."""
    advisor = AIAdvisor()
    recommendation = advisor.analyze_idea("sentiment analysis for reviews")
    
    assert recommendation.idea == "sentiment analysis for reviews"
    assert recommendation.feasibility in ["high", "medium", "low"]
    assert recommendation.complexity in ["beginner", "intermediate", "advanced", "research"]
    assert len(recommendation.roadmap) > 0
    assert len(recommendation.datasets) > 0
    assert len(recommendation.model_architectures) > 0


def test_ai_advisor_object_detection():
    """Test AI advisor with object detection idea."""
    recommendation = analyze_ai_idea("object detection for cars")
    
    assert "object detection" in recommendation.idea.lower()
    assert len(recommendation.tools_libraries) > 0
    assert len(recommendation.challenges) > 0
    assert len(recommendation.resources) > 0


def test_ai_advisor_general_idea():
    """Test AI advisor with general ML idea."""
    recommendation = analyze_ai_idea("predict house prices")
    
    assert len(recommendation.roadmap) > 0
    assert len(recommendation.datasets) > 0
