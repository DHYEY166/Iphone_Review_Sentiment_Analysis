import pytest
import pandas as pd
from src.sentiment_analyzer import SentimentAnalyzer

@pytest.fixture
def analyzer():
    """Create a SentimentAnalyzer instance for testing"""
    return SentimentAnalyzer()

@pytest.fixture
def sample_texts():
    """Provide sample texts for testing"""
    return [
        "This product is amazing! Great battery life and excellent camera.",
        "Terrible product, worst purchase ever. Complete waste of money.",
        "It's okay, nothing special but gets the job done.",
        "",  # Empty text
        None,  # None value
        123,  # Non-string input
        "!@#$%^&*()",  # Special characters only
    ]

def test_preprocess_text(analyzer, sample_texts):
    """Test text preprocessing functionality"""
    # Test normal text
    processed = analyzer.preprocess_text(sample_texts[0])
    assert isinstance(processed, str)
    assert len(processed) > 0
    assert processed.islower()
    assert not any(char in processed for char in '!.,')

    # Test empty string
    assert analyzer.preprocess_text("") == ""
    
    # Test None value
    assert analyzer.preprocess_text(None) == ""
    
    # Test non-string input
    assert analyzer.preprocess_text(123) == ""

def test_get_detailed_sentiment(analyzer, sample_texts):
    """Test sentiment analysis functionality"""
    # Test positive sentiment
    positive_result = analyzer.get_detailed_sentiment(sample_texts[0])
    assert isinstance(positive_result, dict)
    assert 'vader_compound' in positive_result
    assert positive_result['vader_compound'] > 0

    # Test negative sentiment
    negative_result = analyzer.get_detailed_sentiment(sample_texts[1])
    assert negative_result['vader_compound'] < 0

    # Test neutral sentiment
    neutral_result = analyzer.get_detailed_sentiment(sample_texts[2])
    assert abs(neutral_result['vader_compound']) < 0.5

    # Test empty input
    empty_result = analyzer.get_detailed_sentiment("")
    assert isinstance(empty_result, dict)
    assert abs(empty_result['vader_compound']) < 0.1

    # Test None input
    none_result = analyzer.get_detailed_sentiment(None)
    assert isinstance(none_result, dict)

def test_extract_aspects(analyzer, sample_texts):
    """Test aspect extraction functionality"""
    # Test normal text
    aspects = analyzer.extract_aspects(sample_texts[0])
    assert isinstance(aspects, dict)
    assert 'battery' in aspects or 'camera' in aspects

    # Test text without clear aspects
    no_aspects = analyzer.extract_aspects("Good!")
    assert isinstance(no_aspects, dict)
    assert len(no_aspects) == 0

    # Test empty input
    empty_aspects = analyzer.extract_aspects("")
    assert isinstance(empty_aspects, dict)
    assert len(empty_aspects) == 0

    # Test None input
    none_aspects = analyzer.extract_aspects(None)
    assert isinstance(none_aspects, dict)
    assert len(none_aspects) == 0

def test_edge_cases(analyzer):
    """Test edge cases and error handling"""
    # Test very long text
    long_text = "great " * 1000
    result = analyzer.get_detailed_sentiment(long_text)
    assert isinstance(result, dict)
    assert 'vader_compound' in result

    # Test special characters
    special_chars = "!@#$%^&*()"
    result = analyzer.preprocess_text(special_chars)
    assert isinstance(result, str)

    # Test mixed input types
    mixed_text = f"Product review: {123} stars!"
    result = analyzer.get_detailed_sentiment(mixed_text)
    assert isinstance(result, dict)
