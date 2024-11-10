"""
Sentiment Analysis Package for Product Reviews.
This package provides tools for analyzing and classifying sentiment in product reviews.
"""

from .sentiment_analyzer import SentimentAnalyzer
from .sentiment_classifier import SentimentClassifier
from .utils import preprocess_text, create_visualizations

__version__ = '1.0.0'
__author__ = 'Dhyey'
__all__ = ['SentimentAnalyzer', 'SentimentClassifier', 'preprocess_text', 'create_visualizations']
