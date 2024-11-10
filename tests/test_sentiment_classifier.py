import pytest
import pandas as pd
import numpy as np
from src.sentiment_classifier import SentimentClassifier

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'reviewTitle': ['Great product', 'Terrible purchase', 'Okay device'] * 10,
        'reviewDescription': [
            'Amazing battery life and great camera',
            'Worst product ever, complete waste of money',
            'Average performance, decent features'
        ] * 10,
        'variant': ['Model A', 'Model B', 'Model C'] * 10,
        'sentiment_category': ['Positive', 'Negative', 'Neutral'] * 10
    })

@pytest.fixture
def classifier(sample_data):
    """Create a SentimentClassifier instance for testing"""
    return SentimentClassifier(
        sample_data,
        text_column='reviewDescription',
        target_column='sentiment_category',
        test_size=0.2,
        val_size=0.2,
        balance_method='smote'
    )

def test_initialization(classifier):
    """Test classifier initialization"""
    assert hasattr(classifier, 'df')
    assert hasattr(classifier, 'text_column')
    assert hasattr(classifier, 'target_column')
    assert 0 < classifier.test_size < 1
    assert 0 < classifier.val_size < 1

def test_prepare_data(classifier):
    """Test data preparation"""
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data()
    
    # Check that splits are created correctly
    assert len(X_train) + len(X_val) + len(X_test) == len(classifier.df)
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)
    
    # Check class balance
    assert len(set(y_train)) == 3  # Three sentiment classes
    assert len(set(y_val)) == 3
    assert len(set(y_test)) == 3

def test_create_models(classifier):
    """Test model creation"""
    classifier.prepare_data()  # Need to prepare data first to get class weights
    models = classifier.create_models()
    
    # Check that all expected models are created
    assert 'Logistic Regression' in models
    assert 'Random Forest' in models
    assert 'XGBoost' in models
    assert 'Naive Bayes' in models
    
    # Check model pipeline structure
    for model in models.values():
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

def test_train_and_evaluate(classifier):
    """Test model training and evaluation"""
    classifier.train_and_evaluate()
    
    # Check that results are stored
    assert hasattr(classifier, 'results')
    assert len(classifier.results) > 0
    
    # Check results structure for each model
    for model_results in classifier.results.values():
        assert 'val_predictions' in model_results
        assert 'test_predictions' in model_results
        assert 'val_report' in model_results
        assert 'test_report' in model_results
        assert 'cross_val_scores' in model_results

def test_predict_sentiment(classifier):
    """Test sentiment prediction"""
    # Train the models first
    classifier.train_and_evaluate()
    
    # Test prediction on new text
    test_texts = [
        "This is an amazing product with great features",
        "Terrible product, complete waste of money",
        "It's an okay product, nothing special"
    ]
    
    for text in test_texts:
        result = classifier.predict_sentiment(text)
        assert isinstance(result, dict)
        assert 'sentiment' in result
        assert 'probabilities' in result
        assert 'model_used' in result
        assert result['sentiment'] in ['Negative', 'Neutral', 'Positive']
        assert sum(result['probabilities'].values()) > 0.99  # Should sum to ~1

def test_edge_cases(classifier):
    """Test edge cases and error handling"""
    # Test empty text
    result = classifier.predict_sentiment("")
    assert isinstance(result, dict)
    assert 'sentiment' in result
    
    # Test very long text
    long_text = "great product " * 1000
    result = classifier.predict_sentiment(long_text)
    assert isinstance(result, dict)
    assert 'sentiment' in result
    
    # Test text with special characters
    special_text = "Product review: !@#$%^&*()"
    result = classifier.predict_sentiment(special_text)
    assert isinstance(result, dict)
    assert 'sentiment' in result

def test_performance_metrics(classifier):
    """Test performance metric calculations"""
    classifier.train_and_evaluate()
    
    # Check cross-validation scores
    for model_name, results in classifier.results.items():
        cv_scores = results['cross_val_scores']
        assert len(cv_scores) > 0
        assert all(0 <= score <= 1 for score in cv_scores)
        
        # Check confusion matrix
        conf_matrix = results['test_confusion_matrix']
        assert isinstance(conf_matrix, np.ndarray)
        assert conf_matrix.shape == (3, 3)  # 3x3 for three classes
