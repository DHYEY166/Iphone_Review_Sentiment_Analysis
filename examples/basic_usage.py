import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Add the src directory to the Python path
src_path = str(Path(__file__).resolve().parents[1] / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from sentiment_analyzer import SentimentAnalyzer
from sentiment_classifier import SentimentClassifier

def basic_sentiment_analysis():
    """
    Demonstrate basic sentiment analysis functionality
    """
    print("\n=== Basic Sentiment Analysis ===")
    
    # Initialize the analyzer
    analyzer = SentimentAnalyzer()
    
    # Example reviews
    reviews = [
        "This iPhone is absolutely amazing! Great camera and battery life.",
        "The product is okay, but a bit expensive for what it offers.",
        "Terrible experience. The phone keeps freezing and battery drains quickly.",
    ]
    
    # Analyze each review
    for review in reviews:
        print("\nAnalyzing review:", review)
        
        # Get sentiment scores
        sentiment = analyzer.get_detailed_sentiment(review)
        
        print("\nSentiment Scores:")
        print(f"VADER Compound Score: {sentiment['vader_compound']:.3f}")
        print(f"VADER Positive: {sentiment['vader_pos']:.3f}")
        print(f"VADER Negative: {sentiment['vader_neg']:.3f}")
        print(f"VADER Neutral: {sentiment['vader_neu']:.3f}")
        print(f"TextBlob Polarity: {sentiment['textblob_polarity']:.3f}")
        print(f"TextBlob Subjectivity: {sentiment['textblob_subjectivity']:.3f}")
        
        # Extract aspects
        aspects = analyzer.extract_aspects(review)
        if aspects:
            print("\nAspect-Based Sentiment:")
            for aspect, score in aspects.items():
                print(f"{aspect}: {score:.3f}")

def train_and_evaluate_classifier(data_path="../data/iphone.csv"):
    """
    Demonstrate model training and evaluation
    """
    print("\n=== Training and Evaluating Classifier ===")
    
    try:
        # Load data
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} reviews from dataset")
        
        # Initialize classifier
        classifier = SentimentClassifier(
            df,
            text_column='reviewDescription',
            test_size=0.2,
            val_size=0.2,
            balance_method='smote'
        )
        
        # Train and evaluate
        print("\nTraining models...")
        classifier.train_and_evaluate()
        
        # Plot results
        print("\nGenerating performance plots...")
        classifier.plot_results()
        
        return classifier
        
    except FileNotFoundError:
        print(f"Error: Could not find data file at {data_path}")
        return None

def make_predictions(classifier):
    """
    Demonstrate making predictions with trained classifier
    """
    print("\n=== Making Predictions ===")
    
    if classifier is None:
        print("Error: No trained classifier available")
        return
    
    # Example reviews for prediction
    test_reviews = [
        "Best phone I've ever owned! The camera is incredible and battery lasts forever.",
        "It's decent but overpriced. Camera could be better.",
        "Complete waste of money. Keeps crashing and terrible customer service.",
        "The design is beautiful but the performance is just average.",
        "Good value for money, meets all my needs."
    ]
    
    print("\nMaking predictions on new reviews:")
    for review in test_reviews:
        result = classifier.predict_sentiment(review)
        
        print(f"\nReview: {review}")
        print(f"Predicted Sentiment: {result['sentiment']}")
        print("Confidence Scores:")
        for sentiment, prob in result['probabilities'].items():
            print(f"  {sentiment}: {prob:.3f}")

def batch_processing_example(classifier):
    """
    Demonstrate batch processing of reviews
    """
    print("\n=== Batch Processing Example ===")
    
    if classifier is None:
        print("Error: No trained classifier available")
        return
    
    # Create a sample DataFrame with reviews
    sample_data = pd.DataFrame({
        'review_text': [
            "Great product, highly recommended!",
            "Not worth the money",
            "Average performance, nothing special",
            "Excellent features and quality",
            "Disappointed with the purchase"
        ]
    })
    
    # Process batch
    print(f"\nProcessing batch of {len(sample_data)} reviews...")
    
    results = []
    for text in sample_data['review_text']:
        result = classifier.predict_sentiment(text)
        results.append({
            'text': text,
            'sentiment': result['sentiment'],
            'confidence': max(result['probabilities'].values())
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print("\nBatch Processing Results:")
    print(results_df)

def main():
    """
    Main execution function demonstrating all features
    """
    print("=== Sentiment Analysis System Demo ===")
    
    # Basic sentiment analysis
    basic_sentiment_analysis()
    
    # Train classifier
    classifier = train_and_evaluate_classifier()
    
    # Make predictions
    make_predictions(classifier)
    
    # Batch processing
    batch_processing_example(classifier)
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
