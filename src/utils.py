"""
Utility functions for sentiment analysis project.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_text(text):
    """
    Basic text preprocessing function.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ''
    return text.lower().strip()

def create_visualizations(df):
    """
    Create comprehensive visualizations of sentiment analysis results.
    
    Args:
        df (pandas.DataFrame): DataFrame containing sentiment analysis results
    """
    # Plot 1: Sentiment Distribution
    plt.figure(figsize=(12, 6))
    sentiment_counts = df['sentiment_category'].value_counts()
    plt.bar(sentiment_counts.index, sentiment_counts.values)
    plt.title('Distribution of Sentiments')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    # Add value labels on top of bars
    for i, v in enumerate(sentiment_counts.values):
        plt.text(i, v + 30, str(v), ha='center')

    plt.tight_layout()
    plt.show()

    # Plot 2: Sentiment Components
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # VADER components
    vader_means = df[['vader_pos', 'vader_neg', 'vader_neu']].mean()
    vader_means.plot(kind='bar', ax=ax1)
    ax1.set_title('Average VADER Sentiment Components')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Score')

    # TextBlob components
    sns.scatterplot(
        data=df,
        x='textblob_polarity',
        y='textblob_subjectivity',
        hue='sentiment_category',
        alpha=0.6,
        ax=ax2
    )
    ax2.set_title('Sentiment Polarity vs Subjectivity')

    plt.tight_layout()
    plt.show()

def save_results(df, results_dict, filename):
    """
    Save analysis results to CSV file.
    
    Args:
        df (pandas.DataFrame): Processed DataFrame
        results_dict (dict): Dictionary containing analysis results
        filename (str): Output filename
    """
    output_df = pd.DataFrame(results_dict)
    output_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def evaluate_class_metrics(y_true, y_pred, set_name=""):
    """
    Evaluate metrics for each class.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        set_name (str): Name of the dataset (e.g., "Training", "Test")
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    
    metrics = {}
    for label in ['0', '1', '2']:  # Negative, Neutral, Positive
        metrics[f'class_{label}'] = {
            'precision': report[label]['precision'],
            'recall': report[label]['recall'],
            'f1_score': report[label]['f1-score'],
            'support': report[label]['support']
        }
    
    return metrics
