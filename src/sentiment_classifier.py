"""
Module for sentiment classification using machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

class SentimentClassifier:
    def __init__(self, df, text_column='processed_text', target_column='sentiment_category',
                 test_size=0.25, val_size=0.25, balance_method='smote'):
        """
        Initialize the classifier with dataset and parameters.
        
        Args:
            df (pandas.DataFrame): Input DataFrame containing reviews and sentiments
            text_column (str): Name of the column containing processed text
            target_column (str): Name of the column containing sentiment labels
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of training data to use for validation
            balance_method (str): Method to handle class imbalance ('smote' or 'class_weight')
        """
        self.df = df
        self.text_column = text_column
        self.target_column = target_column
        self.test_size = test_size
        self.val_size = val_size
        self.balance_method = balance_method
        self.models = {}
        self.results = {}

    def prepare_data(self):
        """
        Prepare data for ML models with explicit train/validation/test splits.
        
        Returns:
            tuple: Training, validation, and test sets
        """
        # Convert sentiment categories to numeric values
        sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        self.df['sentiment_numeric'] = self.df[self.target_column].map(sentiment_map)

        # First split: separate test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.df[self.text_column],
            self.df['sentiment_numeric'],
            test_size=self.test_size,
            random_state=42,
            stratify=self.df['sentiment_numeric']
        )

        # Second split: separate train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=self.val_size,
            random_state=42,
            stratify=y_train_val
        )

        # Store indices for later use
        self.val_indices = X_val.index
        self.test_indices = X_test.index

        # Compute class weights
        self.class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        self.class_weight_dict = dict(zip(np.unique(y_train), self.class_weights))

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_models(self):
        """
        Create pipeline for different models with class balancing.
        
        Returns:
            dict: Dictionary of model pipelines
        """
        if self.balance_method == 'smote':
            return {
                'Logistic Regression': ImbPipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000)),
                    ('smote', SMOTE(random_state=42)),
                    ('classifier', LogisticRegression(max_iter=1000))
                ]),
                'Random Forest': ImbPipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000)),
                    ('smote', SMOTE(random_state=42)),
                    ('classifier', RandomForestClassifier())
                ]),
                'XGBoost': ImbPipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000)),
                    ('smote', SMOTE(random_state=42)),
                    ('classifier', XGBClassifier(
                        objective='multi:softprob',
                        num_class=3
                    ))
                ]),
                'Naive Bayes': ImbPipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000)),
                    ('smote', SMOTE(random_state=42)),
                    ('classifier', MultinomialNB())
                ])
            }
        else:  # class_weight
            return {
                'Logistic Regression': Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000)),
                    ('classifier', LogisticRegression(
                        max_iter=1000,
                        class_weight=self.class_weight_dict
                    ))
                ]),
                # ... [similar pattern for other models]
            }

    def train_and_evaluate(self):
        """Train and evaluate all models."""
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        self.models = self.create_models()

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

            self.results[name] = {
                'val_predictions': val_pred,
                'test_predictions': test_pred,
                'val_report': classification_report(y_val, val_pred),
                'test_report': classification_report(y_test, test_pred),
                'val_confusion_matrix': confusion_matrix(y_val, val_pred),
                'test_confusion_matrix': confusion_matrix(y_test, test_pred),
                'cross_val_scores': cross_val_score(model, X_train, y_train, cv=5)
            }

    def predict_sentiment(self, text):
        """
        Predict sentiment for new text using the best model.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Prediction results including sentiment and probabilities
        """
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        # Get best model based on cross-validation scores
        model_names = list(self.results.keys())
        cv_means = [np.mean(self.results[name]['cross_val_scores']) for name in model_names]
        best_model = model_names[np.argmax(cv_means)]
        model = self.models[best_model]

        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]

        return {
            'sentiment': sentiment_map[prediction],
            'probabilities': {
                'Negative': probabilities[0],
                'Neutral': probabilities[1],
                'Positive': probabilities[2]
            },
            'model_used': best_model
        }
