"""
Module for sentiment analysis using VADER and TextBlob.
"""

import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the SentimentAnalyzer with required NLTK components."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()

    def preprocess_text(self, text):
        """
        Comprehensive text preprocessing.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ''

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                 if token not in self.stop_words]

        return ' '.join(tokens)

    def get_detailed_sentiment(self, text):
        """
        Get detailed sentiment analysis using multiple methods.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Dictionary containing various sentiment scores
        """
        if not isinstance(text, str):
            return {}

        # VADER sentiment
        vader_scores = self.sia.polarity_scores(text)

        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment

        return {
            'vader_compound': vader_scores['compound'],
            'vader_pos': vader_scores['pos'],
            'vader_neg': vader_scores['neg'],
            'vader_neu': vader_scores['neu'],
            'textblob_polarity': textblob_sentiment.polarity,
            'textblob_subjectivity': textblob_sentiment.subjectivity
        }

    def extract_aspects(self, text):
        """
        Extract aspects and their associated sentiments.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Dictionary of aspects and their sentiment scores
        """
        if not isinstance(text, str):
            return {}

        aspects = {}
        sentences = sent_tokenize(text)

        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)

            for i, (word, tag) in enumerate(tagged):
                if tag.startswith('NN'):
                    start = max(0, i-3)
                    end = min(len(tagged), i+4)
                    context = ' '.join(token for token, _ in tagged[start:end])
                    sentiment = self.sia.polarity_scores(context)['compound']
                    aspects[word] = sentiment

        return aspects

    def analyze_full_review(self, review_title, review_description, variant=''):
        """
        Analyze a complete review with all components.
        
        Args:
            review_title (str): Title of the review
            review_description (str): Main review text
            variant (str, optional): Product variant information
            
        Returns:
            dict: Complete analysis results
        """
        full_text = ' '.join(filter(None, [review_title, review_description, variant]))
        
        return {
            'preprocessed_text': self.preprocess_text(full_text),
            'sentiment_scores': self.get_detailed_sentiment(full_text),
            'aspects': self.extract_aspects(full_text)
        }
