# Sentiment Analysis for Apple Product Reviews

A comprehensive sentiment analysis system designed to analyze product reviews using multiple machine learning approaches and natural language processing techniques.

## Features

- Multi-model sentiment classification (Logistic Regression, Random Forest, XGBoost, Naive Bayes)
- Detailed sentiment analysis using VADER and TextBlob
- Advanced text preprocessing and feature extraction
- Handling of class imbalance using SMOTE
- Comprehensive evaluation metrics and visualizations
- Production-ready prediction interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

The main analysis and implementation can be found in `notebooks/UPDATED.ipynb`, which contains:
- Data preprocessing and exploration
- Sentiment analysis implementation
- Model training and evaluation
- Visualization of results
- Example predictions

## Usage

### Basic Usage

```python
from src.sentiment_analyzer import SentimentAnalyzer
from src.sentiment_classifier import SentimentClassifier

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Analyze a single review
text = "Great product, excellent battery life!"
sentiment = analyzer.get_detailed_sentiment(text)
print(sentiment)

# Train classifier on your dataset
classifier = SentimentClassifier(df, balance_method='smote')
classifier.train_and_evaluate()

# Make predictions
result = classifier.predict_sentiment("This is an amazing product!")
print(result)
```

### Advanced Usage

The complete implementation and analysis workflow can be found in `notebooks/UPDATED.ipynb`. This notebook includes:
- Detailed data analysis
- Feature engineering steps
- Model training and tuning
- Performance evaluation
- Visualization of results
- Example applications

## Model Performance

The system includes multiple classification models:
- Logistic Regression
- Random Forest
- XGBoost
- Naive Bayes

Each model is evaluated using:
- Cross-validation scores
- Precision, Recall, and F1-score per class
- Confusion matrices
- ROC curves

## Documentation

For detailed documentation of each module and class, please refer to the docstrings in the source code or generate documentation using:

```bash
pdoc --html src
```

## Testing

Run tests using pytest:

```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sentiment_analysis_2024,
  author = {Your Name},
  title = {Sentiment Analysis for Product Reviews},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/sentiment-analysis}
}
```

## Notebook Details

The main analysis notebook (`UPDATED.ipynb`) contains a comprehensive implementation of the sentiment analysis system, including:

### Data Processing
- Loading and cleaning iPhone review data
- Text preprocessing and feature extraction
- Sentiment analysis using multiple approaches

### Model Implementation
- Implementation of multiple classification models
- Class imbalance handling using SMOTE
- Cross-validation and performance evaluation

### Visualization
- Sentiment distribution plots
- Model performance comparisons
- Confusion matrices
- Feature importance analysis

### Results
- Detailed performance metrics for each model
- Example predictions on test cases
- Analysis of model strengths and weaknesses
