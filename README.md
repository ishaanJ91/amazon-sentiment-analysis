# Amazon Reviews Sentiment Analysis

A comprehensive sentiment analysis project comparing three different approaches: NLTK's VADER, RoBERTa transformer model, and Hugging Face's default sentiment pipeline on Amazon product reviews.

## Overview

This project performs sentiment analysis on Amazon product reviews using multiple state-of-the-art approaches to compare their effectiveness and accuracy. The analysis includes exploratory data analysis, natural language processing fundamentals, and advanced transformer-based models.

## Features

- **Multi-Model Comparison**: Compares VADER, RoBERTa, and Transformer Pipeline approaches
- **Comprehensive EDA**: Visualizes review score distributions and sentiment patterns
- **NLTK Fundamentals**: Demonstrates tokenization, POS tagging, and named entity recognition
- **Advanced NLP**: Implements transformer models for context-aware sentiment analysis
- **Comparative Analysis**: Side-by-side comparison of different sentiment scoring methods

## Models Used

### 1. VADER (Valence Aware Dictionary and sEntiment Reasoner)

- Rule-based sentiment analysis tool
- Optimized for social media text
- Provides compound, positive, negative, and neutral scores

### 2. RoBERTa (Robustly Optimized BERT Pretraining Approach)

- Pre-trained transformer model: `cardiffnlp/twitter-roberta-base-sentiment`
- Context-aware sentiment classification
- Fine-tuned specifically for sentiment analysis

### 3. Transformer Pipeline

- Hugging Face's default sentiment analysis pipeline
- Easy-to-use interface with pre-trained models
- Returns binary sentiment classification with confidence scores

## Dataset

- **Source**: Amazon Reviews Dataset (`Reviews.csv`)
- **Sample Size**: 500 reviews (subset for analysis)
- **Features**: Review text, star ratings (1-5), and metadata

## Key Visualizations

- Review score distribution by star rating
- Sentiment scores comparison across different models
- Pairplot analysis showing correlations between different sentiment metrics
- Scatter plots comparing model performance

## Installation

```bash
# Create virtual environment
python -m venv sentiment-env
source sentiment-env/bin/activate  # On Windows: sentiment-env\Scripts\activate

# Install required packages
pip install pandas numpy seaborn matplotlib nltk transformers torch scipy tqdm

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Usage

1. **Data Loading**: Load and explore the Amazon reviews dataset
2. **NLTK Analysis**: Perform basic NLP tasks (tokenization, POS tagging, NER)
3. **VADER Scoring**: Apply rule-based sentiment analysis
4. **RoBERTa Analysis**: Use transformer model for advanced sentiment classification
5. **Pipeline Comparison**: Compare results across all three approaches
6. **Visualization**: Generate comparative plots and analysis

## Project Structure

```
sentiment-analysis/
├── AmazonReviews/
│   └── Reviews.csv
├── sentiment_analysis.py
└── README.md
```

## Key Insights

- **Model Performance**: Different models excel in different scenarios
- **Context Matters**: Transformer models better understand nuanced language
- **Score Correlation**: High correlation between advanced models, less with VADER
- **Review Complexity**: Some reviews show sentiment-rating mismatches

## Technical Highlights

- **Error Handling**: Robust exception handling for text processing
- **Batch Processing**: Efficient processing of large datasets with progress tracking
- **Data Integration**: Seamless merging of results from different models
- **Visualization**: Comprehensive plotting for model comparison

## Future Enhancements

- Extend to larger datasets
- Implement custom model fine-tuning
- Add more transformer model comparisons
- Include temporal sentiment analysis
- Deploy as a web application

## Dependencies

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `seaborn/matplotlib` - Data visualization
- `nltk` - Natural language processing
- `transformers` - Hugging Face transformer models
- `torch` - PyTorch framework
- `scipy` - Scientific computing
- `tqdm` - Progress bars

## Contributing

Feel free to contribute by:

- Adding new sentiment analysis models
- Improving visualization techniques
- Expanding the dataset
- Optimizing performance

## License

This project is open source and available under the MIT License.
