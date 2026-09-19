"""
Text Preprocessing Module for E-Consultation Sentiment Analysis.

This module provides NLTK-based text cleaning, tokenization, stopword filtering
(with negation preservation), and lemmatization. It also retains a VADER baseline
for comparative analysis.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure required NLTK resources are available
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Initialize lemmatizer and VADER analyzer
lemmatizer = WordNetLemmatizer()
vader_analyzer = SentimentIntensityAnalyzer()

# Negation words to preserve (crucial for sentiment analysis)
NEGATION_WORDS = {
    'not', 'no', 'nor', 'neither', 'never', 'hardly', 'scarcely', 'barely',
    'cannot', "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't",
    "haven't", "isn't", "wasn't", "weren't", "won't", "wouldn't", "without"
}

# English stopwords excluding negation words
try:
    NLTK_STOPWORDS = set(stopwords.words('english')) - NEGATION_WORDS
except Exception:
    NLTK_STOPWORDS = set()


def clean_text(text: str) -> str:
    """
    Normalizes raw text:
    1. Converts to lowercase.
    2. Removes URLs and HTML markup.
    3. Retains letters, basic punctuation used in negation, and whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Replace contractions with standard forms
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    # Keep only alphabetical characters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_comment(text: str) -> str:
    """
    Full preprocessing pipeline:
    Cleaning -> Tokenization -> Negation-aware Stopword Removal -> Lemmatization.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        tokens = cleaned.split()
    
    # Filter stopwords and lemmatize
    processed_tokens = []
    for token in tokens:
        if token not in NLTK_STOPWORDS and len(token) > 1:
            # Lemmatize as verb then noun
            lemma = lemmatizer.lemmatize(token, pos='v')
            lemma = lemmatizer.lemmatize(lemma, pos='n')
            processed_tokens.append(lemma)
            
    return " ".join(processed_tokens)


def get_vader_sentiment(text: str) -> dict:
    """
    Computes VADER sentiment as a comparative baseline.
    Returns compound polarity score and sentiment label.
    """
    if not isinstance(text, str) or not text.strip():
        return {'sentiment': 'neutral', 'compound': 0.0}
        
    scores = vader_analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound > 0.05:
        sentiment = 'positive'
    elif compound < -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
        
    return {
        'sentiment': sentiment,
        'compound': round(compound, 4),
        'positive': round(scores['pos'], 4),
        'negative': round(scores['neg'], 4),
        'neutral': round(scores['neu'], 4)
    }
