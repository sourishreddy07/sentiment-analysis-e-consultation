"""
Supervised Machine Learning Training & Evaluation Pipeline
for E-Consultation Feedback Sentiment Analysis.

Trains and evaluates candidate classifiers (Logistic Regression, Naive Bayes, Linear SVM)
using TF-IDF feature extraction on econsult_comments_dataset.csv, selects the best model
based on measured test F1-score, and serializes artifacts to models/.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocess import preprocess_comment

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'econsult_comments_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_preprocess_data():
    """Loads dataset and applies NLTK text preprocessing."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    print(f"[*] Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    # Clean text and labels
    df = df.dropna(subset=['comment', 'sentiment_label'])
    df['sentiment_label'] = df['sentiment_label'].astype(str).str.lower().str.strip()
    
    print(f"[*] Preprocessing {len(df)} comments with NLTK...")
    df['cleaned_comment'] = df['comment'].apply(preprocess_comment)
    
    # Class distribution
    counts = df['sentiment_label'].value_counts().to_dict()
    print(f"[*] Class Distribution: {counts}")
    return df


def train_and_evaluate():
    """Trains candidate models, measures test performance, and serializes the best model."""
    df = load_and_preprocess_data()
    
    X = df['cleaned_comment']
    y = df['sentiment_label']
    
    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # TF-IDF Vectorization
    print("[*] Fitting TF-IDF Vectorizer (unigrams + bigrams)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    # Candidate classifiers
    # LinearSVC wrapped in CalibratedClassifierCV to provide probability calibration
    candidates = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Linear SVM': CalibratedClassifierCV(LinearSVC(dual='auto', random_state=42))
    }
    
    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None
    
    print("\n" + "=" * 65)
    print("      EMPIRICAL MODEL EVALUATION & BENCHMARKING RESULTS")
    print("=" * 65)
    
    for name, model in candidates.items():
        # Train
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        
        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=['negative', 'positive']).tolist()
        
        results[name] = {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'confusion_matrix': cm
        }
        
        print(f" Model: {name}")
        print(f"   Accuracy  : {acc * 100:.2f}%")
        print(f"   Precision : {prec * 100:.2f}%")
        print(f"   Recall    : {rec * 100:.2f}%")
        print(f"   F1-Score  : {f1 * 100:.2f}%")
        print(f"   Confusion Matrix [TN, FP / FN, TP]: {cm}")
        print("-" * 65)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model
            
    print(f"\n[+] BEST MODEL SELECTED: '{best_model_name}' with F1-Score: {best_f1 * 100:.2f}%")
    
    # Save artifacts
    model_path = os.path.join(MODELS_DIR, 'sentiment_model.joblib')
    vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    
    joblib.dump(best_model_obj, model_path)
    joblib.dump(tfidf, vectorizer_path)
    
    metrics_payload = {
        'best_model': best_model_name,
        'best_f1': round(best_f1 * 100, 2),
        'dataset_total': len(df),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'num_classes': len(df['sentiment_label'].unique()),
        'classes': sorted(df['sentiment_label'].unique().tolist()),
        'class_distribution': df['sentiment_label'].value_counts().to_dict(),
        'models_comparison': results
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_payload, f, indent=4)
        
    print(f"[+] Model saved to: {model_path}")
    print(f"[+] Vectorizer saved to: {vectorizer_path}")
    print(f"[+] Metrics exported to: {metrics_path}")
    print("=" * 65 + "\n")
    return best_model_name, best_f1


if __name__ == '__main__':
    train_and_evaluate()
