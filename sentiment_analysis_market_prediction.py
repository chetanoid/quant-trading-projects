"""
Sentiment Analysis & Market Direction Prediction
==============================================

This script demonstrates a simple natural language processing (NLP) pipeline for sentiment
analysis and prediction of market moves.  A small synthetic dataset of news headlines and
social media posts is created with sentiment labels (positive or negative).  The text is
vectorised with TF‑IDF and used to train two classifiers: logistic regression and
random forest.  The models' accuracies are printed, and predictions on a few sample
unseen texts are shown.  The aim is to illustrate a basic workflow for transforming
text into features and fitting classification models, not to build a production‑grade
trading signal.

Author: OpenAI assistant
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def create_dataset() -> Tuple[List[str], List[int]]:
    """Create a toy dataset of market‑related sentences with sentiment labels."""
    texts = [
        "Company X reports record profits and beats analyst expectations",
        "Shares of Company Y plunge after weak earnings report",
        "Global markets rally on hopes of central bank stimulus",
        "Concerns over rising inflation send stocks lower",
        "Breakthrough technology by startup Z excites investors",
        "Unexpected decline in retail sales triggers market sell‑off",
        "Analysts upgrade stock A following strong guidance",
        "Political uncertainty weighs on currency markets",
        "New product launch from Company B receives positive reviews",
        "Economic data disappoints, sparking fears of recession",
        # Social media‑style posts
        "Feeling bullish on crypto today!",
        "Market looks scary, selling all my positions",
        "Wow, what an incredible run for tech stocks!",
        "Panic selling everywhere, this can't end well",
        "So excited about the new iPhone release",
        "This rate hike will crush growth stocks",
    ]
    # 1 for positive sentiment (expecting market up), 0 for negative sentiment
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0,  # 10 news headlines
              1, 0, 1, 0, 1, 0]  # 6 social media posts
    return texts, labels


def build_models(texts: List[str], labels: List[int]):
    """Vectorise text and train logistic regression and random forest models."""
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    log_reg = LogisticRegression(max_iter=100)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    return vectorizer, log_reg, acc_lr, rf, acc_rf


def main():
    texts, labels = create_dataset()
    vectorizer, log_reg, acc_lr, rf, acc_rf = build_models(texts, labels)
    print("Sentiment Analysis Model Performance")
    print(f"Logistic Regression Accuracy: {acc_lr * 100:.1f}%")
    print(f"Random Forest Accuracy:    {acc_rf * 100:.1f}%")

    # Test on unseen examples
    test_examples = [
        "Central bank hints at tightening, markets jittery",
        "Record‑breaking quarter sends stock soaring",
        "Economic slowdown worries economists",
        "Excited about the new electric car release",
    ]
    X_new = vectorizer.transform(test_examples)
    predictions_lr = log_reg.predict(X_new)
    predictions_rf = rf.predict(X_new)

    for text, lr_pred, rf_pred in zip(test_examples, predictions_lr, predictions_rf):
        print("\nHeadline:", text)
        print("  Logistic Regression prediction:", "Positive" if lr_pred == 1 else "Negative")
        print("  Random Forest prediction:    ", "Positive" if rf_pred == 1 else "Negative")


if __name__ == "__main__":
    main()