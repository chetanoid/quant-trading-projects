"""
Real‑Data Sentiment Analysis & Market Prediction
==============================================

This script performs sentiment classification on a real financial news
dataset (``all-data.csv``) and illustrates how natural language
processing (NLP) can be used in a trading context.  The dataset
contains short news sentences labelled as ``positive``, ``neutral`` or
``negative``.  We map these labels to numeric classes, build a
TF‑IDF representation, train a logistic regression classifier, and
report accuracy and a full classification report.  A random forest
classifier is also included for comparison.

The ``all-data.csv`` file is expected to reside in the same directory
as this script; if you do not have it, download it from the open
``sentiment-analysis-for-financial-news`` repository on GitHub.  The
dataset is relatively small (~6 k lines) and suitable for quick
prototyping.

Author: OpenAI assistant
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_data(path: str = "all-data.csv") -> pd.DataFrame:
    """Load and preprocess the sentiment dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file.  The CSV should have two columns: a label
        (``positive``, ``neutral`` or ``negative``) and the text.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``label`` and ``text`` where ``label`` is
        mapped to integers.
    """
    df = pd.read_csv(path, names=["label", "text"], encoding="utf-8")
    label_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["label"] = df["label"].str.strip().map(label_map)
    return df


def train_models(df: pd.DataFrame) -> None:
    """Train logistic regression and random forest classifiers on the dataset.

    Splits the data into training and test sets, fits the models,
    evaluates them, and prints accuracy and classification reports.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(df["text"].astype(str))
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    preds_lr = log_reg.predict(X_test)
    acc_lr = accuracy_score(y_test, preds_lr)
    print("Logistic Regression Accuracy: {:.2%}".format(acc_lr))
    print("Classification Report (Logistic Regression):")
    print(classification_report(y_test, preds_lr, target_names=["negative", "neutral", "positive"]))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, preds_rf)
    print("Random Forest Accuracy: {:.2%}".format(acc_rf))
    print("Classification Report (Random Forest):")
    print(classification_report(y_test, preds_rf, target_names=["negative", "neutral", "positive"]))


def main() -> None:
    df = load_data()
    train_models(df)


if __name__ == "__main__":
    main()