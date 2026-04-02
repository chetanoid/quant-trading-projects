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
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = [-1, 0, 1]
LABEL_NAMES = ["negative", "neutral", "positive"]

# Attempt to import matplotlib for visualising confusion matrices.  If
# not available, the script will run without generating figures.
def build_fallback_data() -> pd.DataFrame:
    """Generate a larger synthetic finance-news dataset for offline demos."""
    entities = [
        "Bank",
        "Chipmaker",
        "Retailer",
        "Brokerage",
        "Energy group",
        "Software company",
        "Industrial firm",
        "Automaker",
    ]
    positive_events = [
        "beats estimates and raises guidance",
        "posts strong margins and record revenue",
        "reports robust demand and improving cash flow",
        "announces upbeat outlook after a strong quarter",
        "wins analyst upgrades on accelerating growth",
    ]
    neutral_events = [
        "reiterates full-year guidance and leaves forecasts unchanged",
        "announces a management update with no financial impact",
        "trades flat as investors wait for more data",
        "holds an investor presentation without changing targets",
        "reports results broadly in line with expectations",
    ]
    negative_events = [
        "misses earnings estimates and cuts guidance",
        "warns of weaker demand and softer margins",
        "falls after a profit warning and lower outlook",
        "faces credit concerns and rising losses",
        "slides on disappointing results and slower growth",
    ]

    rows = []
    for entity in entities:
        for event in positive_events:
            rows.append(("positive", f"{entity} {event}."))
        for event in neutral_events:
            rows.append(("neutral", f"{entity} {event}."))
        for event in negative_events:
            rows.append(("negative", f"{entity} {event}."))
    return pd.DataFrame(rows, columns=["label", "text"])


try:
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))
    import matplotlib.pyplot as plt  # type: ignore
    _HAVE_MATPLOTLIB = True
except Exception:
    _HAVE_MATPLOTLIB = False


def load_data(path: str = "all-data.csv") -> tuple[pd.DataFrame, bool]:
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
    label_map = {"positive": 1, "neutral": 0, "negative": -1}

    if os.path.isfile(path):
        df = pd.read_csv(path, names=["label", "text"], encoding="utf-8")
        used_fallback = False
    else:
        df = build_fallback_data()
        used_fallback = True

    df["label"] = df["label"].astype(str).str.strip().str.lower().map(label_map)
    df["text"] = df["text"].astype(str).str.strip()
    df = df.dropna(subset=["label", "text"])
    df = df[df["text"] != ""].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df, used_fallback


def _print_report(model_name: str, y_true: pd.Series, predictions) -> None:
    accuracy = accuracy_score(y_true, predictions)
    print(f"{model_name} Accuracy: {accuracy:.2%}")
    print(f"Classification Report ({model_name}):")
    print(
        classification_report(
            y_true,
            predictions,
            labels=LABELS,
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )


def _save_confusion_matrix(y_true: pd.Series, predictions, title: str, output_file: str) -> None:
    if not _HAVE_MATPLOTLIB:
        return

    try:
        matrix = confusion_matrix(y_true, predictions, labels=LABELS)
        plt.figure()
        plt.imshow(matrix, interpolation="nearest")
        plt.title(title)
        plt.colorbar()
        tick_marks = range(len(LABEL_NAMES))
        plt.xticks(tick_marks, LABEL_NAMES, rotation=45)
        plt.yticks(tick_marks, LABEL_NAMES)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Saved {output_file}")
    except Exception:
        pass


def train_models(df: pd.DataFrame) -> None:
    """Train logistic regression and random forest classifiers on the dataset.

    Splits the data into training and test sets, fits the models,
    evaluates them, and prints accuracy and classification reports.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    preds_lr = log_reg.predict(X_test)
    _print_report("Logistic Regression", y_test, preds_lr)
    _save_confusion_matrix(
        y_test,
        preds_lr,
        "Logistic Regression Confusion Matrix",
        "logistic_confusion_matrix.png",
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    _print_report("Random Forest", y_test, preds_rf)
    _save_confusion_matrix(
        y_test,
        preds_rf,
        "Random Forest Confusion Matrix",
        "random_forest_confusion_matrix.png",
    )


def main() -> None:
    df, used_fallback = load_data()
    if used_fallback:
        print("Dataset file 'all-data.csv' not found. Using embedded fallback samples.")
    train_models(df)


if __name__ == "__main__":
    main()
