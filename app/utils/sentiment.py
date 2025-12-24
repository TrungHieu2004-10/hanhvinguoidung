# sentiment.py
from typing import Tuple, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_sentiment_model(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
) -> Tuple[Pipeline, float]:
    """
    Train a simple sentiment classifier (TF-IDF + LogisticRegression).

    Returns:
        (trained_pipeline, accuracy_on_test_set)

    Raises:
        ValueError if there are too few labeled samples or columns missing.
    """
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError("Specified text_col or label_col not found in DataFrame")

    data = df[[text_col, label_col]].dropna()
    if len(data) < 10:
        raise ValueError(
            f"Too few labeled samples for training (found {len(data)}, need >= 10)."
        )

    X = data[text_col].astype(str)
    y = data[label_col].astype(str)

    # stratify only if more than one class in y
    stratify = y if len(y.unique()) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    clf = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
            ("logreg", LogisticRegression(max_iter=200)),
        ]
    )

    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    return clf, float(acc)


def predict_sentiment(model: Pipeline, texts: List[str]) -> List[Any]:
    """
    Predict labels for a list of texts using the trained pipeline.
    """
    if not hasattr(model, "predict"):
        raise ValueError("model does not support predict()")
    return model.predict(texts).tolist()


def predict_sentiment_proba(model: Pipeline, texts: List[str]) -> List[List[float]]:
    """
    Predict probability distribution for each input text.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("model does not support predict_proba()")
    proba = model.predict_proba(texts)
    return np.array(proba).tolist()


def sentiment_bucket(label: str) -> str:
    """
    Map a label string to 'Positive' | 'Negative' | 'Other'
    Uses lowercase contains checks for common tokens.
    """
    label_lower = str(label).lower()
    if any(x in label_lower for x in ("pos", "positive", "1", "good")):
        return "Positive"
    if any(x in label_lower for x in ("neg", "negative", "0", "bad")):
        return "Negative"
    return "Other"
