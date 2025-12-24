# __init__.py
"""
Package entry for social_sentiment (or any package name you want).
Exports commonly used functions for convenience.
"""
from .data_loader import load_csv
from .preprocessing import clean_text, simple_tokenize
from .sentiment import (
    train_sentiment_model,
    predict_sentiment,
    predict_sentiment_proba,
    sentiment_bucket,
)
from .lda_topic import extract_topics

__all__ = [
    "load_csv",
    "clean_text",
    "simple_tokenize",
    "train_sentiment_model",
    "predict_sentiment",
    "predict_sentiment_proba",
    "sentiment_bucket",
    "extract_topics",
]
