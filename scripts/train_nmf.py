#!/usr/bin/env python3
# scripts/train_nmf.py
"""
Train a standalone NMF (Non-Negative Matrix Factorization) topic model.

- Uses TF-IDF vectorizer
- Saves:
    nmf_model_<n_topics>.pkl
    vectorizer_tfidf.pkl
"""

import os
import re
import argparse
import joblib
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


# ---------------- Paths ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "tweets.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- Preprocess ----------------
VI_CHARS = (
    "0-9a-zA-Záàảãạăắằẳẵặâấầẩẫậ"
    "éèẻẽẹêếềểễệíìỉĩị"
    "óòỏõọôốồổỗộơớờởỡợ"
    "úùủũụưứừửữự"
    "ýỳỷỹỵđ"
)

CLEAN_RE = re.compile(rf"[^{VI_CHARS}\s]", flags=re.UNICODE)
MULTI_SPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URL
    text = re.sub(r"@\w+", " ", text)              # @tag
    text = text.replace("#", " ")
    text = CLEAN_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text

def preprocess_series(series: pd.Series):
    series = series.dropna().astype(str)
    processed = series.apply(clean_text)
    processed = processed[processed.str.len() > 0]
    return processed.tolist()


# ---------------- Auto-detect text column ----------------
def detect_text_col(df: pd.DataFrame):
    candidates = ["text", "tweet", "content", "message", "body"]
    cols_lower = [c.lower() for c in df.columns]
    
    for c in candidates:
        if c in cols_lower:
            return df.columns[cols_lower.index(c)]
    
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None


# ---------------- Train NMF ----------------
def train_nmf(texts, n_topics=6, max_features=2000, stop_words="english"):
    tfidf = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=max_features,
        stop_words=stop_words
    )
    X = tfidf.fit_transform(texts)

    n_components = min(n_topics, X.shape[0])
    nmf = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=42,
        max_iter=250
    )
    nmf.fit(X)
    return nmf, tfidf


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Train standalone NMF topic model")
    p.add_argument("--data", "-d", default=DEFAULT_DATA, help="Path to CSV dataset")
    p.add_argument("--n_topics", "-k", type=int, default=6, help="Number of topics")
    p.add_argument("--max_features", type=int, default=2000, help="Max TF-IDF features")
    p.add_argument("--stop_words", default="english", help="Stop words (use None for Vietnamese)")
    return p.parse_args()


# ---------------- Main ----------------
def main():
    args = parse_args()

    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        return

    print(f"📥 Loading dataset: {args.data}")
    df = pd.read_csv(args.data)

    text_col = detect_text_col(df)
    if not text_col:
        print("❌ No text column detected.")
        return

    print(f"📝 Using text column: {text_col}")

    texts = preprocess_series(df[text_col])
    print(f"✅ Preprocessed {len(texts)} texts")

    if len(texts) < 10:
        print("⚠️ Warning: Too few texts for topic modeling!")

    print(f"⏳ Training NMF topic model with k={args.n_topics} ...")
    nmf, tfidf = train_nmf(
        texts,
        n_topics=args.n_topics,
        max_features=args.max_features,
        stop_words=(None if args.stop_words == "None" else args.stop_words)
    )

    # Save models
    nmf_name = f"nmf_model_{args.n_topics}.pkl"
    tfidf_name = "vectorizer_tfidf.pkl"

    joblib.dump(nmf, os.path.join(OUT_DIR, nmf_name))
    joblib.dump(tfidf, os.path.join(OUT_DIR, tfidf_name))

    print("\n🎉 Training complete.")
    print("Saved:")
    print(" -", nmf_name)
    print(" -", tfidf_name)
    print("\nModels stored in:", OUT_DIR)


if __name__ == "__main__":
    main()
