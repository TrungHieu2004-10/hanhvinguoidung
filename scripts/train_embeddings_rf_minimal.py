#!/usr/bin/env python3
"""
scripts/train_embeddings_rf_minimal.py

Train classifier using SentenceTransformer embeddings + RandomForest.

Saves minimal artifacts:
 - models/rf_emb_model.pkl
 - models/sbert_model/   (saved SBERT directory)
 - models/label_encoder.pkl
 - models/emb_cfg.json

Usage example:
python scripts/train_embeddings_rf_minimal.py --data data/raw/tweets.csv --label_col is_there_an_emotion_directed_at_a_brand_or_product --oversample

Notes:
- For multilingual / Vietnamese data, use a multilingual SBERT model (see --model_name).
- RandomForest does not support early stopping; use n_estimators and class weights / oversampling to balance.
"""
import os
import argparse
import json
import joblib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

from sentence_transformers import SentenceTransformer

# ---------------- paths ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "tweets.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- simple cleaning ----------------
import re
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
MULTI_SPACE_RE = re.compile(r"\s+")

def clean_text_keep(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    t = URL_RE.sub(" ", text)
    t = MENTION_RE.sub(" ", t)
    t = MULTI_SPACE_RE.sub(" ", t).strip()
    return t

def detect_text_col(df: pd.DataFrame):
    candidates = ["tweet_text","text","tweet","content","message","body"]
    cols_lower = [c.lower() for c in df.columns]
    for c in candidates:
        if c in cols_lower:
            return df.columns[cols_lower.index(c)]
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None

def train(
    data_path,
    label_col,
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    test_size=0.2,
    random_state=42,
    oversample=False,
    rf_n_estimators=500,
    rf_max_depth=None,
    rf_n_jobs=-1
):
    # load
    df = pd.read_csv(data_path)
    text_col = detect_text_col(df)
    if text_col is None:
        raise ValueError("No text column detected in CSV.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV.")
    print("Text column:", text_col, "Label column:", label_col)

    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    texts_raw = df[text_col].astype(str).tolist()
    labels_raw = df[label_col].astype(str).tolist()
    print("Rows loaded:", len(texts_raw))

    # clean
    texts = [clean_text_keep(t) for t in texts_raw]

    # label encode
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)
    print("Label counts:", dict(zip(le.classes_, np.bincount(y))))

    # split
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y))>1 else None
    )
    print("Train size:", len(X_train_texts), "Test size:", len(X_test_texts))

    # load SBERT
    print("Loading SBERT model:", model_name)
    sbert = SentenceTransformer(model_name)

    # encode
    print("Encoding train embeddings...")
    emb_train = sbert.encode(X_train_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    print("Encoding test embeddings...")
    emb_test = sbert.encode(X_test_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    # optional oversample on train
    if oversample:
        print("Oversampling training set...")
        ros = RandomOverSampler(random_state=random_state)
        emb_train, y_train = ros.fit_resample(emb_train, y_train)
        print("After oversample counts:", Counter(y_train))

    # class weights computed from training labels (works well for RF)
    # Use balanced weights to mitigate imbalance (RF accepts dict mapping label->weight)
    classes_unique = np.unique(y_train)
    # compute weight inversely proportional to class frequency in train
    counts = np.bincount(y_train)
    class_weight = {int(cls): float(len(y_train) / (len(classes_unique) * counts[int(cls)])) for cls in classes_unique}
    print("Using class_weight (per-class):", class_weight)

    # RandomForest training
    clf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=rf_n_jobs
    )
    print("Training RandomForest...")
    clf.fit(emb_train, y_train)

    # evaluate
    y_pred = clf.predict(emb_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc*100:.2f}%")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0, target_names=le.classes_))

    # save minimal artifacts
    joblib.dump(clf, os.path.join(OUT_DIR, "rf_emb_model.pkl"))
    sbert.save(os.path.join(OUT_DIR, "sbert_model"))
    joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.pkl"))
    cfg = {
        "model": "rf_with_sbert_embeddings",
        "sbert_model_name": model_name,
        "rf_n_estimators": rf_n_estimators,
        "rf_max_depth": rf_max_depth,
        "oversample": oversample,
        "data_path": data_path,
        "text_col": text_col,
        "label_col": label_col
    }
    with open(os.path.join(OUT_DIR, "emb_cfg.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("Saved artifacts to:", OUT_DIR)
    return clf, sbert, le

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", "-d", default=DEFAULT_DATA)
    p.add_argument("--label_col", "-l", required=True)
    p.add_argument("--model_name", default="paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformer model")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--oversample", action="store_true")
    p.add_argument("--rf_n_estimators", type=int, default=500)
    p.add_argument("--rf_max_depth", type=int, default=None)
    p.add_argument("--rf_n_jobs", type=int, default=-1)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(
        data_path=args.data,
        label_col=args.label_col,
        model_name=args.model_name,
        test_size=args.test_size,
        random_state=args.random_state,
        oversample=args.oversample,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        rf_n_jobs=args.rf_n_jobs
    )
