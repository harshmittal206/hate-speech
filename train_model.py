"""
Improved 3-class hate-speech / offensive-language classifier.

Run from this directory:
    python train_model.py

Artifacts:
    model.pkl, vectorizer.pkl, metrics.json
"""
import json
import pickle
import re
import html
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "labeled_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

LABEL_MAP = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "No Hate or Offensive Language",
}
LABEL_ORDER = [
    "Hate Speech",
    "Offensive Language",
    "No Hate or Offensive Language",
]

def clean_data(text):
    """Light normalization; deliberately preserves negation and profanity."""
    text = html.unescape(str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\brt\b", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_vectorizer():
    # Word n-grams learn phrases; character n-grams help with spelling variation.
    return FeatureUnion([
        ("word", TfidfVectorizer(
            preprocessor=clean_data,
            tokenizer=str.split,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=100000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("char", TfidfVectorizer(
            preprocessor=clean_data,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=60000,
            sublinear_tf=True,
        )),
    ])

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"class", "tweet"}
    if not required.issubset(df.columns):
        raise ValueError("labeled_data.csv must contain 'class' and 'tweet' columns.")

    df = df[["class", "tweet"]].dropna()
    df["label"] = df["class"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["tweet"] = df["tweet"].astype(str)
    df = df[df["tweet"].str.strip().ne("")]

    X_train, X_test, y_train, y_test = train_test_split(
        df["tweet"], df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    print("Fitting word + character TF-IDF features...")
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        C=3.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )
    print("Training balanced Logistic Regression...")
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    report = classification_report(
        y_test, predictions, labels=LABEL_ORDER,
        output_dict=True, zero_division=0
    )
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=LABEL_ORDER, zero_division=0
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=LABEL_ORDER))

    metrics = {
        "model": "Balanced Logistic Regression",
        "features": "Word TF-IDF (1,2) + character TF-IDF (3,5)",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "class_report": report,
        "class_order": LABEL_ORDER,
        "dataset_rows": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "note": "Held-out random stratified split; results are dataset-specific and not a guarantee of real-world performance."
    }

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)
    with VECTORIZER_PATH.open("wb") as f:
        pickle.dump(vectorizer, f)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("\nSaved model.pkl, vectorizer.pkl and metrics.json")

if __name__ == "__main__":
    main()
