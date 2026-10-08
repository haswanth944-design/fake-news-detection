"""
=============================================================================
Fake News Detection in Social Media Using Machine Learning
File: train_model.py
CSE Mini Project - Model Training & Evaluation Pipeline
=============================================================================
This script:
  1. Dynamically detects text and label columns from dataset.csv
  2. Combines title and text if both exist
  3. Automatically resolves label semantics (identifies REAL vs FAKE)
  4. Preprocesses text (lowercase, regex cleanup, URL/HTML removal, punctuation)
  5. Extracts features using TF-IDF Vectorizer (n-grams (1, 2))
  6. Trains both Logistic Regression and Multinomial Naive Bayes models
  7. Evaluates both models: Accuracy, Precision, Recall, F1-score, Confusion Matrix
  8. Automatically selects the top-performing model based on F1-score
  9. Saves model.pkl, vectorizer.pkl, and metadata.json in models/ directory
=============================================================================
"""

import os
import re
import string
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------------------------
# 1. Configuration & Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODELS_DIR, "model.pkl")
VECTORIZER_FILE = os.path.join(MODELS_DIR, "vectorizer.pkl")
METADATA_FILE = os.path.join(MODELS_DIR, "metadata.json")

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 2. Text Preprocessing Function (Identical during Training & Prediction)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Cleans raw text for machine learning input:
      - Converts to string and lowercases
      - Strips URLs and hyper-links
      - Strips HTML tags and angle brackets
      - Strips email addresses and mentions
      - Removes special punctuation and non-alphanumeric symbols
      - Normalizes repeated whitespaces
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove punctuation and special characters
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)

    # Remove isolated digits/numbers while retaining words
    text = re.sub(r"\b\d+\b", " ", text)

    # Collapse multi-spaces and strip leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# 3. Flexible Dataset Column & Label Detection
# ---------------------------------------------------------------------------
def load_and_prepare_dataset(file_path: str):
    """
    Loads dataset and intelligently detects text and label columns.
    Handles 'title', 'content', 'text', 'label', 'class', etc.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please place your dataset.csv in data/")

    df = pd.read_csv(file_path)
    print("=" * 70)
    print("STEP 1: DATASET INSPECTION & COLUMN DETECTION")
    print("=" * 70)
    print(f"Dataset Loaded: {len(df)} initial rows")
    print(f"Available Columns: {list(df.columns)}")

    # Detect label column
    possible_label_cols = ["label", "class", "target", "category", "news_type", "is_fake"]
    label_col = None
    for col in df.columns:
        if col.lower().strip() in possible_label_cols:
            label_col = col
            break

    if not label_col:
        # Fallback to the last column
        label_col = df.columns[-1]
    print(f"-> Detected Label Column: '{label_col}'")

    # Detect text columns (title, text, content, headline, body)
    title_col = None
    text_col = None
    for col in df.columns:
        c_lower = col.lower().strip()
        if c_lower in ["title", "headline"]:
            title_col = col
        elif c_lower in ["text", "content", "body", "article", "statement", "news"]:
            text_col = col

    # Combine or assign feature column
    if title_col and text_col and title_col != text_col:
        print(f"-> Combining title ('{title_col}') and text ('{text_col}') into 'combined_text'")
        df["raw_text"] = df[title_col].fillna("") + " " + df[text_col].fillna("")
    elif text_col:
        print(f"-> Using text column: '{text_col}'")
        df["raw_text"] = df[text_col].fillna("")
    elif title_col:
        print(f"-> Using title column: '{title_col}'")
        df["raw_text"] = df[title_col].fillna("")
    else:
        # Use first non-label column
        candidate_cols = [c for c in df.columns if c != label_col]
        if not candidate_cols:
            raise ValueError("No text feature column found in dataset!")
        df["raw_text"] = df[candidate_cols[0]].fillna("")
        print(f"-> Using default candidate column: '{candidate_cols[0]}'")

    # Inspect missing values
    missing_texts = df["raw_text"].isna().sum() + (df["raw_text"].str.strip() == "").sum()
    missing_labels = df[label_col].isna().sum()
    print(f"Missing/Blank Texts: {missing_texts} | Missing Labels: {missing_labels}")

    # Drop missing or empty rows
    df = df[df["raw_text"].str.strip() != ""]
    df = df[~df[label_col].isna()].copy()

    # Drop duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=["raw_text"])
    print(f"Removed {initial_len - len(df)} duplicate records. Clean row count: {len(df)}")

    # -----------------------------------------------------------------------
    # Intelligent Label Mapping (Do NOT assume 0 is Fake or 1 is Fake)
    # Target standard: 0 = REAL, 1 = FAKE
    # -----------------------------------------------------------------------
    raw_labels = df[label_col].unique()
    print(f"Raw Label Classes Found: {raw_labels}")

    label_mapping = {}
    for val in raw_labels:
        s_val = str(val).strip().upper()
        if s_val in ["FAKE", "FALSE", "0", "F", "SPAM", "UNRELIABLE"]:
            # Check context: if numeric 0 was used, check if 1 is FAKE or REAL
            # Standardizing: 1 = FAKE, 0 = REAL
            label_mapping[val] = 1 if s_val in ["FAKE", "FALSE", "F", "SPAM", "UNRELIABLE"] else None
        elif s_val in ["REAL", "TRUE", "1", "T", "HAM", "RELIABLE"]:
            label_mapping[val] = 0 if s_val in ["REAL", "TRUE", "T", "HAM", "RELIABLE"] else None

    # Handle numeric edge case: if labels are strictly 0 and 1 without text tags
    if any(v is None for v in label_mapping.values()):
        # Check standard convention in Kaggle fake news datasets where 1=FAKE, 0=REAL
        for val in raw_labels:
            s_val = str(val).strip()
            if s_val == "1":
                label_mapping[val] = 1  # FAKE
            elif s_val == "0":
                label_mapping[val] = 0  # REAL
            else:
                # Default alphabetical
                label_mapping[val] = 1 if "fake" in s_val.lower() else 0

    df["target"] = df[label_col].map(label_mapping)
    real_count = (df["target"] == 0).sum()
    fake_count = (df["target"] == 1).sum()

    print(f"Detected Label Mapping: {label_mapping}")
    print(f"Label Distribution: REAL (0) = {real_count} | FAKE (1) = {fake_count}")

    # Clean text column
    print("-> Applying text cleaning pipeline...")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    # Remove any rows that became blank after cleaning
    df = df[df["clean_text"].str.strip() != ""].copy()

    return df, label_mapping


# ---------------------------------------------------------------------------
# 4. Training Pipeline & Model Comparison
# ---------------------------------------------------------------------------
def run_training_pipeline():
    print("=" * 70)
    print("FAKE NEWS DETECTION - MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load dataset
    df, label_map = load_and_prepare_dataset(DATA_PATH)

    # 2. Train / Test Split (80% Train, 20% Test)
    X = df["clean_text"]
    y = df["target"]

    # Check stratification possibility
    min_class_count = min((y == 0).sum(), (y == 1).sum())
    stratify = y if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )
    print(f"Train Set Size: {len(X_train)} samples | Test Set Size: {len(X_test)} samples")

    # 3. TF-IDF Vectorization
    print("=" * 70)
    print("STEP 2: TF-IDF VECTORIZATION")
    print("=" * 70)
    print("Parameters: max_features=10000, ngram_range=(1, 2), stop_words='english'")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary Size: {len(vectorizer.vocabulary_)} features")

    # 4. Model 1: Logistic Regression
    print("=" * 70)
    print("STEP 3: TRAINING LOGISTIC REGRESSION")
    print("=" * 70)
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
    lr_model.fit(X_train_tfidf, y_train)
    lr_preds = lr_model.predict(X_test_tfidf)

    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec = precision_score(y_test, lr_preds, zero_division=0)
    lr_rec = recall_score(y_test, lr_preds, zero_division=0)
    lr_f1 = f1_score(y_test, lr_preds, zero_division=0)
    lr_cm = confusion_matrix(y_test, lr_preds).tolist()

    print(f"Logistic Regression -> Accuracy: {lr_acc:.4f} | Precision: {lr_prec:.4f} | Recall: {lr_rec:.4f} | F1: {lr_f1:.4f}")

    # 5. Model 2: Multinomial Naive Bayes
    print("=" * 70)
    print("STEP 4: TRAINING MULTINOMIAL NAIVE BAYES")
    print("=" * 70)
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_tfidf, y_train)
    nb_preds = nb_model.predict(X_test_tfidf)

    nb_acc = accuracy_score(y_test, nb_preds)
    nb_prec = precision_score(y_test, nb_preds, zero_division=0)
    nb_rec = recall_score(y_test, nb_preds, zero_division=0)
    nb_f1 = f1_score(y_test, nb_preds, zero_division=0)
    nb_cm = confusion_matrix(y_test, nb_preds).tolist()

    print(f"Naive Bayes         -> Accuracy: {nb_acc:.4f} | Precision: {nb_prec:.4f} | Recall: {nb_rec:.4f} | F1: {nb_f1:.4f}")

    # 6. Model Comparison Table
    print("=" * 70)
    print("STEP 5: MODEL COMPARISON & SELECTION")
    print("=" * 70)
    print(f"{'Model':<24} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 70)
    print(f"{'Logistic Regression':<24} | {lr_acc*100:>8.2f}% | {lr_prec*100:>8.2f}% | {lr_rec*100:>8.2f}% | {lr_f1*100:>8.2f}%")
    print(f"{'Multinomial Naive Bayes':<24} | {nb_acc*100:>8.2f}% | {nb_prec*100:>8.2f}% | {nb_rec*100:>8.2f}% | {nb_f1*100:>8.2f}%")
    print("-" * 70)

    # Automatic selection based on F1-score (harmonic mean of precision and recall)
    if lr_f1 >= nb_f1:
        best_model_name = "Logistic Regression"
        best_model = lr_model
        best_metrics = {
            "accuracy": round(lr_acc * 100, 2),
            "precision": round(lr_prec * 100, 2),
            "recall": round(lr_rec * 100, 2),
            "f1_score": round(lr_f1 * 100, 2),
            "confusion_matrix": lr_cm
        }
    else:
        best_model_name = "Multinomial Naive Bayes"
        best_model = nb_model
        best_metrics = {
            "accuracy": round(nb_acc * 100, 2),
            "precision": round(nb_prec * 100, 2),
            "recall": round(nb_rec * 100, 2),
            "f1_score": round(nb_f1 * 100, 2),
            "confusion_matrix": nb_cm
        }

    print(f"-> Selected Best Model: {best_model_name} (F1 Score: {best_metrics['f1_score']}%)")

    # 7. Classification Report
    print("\nDetailed Classification Report for Best Model:")
    active_preds = lr_preds if best_model_name == "Logistic Regression" else nb_preds
    print(classification_report(y_test, active_preds, target_names=["REAL", "FAKE"], zero_division=0))

    # 8. Save Artifacts
    print("=" * 70)
    print("STEP 6: SAVING ARTIFACTS")
    print("=" * 70)
    joblib.dump(best_model, MODEL_FILE)
    print(f"[OK] Model saved to: {MODEL_FILE}")

    joblib.dump(vectorizer, VECTORIZER_FILE)
    print(f"[OK] Vectorizer saved to: {VECTORIZER_FILE}")

    # Extract Top Indicative Words (for viva explanation and explainability)
    feature_names = np.array(vectorizer.get_feature_names_out())
    if hasattr(lr_model, "coef_"):
        # For Logistic Regression, positive weights indicate Fake (class 1), negative indicate Real (class 0)
        coefs = lr_model.coef_[0]
        top_fake_indices = np.argsort(coefs)[-15:][::-1]
        top_real_indices = np.argsort(coefs)[:15]
        top_fake_words = [feature_names[i] for i in top_fake_indices]
        top_real_words = [feature_names[i] for i in top_real_indices]
    else:
        top_fake_words = []
        top_real_words = []

    # Metadata dictionary
    metadata = {
        "best_model_name": best_model_name,
        "metrics": best_metrics,
        "all_models": {
            "logistic_regression": {
                "accuracy": round(lr_acc * 100, 2),
                "precision": round(lr_prec * 100, 2),
                "recall": round(lr_rec * 100, 2),
                "f1_score": round(lr_f1 * 100, 2),
                "confusion_matrix": lr_cm
            },
            "naive_bayes": {
                "accuracy": round(nb_acc * 100, 2),
                "precision": round(nb_prec * 100, 2),
                "recall": round(nb_rec * 100, 2),
                "f1_score": round(nb_f1 * 100, 2),
                "confusion_matrix": nb_cm
            }
        },
        "dataset_summary": {
            "total_records": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "real_samples": int((df["target"] == 0).sum()),
            "fake_samples": int((df["target"] == 1).sum()),
            "vocabulary_size": len(vectorizer.vocabulary_)
        },
        "label_mapping": {
            "0": "REAL",
            "1": "FAKE"
        },
        "top_fake_indicative_words": top_fake_words,
        "top_real_indicative_words": top_real_words
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[OK] Metadata saved to: {METADATA_FILE}")

    print("=" * 70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
