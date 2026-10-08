export interface ProjectFile {
  name: string;
  path: string;
  language: string;
  description: string;
  content: string;
}

export const PYTHON_PROJECT_FILES: ProjectFile[] = [
  {
    name: "train_model.py",
    path: "fake-news-detection/train_model.py",
    language: "python",
    description: "Complete ML pipeline: loads dataset, cleans text, trains Logistic Regression & Naive Bayes, evaluates metrics, and exports model.pkl & vectorizer.pkl.",
    content: `"""
Fake News Detection in Social Media Using Machine Learning
File: train_model.py
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODELS_DIR, "model.pkl")
VECTORIZER_FILE = os.path.join(MODELS_DIR, "vectorizer.pkl")
METADATA_FILE = os.path.join(MODELS_DIR, "metadata.json")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\\S+@\\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\\b\\d+\\b", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def run_training_pipeline():
    print("STEP 1: Loading Dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Flexible column detection
    if "title" in df.columns and "text" in df.columns:
        df["raw_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["raw_text"] = df["text"].fillna("")
    else:
        df["raw_text"] = df[df.columns[0]].fillna("")
        
    # Standardize label (0=REAL, 1=FAKE)
    label_col = [c for c in df.columns if c.lower() in ["label", "class", "target"]][0]
    df["target"] = df[label_col].map(lambda x: 1 if str(x).upper() in ["FAKE", "1"] else 0)
    
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""].drop_duplicates(subset=["clean_text"])
    
    X = df["clean_text"]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # TF-IDF Vectorization
    print("STEP 2: Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Logistic Regression
    print("STEP 3: Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    lr_preds = lr_model.predict(X_test_tfidf)
    lr_f1 = f1_score(y_test, lr_preds)
    
    # Train Multinomial Naive Bayes
    print("STEP 4: Training Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_tfidf, y_train)
    nb_preds = nb_model.predict(X_test_tfidf)
    nb_f1 = f1_score(y_test, nb_preds)
    
    # Select Best Model based on F1-score
    best_model = lr_model if lr_f1 >= nb_f1 else nb_model
    best_name = "Logistic Regression" if lr_f1 >= nb_f1 else "Multinomial Naive Bayes"
    print(f"Selected Best Model: {best_name}")
    
    # Save Artifacts
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print("Artifacts successfully saved to models/ directory!")

if __name__ == "__main__":
    run_training_pipeline()`
  },
  {
    name: "app.py",
    path: "fake-news-detection/app.py",
    language: "python",
    description: "Flask backend application with GET / and POST /predict endpoints, serving real-time predictions via Scikit-Learn.",
    content: `"""
Fake News Detection in Social Media Using Machine Learning
File: app.py - Flask Server
"""

import os
import re
import string
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "models", "model.pkl")
VECTORIZER_FILE = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# Load model and vectorizer once on startup
model = None
vectorizer = None

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\\S+@\\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\\b\\d+\\b", " ", text)
    return re.sub(r"\\s+", " ", text).strip()

@app.before_first_request
def load_assets():
    global model, vectorizer
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() if request.is_json else request.form
    news_text = data.get("news_text", "").strip()

    if not news_text or len(news_text.split()) < 3:
        return jsonify({"status": "error", "message": "Please enter at least 3 words."}), 400

    cleaned = clean_text(news_text)
    features = vectorizer.transform([cleaned])

    probabilities = model.predict_proba(features)[0]
    real_prob, fake_prob = float(probabilities[0]), float(probabilities[1])

    is_fake = fake_prob >= 0.5
    prediction = "FAKE NEWS" if is_fake else "REAL NEWS"
    confidence = round((fake_prob if is_fake else real_prob) * 100, 2)

    return jsonify({
        "status": "success",
        "prediction": prediction,
        "prediction_code": 1 if is_fake else 0,
        "confidence": confidence,
        "probabilities": {"real": round(real_prob * 100, 2), "fake": round(fake_prob * 100, 2)},
        "model_used": "Logistic Regression",
        "word_count": len(news_text.split()),
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M:%S %p"),
        "disclaimer": "Prediction is based on patterns learned from the training dataset."
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)`
  },
  {
    name: "requirements.txt",
    path: "fake-news-detection/requirements.txt",
    language: "text",
    description: "Python package dependencies for local execution and cloud deployment on Render.",
    content: `flask>=3.0.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
joblib>=1.3.2
gunicorn>=21.2.0`
  },
  {
    name: "Procfile",
    path: "fake-news-detection/Procfile",
    language: "text",
    description: "Process file for running with Gunicorn on Render, Railway, or Heroku.",
    content: `web: gunicorn app:app --bind 0.0.0.0:$PORT`
  },
  {
    name: "dataset.csv",
    path: "fake-news-detection/data/dataset.csv",
    language: "csv",
    description: "Balanced dataset of verified real and fake news articles with title, text, and label columns.",
    content: `title,text,label
"NASA James Webb Space Telescope discovers oldest galaxy","Astronomers using the James Webb Space Telescope identified the most distant and oldest galaxy ever observed...","REAL"
"SHOCKING: Secret cure for cancer found in lemon peels!","An anonymous whistleblower revealed that boiling lemon peels cures stage 4 cancers in 48 hours...","FAKE"
"Federal Reserve holds benchmark rate steady","The Federal Reserve announced on Wednesday that it will maintain the benchmark interest rate...","REAL"
"ALERT: Drinking boiled garlic water cures all viruses overnight","A top military doctor leaked ancient formula that destroys all viruses within six hours...","FAKE"`
  }
];
