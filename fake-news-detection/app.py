"""
=============================================================================
Fake News Detection in Social Media Using Machine Learning
File: app.py
Flask Web Application Backend
=============================================================================
Routes:
  GET  /        - Renders interactive home dashboard
  POST /predict - Accepts news text, preprocesses, predicts via Scikit-Learn
  GET  /metrics - Returns model evaluation metrics, confusion matrix, dataset info
=============================================================================
"""

import os
import re
import string
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# 1. Flask App Initialization & Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "model.pkl")
VECTORIZER_FILE = os.path.join(MODELS_DIR, "vectorizer.pkl")
METADATA_FILE = os.path.join(MODELS_DIR, "metadata.json")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Global variables loaded once on startup
model = None
vectorizer = None
metadata = {}


# ---------------------------------------------------------------------------
# 2. Text Preprocessing (Identical to Training Preprocessing)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Cleans raw input text identically to training time.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 3. Model Loader (Runs once at startup)
# ---------------------------------------------------------------------------
def load_ml_assets():
    global model, vectorizer, metadata
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
            model = joblib.load(MODEL_FILE)
            vectorizer = joblib.load(VECTORIZER_FILE)
            if os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            print("[INFO] Model, Vectorizer, and Metadata loaded successfully!")
        else:
            print("[WARNING] Model artifacts not found. Attempting automatic training...")
            from train_model import run_training_pipeline
            run_training_pipeline()
            model = joblib.load(MODEL_FILE)
            vectorizer = joblib.load(VECTORIZER_FILE)
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print("[INFO] Model trained and loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Could not load or train model: {e}")


# Run loader on startup
load_ml_assets()


# ---------------------------------------------------------------------------
# 4. Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """
    Renders the primary web interface.
    """
    return render_template("index.html", metadata=metadata)


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """
    API endpoint to fetch model evaluation metrics for the UI.
    """
    return jsonify({
        "status": "success",
        "metadata": metadata
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint to analyze submitted news text.
    Accepts both JSON payloads and standard form submissions.
    """
    global model, vectorizer, metadata

    # 1. Extract text from request
    if request.is_json:
        data = request.get_json()
        news_text = data.get("news_text", "")
        model_choice = data.get("model_choice", "best")  # 'best', 'lr', 'nb'
    else:
        news_text = request.form.get("news_text", "")
        model_choice = request.form.get("model_choice", "best")

    # 2. Input validation
    if not news_text or not news_text.strip():
        return jsonify({
            "status": "error",
            "message": "Please enter or paste a news article, headline, or post to analyze."
        }), 400

    raw_text = news_text.strip()
    word_count = len(raw_text.split())
    text_length = len(raw_text)

    if word_count < 3:
        return jsonify({
            "status": "error",
            "message": "Input text is too short. Please provide at least 3 words for accurate analysis."
        }), 400

    # Ensure model is loaded
    if model is None or vectorizer is None:
        load_ml_assets()
        if model is None or vectorizer is None:
            return jsonify({
                "status": "error",
                "message": "Machine learning model is not available. Please run train_model.py first."
            }), 500

    # 3. Apply identical preprocessing
    cleaned = clean_text(raw_text)
    if not cleaned:
        return jsonify({
            "status": "error",
            "message": "Text contained only punctuation or symbols. Please provide meaningful words."
        }), 400

    # 4. TF-IDF Transformation
    try:
        tfidf_features = vectorizer.transform([cleaned])

        # 5. Predict & Calculate Confidence Probability
        # Label convention: 0 = REAL, 1 = FAKE
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(tfidf_features)[0]
            real_prob = float(probabilities[0])
            fake_prob = float(probabilities[1])

            if fake_prob >= 0.5:
                pred_label = "FAKE NEWS"
                pred_code = 1
                confidence_pct = round(fake_prob * 100, 2)
            else:
                pred_label = "REAL NEWS"
                pred_code = 0
                confidence_pct = round(real_prob * 100, 2)
        else:
            # Fallback for models without predict_proba
            pred_code = int(model.predict(tfidf_features)[0])
            pred_label = "FAKE NEWS" if pred_code == 1 else "REAL NEWS"
            confidence_pct = 85.0
            real_prob = 0.15 if pred_code == 1 else 0.85
            fake_prob = 0.85 if pred_code == 1 else 0.15

        # 6. Extract key informative words found in this specific text
        matched_tokens = []
        words = cleaned.split()
        vocab = vectorizer.vocabulary_
        for w in set(words):
            if w in vocab:
                matched_tokens.append(w)
        # Limit to top 8 tokens
        matched_tokens = matched_tokens[:8]

        model_name = metadata.get("best_model_name", "Logistic Regression")

        explanation = (
            f"The trained {model_name} classifier identified language patterns consistent "
            f"with {'unverified or sensationalized claims' if pred_code == 1 else 'credible factual reporting'} "
            f"in the training corpus."
        )

        timestamp_str = datetime.now().strftime("%d %b %Y, %I:%M:%S %p")

        disclaimer = (
            "Prediction is based on statistical patterns learned from the training dataset "
            "and should not be treated as definitive proof that a news story is true or false."
        )

        response_data = {
            "status": "success",
            "prediction": pred_label,
            "prediction_code": pred_code,
            "confidence": confidence_pct,
            "probabilities": {
                "real": round(real_prob * 100, 2),
                "fake": round(fake_prob * 100, 2)
            },
            "explanation": explanation,
            "model_used": model_name,
            "text_length": text_length,
            "word_count": word_count,
            "timestamp": timestamp_str,
            "matched_keywords": matched_tokens,
            "disclaimer": disclaimer
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500


# ---------------------------------------------------------------------------
# 5. App Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
