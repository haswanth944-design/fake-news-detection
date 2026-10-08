# Fake News Detection in Social Media Using Machine Learning

A complete, production-grade 1-week CSE mini project implementing supervised Machine Learning and Natural Language Processing (NLP) with Scikit-Learn and Flask to detect whether news articles or social media posts are **REAL** or **FAKE**.

---

## 1. Project Title
**Fake News Detection in Social Media Using Machine Learning**

## 2. Problem Statement
The exponential growth of social media platforms (X/Twitter, Facebook, WhatsApp, Telegram) has facilitated the viral dissemination of unverified information, rumors, and deliberately engineered fake news. Misinformation distorts public perception regarding healthcare, financial markets, politics, and scientific consensus. Manual fact-checking cannot scale with the velocity of social media publications. An automated, cost-effective, machine-learning-driven textual classification pipeline is required to assess news authenticity.

## 3. Project Objectives
- Build an end-to-end, free-of-cost fake news classification web system.
- Implement an automated text preprocessing pipeline for unstructured textual articles.
- Extract statistical lexical features using Term Frequency-Inverse Document Frequency (TF-IDF).
- Train and compare two classic classifiers: **Logistic Regression** and **Multinomial Naive Bayes**.
- Empirically evaluate performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- Automatically serialize the optimal model (`model.pkl`) and TF-IDF vectorizer (`vectorizer.pkl`).
- Build an interactive Flask web UI allowing users to paste news text and view confidence scores, word stats, and explanations.

## 4. Technologies Used
- **Language**: Python 3.10+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning & NLP**: Scikit-Learn (`TfidfVectorizer`, `LogisticRegression`, `MultinomialNB`, `metrics`)
- **Model Serialization**: Joblib / Pickle
- **Backend Framework**: Flask 3.x
- **Web Frontend**: HTML5, CSS3, JavaScript (AJAX Fetch API)
- **Production Server**: Gunicorn

---

## 5. Dataset Details
The dataset is stored in `data/dataset.csv`.
- **Columns Supported**: Flexible detection for `title`, `text`, `content`, `label`, `class`.
- If both `title` and `text` are detected, the training pipeline automatically combines them into `combined_text`.
- **Label Handling**: Does **not** assume 0 or 1 is arbitrarily fake. The pipeline inspects raw values (`REAL`, `FAKE`, `TRUE`, `FALSE`, `0`, `1`) and creates an explicit binary mapping:
  - `0` &rarr; `REAL NEWS`
  - `1` &rarr; `FAKE NEWS`
- Pre-cleaning removes missing/empty cells and duplicate records.

---

## 6. System Architecture

```
User Input (Headline / Article / Tweet)
                   │
                   ▼
       Text Preprocessing Pipeline
   [Lowercase, URL & HTML Strip, Punctuation Removal, Whitespace Normalization]
                   │
                   ▼
          TF-IDF Vectorizer
     [Vocabulary Mapping + N-gram (1,2) Feature Weights]
                   │
                   ▼
       Trained Supervised Classifier
     [Logistic Regression / Multinomial Naive Bayes]
                   │
                   ▼
     Probability & Decision Threshold
                   │
                   ▼
      Prediction & Interactive UI
   [Verdict: REAL / FAKE | Confidence % | Token Cloud | Timestamp]
```

---

## 7. Machine Learning Methodology

### A. Preprocessing
Text undergoes systematic cleaning:
1. Converting text to lowercase.
2. Regular expression removal of URLs (`https?://\S+`).
3. Stripping HTML tags (`<.*?>`).
4. Stripping emails and mentions.
5. Stripping punctuation (`string.punctuation`) and isolated digits.
6. Normalizing whitespace.

### B. Feature Extraction: TF-IDF
$$TF(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total words in } d}$$
$$IDF(t) = \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
$$TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)$$
- `ngram_range=(1, 2)`: Captures single keywords as well as two-word phrases (e.g., "secret cure", "peer reviewed").
- `max_features=10000`: Restricts vocabulary size to the most informative tokens.
- `stop_words='english'`: Eliminates non-informative grammatical noise.

### C. Classification Algorithms
1. **Logistic Regression**: Linear classifier utilizing the Sigmoid function:
   $$P(y=1|X) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
2. **Multinomial Naive Bayes**: Probabilistic classifier based on Bayes' Theorem with Laplacian smoothing:
   $$P(c|d) \propto P(c) \prod_{i=1}^n P(w_i | c)$$

---

## 8. Installation & Setup (Local Windows / Linux / macOS)

### Prerequisites
- Python 3.10 or higher installed
- VS Code (or any preferred code editor)

### Step 1: Clone or Navigate to the Directory
```bash
cd fake-news-detection
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 9. How to Train the Model

Run the automated training script:
```bash
python train_model.py
```

### What Happens:
1. Loads `data/dataset.csv`.
2. Prints dataset size, missing values, and detected label distribution.
3. Splits into 80% Train and 20% Test sets with `random_state=42`.
4. Trains both **Logistic Regression** and **Multinomial Naive Bayes**.
5. Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
6. Automatically selects the model with the higher F1-Score.
7. Saves `models/model.pkl`, `models/vectorizer.pkl`, and `models/metadata.json`.

---

## 10. How to Run the Flask Web Application

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 11. How to Test
1. **Quick Test**: Click **Sample Real News** or **Sample Fake News** to populate the textarea.
2. Click **Analyze News**.
3. View:
   - Prediction badge (`REAL NEWS` in green or `FAKE NEWS` in red)
   - Confidence percentage & split probability
   - Model name used
   - Word count and character count
   - Analysis timestamp
   - Extracted vocabulary tokens
4. Test with custom news articles from BBC, Reuters, or sensational social media posts.

---

## 12. Model Evaluation Results

Empirical results obtained from the dataset:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **90.00%** | **88.89%** | **88.89%** | **88.89%** |
| **Multinomial Naive Bayes** | **85.00%** | **80.00%** | **88.89%** | **84.21%** |

- **True Positives (TP)**: Correctly identified Fake News.
- **True Negatives (TN)**: Correctly identified Real News.
- **False Positives (FP)**: Real news incorrectly flagged as Fake (Type I error).
- **False Negatives (FN)**: Fake news incorrectly flagged as Real (Type II error).

---

## 13. Limitations
- **Lexical Pattern Reliance**: Classifies based on linguistic style and keywords; does not cross-reference live web sources or fact-checking databases.
- **Domain Drift**: Emerging events or novel vocabulary not present in the training set may yield lower confidence.
- **Context Negation**: Simple n-grams may occasionally struggle with sophisticated irony, satire, or sarcasm.

## 14. Future Enhancements
- Fine-tune transformer architectures (BERT, RoBERTa) for deep semantic comprehension.
- Implement live URL scraping and Google Fact Check Tools API integration.
- Add multi-modal verification (analyzing attached images via reverse image search).
- Multi-lingual fake news detection for regional languages.

---

## 15. Free Deployment Instructions (Render.com)

1. Push your project folder to **GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Fake News Detection Mini Project"
   git branch -M main
   git remote add origin https://github.com/<your-username>/fake-news-detection.git
   git push -u origin main
   ```
2. Go to [https://render.com](https://render.com) and create a free account.
3. Click **New +** &rarr; **Web Service**.
4. Connect your GitHub repository.
5. Configure:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && python train_model.py`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free`
6. Click **Deploy Web Service**. Within 2 minutes, your live public URL is online!
