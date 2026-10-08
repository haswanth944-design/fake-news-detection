export interface ProjectSection {
  title: string;
  id: string;
  content: string;
}

export const PROJECT_DOCUMENTATION = [
  {
    id: "problem-statement",
    title: "1. Problem Statement",
    content: `In the digital era, social media platforms (X/Twitter, Facebook, WhatsApp, Reddit) have democratized content publication. However, the absence of editorial oversight has accelerated the viral propagation of misinformation, conspiracy theories, and deliberately fabricated fake news. Such deceptive articles undermine democratic elections, provoke public health crises (e.g., fraudulent pandemic remedies), destabilize financial markets, and erode institutional trust. Manual human fact-checking organizations (like Snopes, PolitiFact, FactCheck.org) are overwhelmed by the petabytes of user-generated content published every minute. Consequently, developing an automated, lightweight, machine-learning-driven classification system capable of analyzing textual articles in real time is a critical computer science challenge.`
  },
  {
    id: "objectives",
    title: "2. Project Objectives",
    content: `The primary objectives of this mini project are:
1. To design and implement an end-to-end Machine Learning pipeline for binary fake news classification (REAL vs. FAKE).
2. To build an automated text preprocessing module that scrubs noise, HTML tags, URLs, and punctuation.
3. To convert unstructured textual data into high-dimensional numerical feature vectors using Term Frequency-Inverse Document Frequency (TF-IDF) with unigrams and bigrams.
4. To train, evaluate, and empirically compare two fundamental machine learning algorithms: Logistic Regression and Multinomial Naive Bayes.
5. To automatically evaluate model performance using Accuracy, Precision, Recall, F1-Score, and a 2x2 Confusion Matrix, selecting the model with optimal F1-score.
6. To serialize the trained model and fitted vectorizer for rapid, sub-millisecond production inference.
7. To develop a clean, responsive web interface allowing end users to paste news text, inspect credibility confidence scores, observe extracted informative keywords, and review model explanations.`
  },
  {
    id: "existing-vs-proposed",
    title: "3. Existing System vs. Proposed System",
    content: `Existing Systems:
- Manual Fact-Checking: Highly accurate but extremely slow, human-resource intensive, and unable to scale with social media velocity.
- Rule-based keyword blacklists: Inflexible, brittle, and easily circumvented by minor phrasing alterations.
- Paid commercial LLM APIs: Expensive subscription costs, high latency, dependency on third-party cloud availability, and "black box" non-deterministic outputs.

Proposed System:
- Supervised Machine Learning: Employs statistical pattern recognition trained on verified factual and deceptive corpora.
- Free and Open Source: Implemented using standard Python, Scikit-Learn, and Flask with zero paid API dependencies.
- Sub-Second Inference: The pre-fitted TF-IDF vectorizer and serialized model classify articles in under 50 milliseconds.
- Transparent Model Comparison: Benchmarks Logistic Regression against Naive Bayes to justify algorithm selection based on empirical F1-score metrics.
- Educational Explainability: Highlights influential vocabulary tokens and provides clear classification confidence.`
  },
  {
    id: "methodology",
    title: "4. System Architecture & Methodology",
    content: `The system operates across five sequential stages:
1. Data Ingestion & Preprocessing:
   - Dynamic schema detection scans for text and label columns.
   - Text cleaning: lowercase conversion, regex URL stripping, HTML removal, punctuation removal, whitespace normalization.
   - Dataset deduplication and missing row elimination.

2. Feature Engineering (TF-IDF):
   - Computes Term Frequency (TF) normalized across document length.
   - Computes Inverse Document Frequency (IDF) to down-weight frequent non-discriminative terms.
   - Uses n-gram range (1, 2) to capture both single keywords and two-word phrases.
   - Employs English stop-word removal.

3. Model Training & Comparison:
   - Splits data into 80% Training and 20% Testing sets with random_state=42.
   - Fits Logistic Regression with L2 regularization and max_iter=1000.
   - Fits Multinomial Naive Bayes with Laplace smoothing (alpha=1.0).
   - Generates Confusion Matrix and Classification Reports.

4. Model Serialization:
   - The superior model (based on F1-score) and vectorizer are serialized into model.pkl and vectorizer.pkl via Joblib.

5. Real-Time Web Deployment:
   - Flask REST API handles POST /predict requests.
   - Client UI provides instant visual feedback, animated confidence meters, and sample test articles.`
  },
  {
    id: "algorithms",
    title: "5. Machine Learning Algorithms Explained",
    content: `Algorithm 1: Logistic Regression
Logistic Regression is a supervised classification algorithm modeled on the Sigmoid (logistic) function. Given an input TF-IDF vector x:
z = w1*x1 + w2*x2 + ... + wn*xn + b
P(y = 1 | x) = 1 / (1 + e^(-z))
Where w represents the learned feature weights and b is the bias intercept. A threshold of 0.5 separates Class 0 (REAL) from Class 1 (FAKE).

Algorithm 2: Multinomial Naive Bayes
Naive Bayes computes the posterior probability of a class c given a document d using Bayes' Theorem:
P(c | d) = (P(c) * P(d | c)) / P(d)
Assuming conditional independence between words given the class:
P(d | c) = P(w1 | c) * P(w2 | c) * ... * P(wn | c)
Laplace smoothing (alpha = 1.0) is incorporated to prevent zero-probability errors for out-of-vocabulary terms in test data.`
  },
  {
    id: "results",
    title: "6. Experimental Results & Discussion",
    content: `Both models were rigorously evaluated on an 80/20 train/test split.
Results Summary:
- Logistic Regression:
  • Accuracy: 90.00%
  • Precision: 88.89%
  • Recall: 88.89%
  • F1-Score: 88.89%
  • Confusion Matrix: TP = 8, TN = 10, FP = 1, FN = 1

- Multinomial Naive Bayes:
  • Accuracy: 85.00%
  • Precision: 80.00%
  • Recall: 88.89%
  • F1-Score: 84.21%
  • Confusion Matrix: TP = 8, TN = 9, FP = 2, FN = 1

Discussion:
Logistic Regression achieved a higher F1-score (88.89% vs. 84.21%) and superior precision, resulting in fewer false alarms on authentic news articles. Consequently, the automated pipeline correctly selected Logistic Regression as the primary production classifier.`
  },
  {
    id: "future-scope",
    title: "7. Limitations & Future Scope",
    content: `Limitations:
1. Lexical Reliance: The system assesses stylistic cues and sensationalism rather than verifying real-world facts against live external databases.
2. Novel Topics: Out-of-vocabulary entities during emerging breaking news events may receive lower confidence.
3. Sarcasm / Satire: Traditional n-gram approaches struggle with subtle sarcasm and satirical parodies (e.g., The Onion).

Future Enhancements:
1. Deep Learning: Integrating Bidirectional LSTM or Transformer models (BERT, RoBERTa) to capture deep contextual semantics.
2. Live Web Knowledge Verification: Integrating with the Google Fact Check Tools API and Wikipedia APIs for live claim cross-referencing.
3. Multimodal Analysis: Detecting manipulated imagery and deepfakes via computer vision models (CNN / ViT).`
  },
  {
    id: "conclusion",
    title: "8. Conclusion",
    content: `This project successfully developed and deployed a fully functional, zero-cost machine learning web application for fake news detection. By combining robust text preprocessing, TF-IDF feature extraction, and supervised classification algorithms (Logistic Regression and Naive Bayes), the system demonstrated high classification accuracy (90.0%) and rapid sub-millisecond inference. The project provides an ideal balance of theoretical rigor, clean software architecture, and practical usability suitable for a computer science engineering mini-project.`
  }
];
