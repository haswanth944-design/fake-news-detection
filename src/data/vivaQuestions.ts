export interface VivaQuestion {
  id: number;
  category: string;
  question: string;
  answer: string;
  formula?: string;
  keyTakeaway: string;
}

export const VIVA_QUESTIONS: VivaQuestion[] = [
  {
    id: 1,
    category: "Fundamentals",
    question: "What is Fake News?",
    answer: "Fake news is false, misleading, or fabricated information presented as legitimate news. It is typically spread on social media to deceive readers, generate revenue through sensational clickbait, or manipulate public opinion.",
    keyTakeaway: "Disinformation designed to deceive and mimic journalistic style."
  },
  {
    id: 2,
    category: "Fundamentals",
    question: "What is Machine Learning and why is it used here?",
    answer: "Machine Learning is a subset of Artificial Intelligence where algorithms learn statistical patterns from historical data to make predictions on unseen data without being explicitly programmed. In our project, it automatically identifies subtle linguistic patterns that differentiate real news from fake news.",
    keyTakeaway: "Data-driven statistical learning instead of hardcoded rules."
  },
  {
    id: 3,
    category: "Fundamentals",
    question: "Why is Fake News Detection treated as a Classification Problem?",
    answer: "Because our target outcome is discrete and categorical: a given news article belongs to one of two distinct predefined classes: 'REAL' (Class 0) or 'FAKE' (Class 1). This makes it a classic binary text classification task.",
    keyTakeaway: "Binary supervised classification (REAL vs FAKE)."
  },
  {
    id: 4,
    category: "NLP & Feature Extraction",
    question: "What is TF-IDF and how does it work?",
    answer: "TF-IDF stands for Term Frequency-Inverse Document Frequency. Term Frequency (TF) counts how often a word appears in a specific document. Inverse Document Frequency (IDF) penalizes common words (like 'the', 'is') and elevates unique, discriminative words across the whole corpus.",
    formula: "TF-IDF(t, d) = TF(t, d) × log((1 + N) / (1 + DF(t)))",
    keyTakeaway: "Transforms raw text into a numerical matrix highlighting informative words."
  },
  {
    id: 5,
    category: "NLP & Feature Extraction",
    question: "Why use N-Grams (1, 2) in TF-IDF?",
    answer: "An n-gram of range (1, 2) extracts both unigrams (single words like 'cure', 'cancer') and bigrams (two adjacent words like 'secret cure', 'peer reviewed'). Bigrams preserve vital local word order and phrase meaning that single words lose.",
    keyTakeaway: "Captures context like 'not guilty' vs 'guilty'."
  },
  {
    id: 6,
    category: "NLP & Feature Extraction",
    question: "What text preprocessing steps did you perform and why?",
    answer: "We converted text to lowercase, stripped URLs, removed HTML tags, cleared emails, removed punctuation and isolated digits, and collapsed multiple whitespaces. This standardizes text and eliminates noisy characters that would dilute the vocabulary.",
    keyTakeaway: "Noise reduction and vocabulary normalization."
  },
  {
    id: 7,
    category: "Machine Learning Models",
    question: "What is Logistic Regression and how does it classify text?",
    answer: "Logistic Regression is a linear classification algorithm. It computes a linear weighted sum of the input TF-IDF features (z = w·x + b) and passes it through the non-linear Sigmoid function, mapping the output into a probability between 0 and 1. If probability ≥ 0.5, it predicts FAKE; otherwise REAL.",
    formula: "P(y = 1 | x) = 1 / (1 + e^-(w·x + b))",
    keyTakeaway: "Linear decision boundary mapped to probability via Sigmoid."
  },
  {
    id: 8,
    category: "Machine Learning Models",
    question: "What is Multinomial Naive Bayes and what is its 'naive' assumption?",
    answer: "Multinomial Naive Bayes is a probabilistic classifier based on Bayes' Theorem. It is 'naive' because it makes the strong assumption that all words in a document are conditionally independent of each other given the class label. Despite this simplification, it performs exceptionally well on text classification.",
    formula: "P(c | d) ∝ P(c) × ∏ P(w_i | c)",
    keyTakeaway: "Fast, probabilistic classifier assuming conditional feature independence."
  },
  {
    id: 9,
    category: "Machine Learning Models",
    question: "Why train and compare both Logistic Regression and Naive Bayes?",
    answer: "Training both models allows empirical benchmarking. Logistic Regression is discriminative (learns P(y|x) directly), while Naive Bayes is generative (models P(x|y) and P(y)). Comparing both demonstrates rigorous engineering, and we automatically deploy the one with the superior F1-score.",
    keyTakeaway: "Compares discriminative vs generative approaches to pick the optimal model."
  },
  {
    id: 10,
    category: "Dataset & Validation",
    question: "What is Train/Test Split and why use 80/20?",
    answer: "Train/test split divides the labeled dataset into two independent subsets: 80% for training the model and 20% held out strictly for testing. This simulates real-world unseen data, verifying that the model generalizes rather than merely memorizing training examples.",
    keyTakeaway: "Prevents data leakage and validates true generalization."
  },
  {
    id: 11,
    category: "Dataset & Validation",
    question: "What is Overfitting and how do we prevent it?",
    answer: "Overfitting occurs when a model learns the training data and noise too well, achieving high training accuracy but failing on new, unseen test data. We prevent it using L2 regularization (Ridge penalty in Logistic Regression), limiting TF-IDF max_features to 10,000, and validating on held-out test splits.",
    keyTakeaway: "High train accuracy with poor test performance; prevented via regularization."
  },
  {
    id: 12,
    category: "Evaluation Metrics",
    question: "What is Accuracy and why is accuracy alone not enough?",
    answer: "Accuracy is the ratio of correct predictions to total predictions. However, if a dataset is imbalanced (e.g., 95% Real and 5% Fake), a naive model predicting 'Real' every time gets 95% accuracy while failing completely at detecting fake news. Hence, Precision, Recall, and F1-score are essential.",
    formula: "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
    keyTakeaway: "Overall correctness percentage; misleading on imbalanced datasets."
  },
  {
    id: 13,
    category: "Evaluation Metrics",
    question: "What is Precision?",
    answer: "Precision measures the reliability of positive predictions. Out of all news articles that our model predicted were FAKE, what percentage was actually FAKE? High precision means minimal False Positives (real news being falsely labeled as fake).",
    formula: "Precision = TP / (TP + FP)",
    keyTakeaway: "Accuracy of positive claims (minimizes false alarms)."
  },
  {
    id: 14,
    category: "Evaluation Metrics",
    question: "What is Recall (Sensitivity)?",
    answer: "Recall measures the model's ability to catch all actual positive instances. Out of all truly FAKE news articles in the dataset, what percentage did our model successfully catch? High recall means minimal False Negatives (fake news slipping through unnoticed).",
    formula: "Recall = TP / (TP + FN)",
    keyTakeaway: "Coverage of true positives (minimizes missed fake news)."
  },
  {
    id: 15,
    category: "Evaluation Metrics",
    question: "What is the F1-Score and why is it preferred for model selection?",
    answer: "F1-Score is the harmonic mean of Precision and Recall. Unlike the arithmetic mean, the harmonic mean penalizes extreme imbalances between Precision and Recall. Selecting the model with the highest F1-score ensures a balanced trade-off between avoiding false alarms and catching deceptive news.",
    formula: "F1 = 2 × (Precision × Recall) / (Precision + Recall)",
    keyTakeaway: "Balanced harmonic metric combining precision and recall."
  },
  {
    id: 16,
    category: "Evaluation Metrics",
    question: "What is a Confusion Matrix?",
    answer: "A Confusion Matrix is a 2x2 contingency table summarizing model classification performance: True Positives (Fake correctly predicted Fake), True Negatives (Real correctly predicted Real), False Positives (Real incorrectly predicted Fake), and False Negatives (Fake incorrectly predicted Real).",
    keyTakeaway: "Tabular breakdown of true vs predicted classes."
  },
  {
    id: 17,
    category: "Software Architecture",
    question: "Why do we save 'model.pkl' and 'vectorizer.pkl'?",
    answer: "We serialize the trained model and fitted vectorizer using Joblib/Pickle so they can be loaded instantly into production by the Flask server. This decouples the time-consuming training phase from runtime inference, allowing instantaneous prediction in milliseconds without retraining on every user request.",
    keyTakeaway: "Decouples offline model training from real-time production inference."
  },
  {
    id: 18,
    category: "Software Architecture",
    question: "Why MUST the saved vectorizer.pkl be used during prediction rather than fitting a new one?",
    answer: "The vectorizer defines the exact vocabulary indices and IDF weights learned during training. If you fit a new vectorizer on user input, the feature indices would not match the model's weight coefficients, causing catastrophic prediction errors.",
    keyTakeaway: "Feature dimension and token indices must strictly match the trained model."
  },
  {
    id: 19,
    category: "Software Architecture",
    question: "Why did you choose Flask for the backend?",
    answer: "Flask is a lightweight, Python-native micro-framework with zero boilerplate. It is fast, easy to learn, seamlessly integrates with Python ML libraries (Scikit-Learn, Pandas, NumPy), and offers clean REST API routing.",
    keyTakeaway: "Lightweight, Python-native web framework ideal for ML deployments."
  },
  {
    id: 20,
    category: "Software Architecture",
    question: "How does the frontend communicate with the Flask backend?",
    answer: "The frontend uses asynchronous JavaScript (Fetch API / AJAX) to send a POST request with the user's news text formatted as JSON to the '/predict' endpoint. Flask processes the text, executes model inference, and returns JSON. The frontend updates the DOM dynamically without page reload.",
    keyTakeaway: "Asynchronous REST API communication using JSON payloads."
  },
  {
    id: 21,
    category: "Dataset Handling",
    question: "How does your code handle diverse dataset column names and label formats?",
    answer: "Our pipeline dynamically scans columns for names like 'title', 'text', 'content', 'label', 'class', combining title and content if both exist. Crucially, it inspects label values rather than assuming 0 or 1 is arbitrarily fake, building a safe verified mapping dictionary.",
    keyTakeaway: "Flexible column detection and adaptive label semantic mapping."
  },
  {
    id: 22,
    category: "Limitations & Ethics",
    question: "What are the limitations of this machine learning system?",
    answer: "1. Lexical dependency: It relies on linguistic style, sensationalism, and vocabulary, rather than verifying factual ground truth. 2. Out-of-vocabulary terms: Unprecedented breaking events with new names may yield lower confidence. 3. Sarcasm and satire can sometimes mislead bag-of-words models.",
    keyTakeaway: "Analyzes linguistic patterns, not real-time world facts."
  },
  {
    id: 23,
    category: "Limitations & Ethics",
    question: "Why is the disclaimer in the user interface important?",
    answer: "Ethical AI practice requires transparency. Machine learning models provide probabilistic assessments based on historical training data, not certified factual verification. The disclaimer prevents users from treating the classification as infallible truth.",
    keyTakeaway: "Promotes responsible AI and sets appropriate user expectations."
  },
  {
    id: 24,
    category: "Advanced Extensions",
    question: "How can this project be enhanced using Deep Learning (LSTM / BiLSTM)?",
    answer: "Long Short-Term Memory (LSTM) and Bidirectional LSTM networks process sequential word embeddings (Word2Vec / GloVe). They maintain recurrent memory gates over long texts, capturing word order and long-range semantic dependencies that TF-IDF n-grams miss.",
    keyTakeaway: "Captures sequential word ordering and long-term context."
  },
  {
    id: 25,
    category: "Advanced Extensions",
    question: "How can Transformers (BERT / RoBERTa) improve Fake News Detection?",
    answer: "BERT uses bidirectional self-attention mechanisms pre-trained on billions of words. It understands deep context, syntax, and nuanced semantic meaning (e.g. knowing whether 'bank' refers to a river or a financial institution). Fine-tuning BERT on fake news datasets achieves state-of-the-art accuracy exceeding 95%.",
    keyTakeaway: "Bidirectional self-attention captures deep contextual semantics."
  },
  {
    id: 26,
    category: "Advanced Extensions",
    question: "How does adding Web Browsing & Search Grounding solve the limitation of offline ML models?",
    answer: "While an offline ML model evaluates stylometry (detecting whether text 'sounds' sensational or journalistic based on past training data), it cannot verify brand-new breaking news. Integrating live web browsing (via Google Search grounding) enables the system to actively search current internet sources, cross-referencing claims against reputable wire agencies (Reuters, AP, BBC) and fact-check registries (Snopes, PolitiFact) to verify ground-truth facts in real time.",
    keyTakeaway: "Combines linguistic pattern detection with real-time web fact verification."
  }
];
