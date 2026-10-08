"""
Exploratory Data Analysis (EDA) & Model Prototyping Script
File: notebooks/fake_news_analysis.py
Can be executed in VS Code or converted to Jupyter Notebook.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

print("1. Loading dataset...")
df = pd.read_csv("../data/dataset.csv")
print(f"Total rows: {len(df)}")
print(df.head())

print("\n2. Class balance:")
print(df['label'].value_counts())

print("\n3. Text length statistics:")
df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
print(df.groupby('label')[['char_count', 'word_count']].mean())

print("\n4. Feature extraction & TF-IDF modeling:")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(df['text'])
y = df['label'].map({'REAL': 0, 'FAKE': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("\n5. Logistic Regression Report:")
print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
