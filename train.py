# EMAIL SPAM CLASSIFIER - TRAIN

import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")

# LOAD DATA
df = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1","v2"]]
df.columns = ["label","message"]

print("Dataset shape:", df.shape)

# CONVERT LABEL TO NUMBER
df["label"] = df["label"].map({"ham":0, "spam":1})

# CLEAN TEXT FUNCTION
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

df["message"] = df["message"].apply(clean_text)

print("Text cleaned!")

# TF-IDF VECTORIZATION
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["message"])
y = df["label"]

print("Text vectorized!")

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# TRAIN MODEL
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model trained!")

# PREDICT
y_pred = model.predict(X_test)

# EVALUATE
accuracy = accuracy_score(y_test, y_pred)
print("\n===== MODEL EVALUATION =====")
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# SAVE MODEL + VECTORIZER
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved as spam_model.pkl")
print("Vectorizer saved as tfidf_vectorizer.pkl")
