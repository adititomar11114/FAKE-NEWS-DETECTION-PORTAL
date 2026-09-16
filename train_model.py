import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Dataset location
DATASET_PATH = "dataset/news.csv"


print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# Check required columns
required_columns = ["text", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' is missing from dataset."
        )


# Remove missing values
df = df.dropna(subset=["text", "label"])


# Convert to strings
df["text"] = df["text"].astype(str)

df["label"] = (
    df["label"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# Keep only FAKE and REAL
df = df[df["label"].isin(["FAKE", "REAL"])]


print("\nLabel distribution:")
print(df["label"].value_counts())


# Features and target
X = df["text"]
y = df["label"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)


X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# Machine Learning model
model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)


# Prediction
y_pred = model.predict(X_test_tfidf)


# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# Create model folder
os.makedirs(
    "model",
    exist_ok=True
)


# Save model
joblib.dump(
    model,
    "model/fake_news_model.pkl"
)


# Save vectorizer
joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)


print("\nModel saved successfully.")

print(
    "model/fake_news_model.pkl"
)

print(
    "model/tfidf_vectorizer.pkl"
)
