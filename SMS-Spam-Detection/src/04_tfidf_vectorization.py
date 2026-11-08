
# Project 3: SMS Spam Detection
# Step 4: TF-IDF Feature Extraction

import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# File paths
DATA_PATH = BASE_DIR / "data" / "cleaned_sms.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

print("Cleaned dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Prepare features and labels
X = df["clean_message"].fillna("")
y = df["label"].map({"ham": 0, "spam": 1})

if y.isna().any():
    raise ValueError("Unexpected labels found in the dataset.")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

# Learn vocabulary from training data only
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform testing data
X_test_tfidf = vectorizer.transform(X_test)

# Save vectorizer
joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

# Save processed training and testing data
joblib.dump(
    {
        "X_train": X_train_tfidf,
        "X_test": X_test_tfidf,
        "y_train": y_train,
        "y_test": y_test
    },
    MODEL_DIR / "tfidf_data.pkl"
)

print("\nTF-IDF feature extraction completed!")

print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)

print("\nNumber of vocabulary features:", len(vectorizer.vocabulary_))

print("\nFiles saved successfully in:", MODEL_DIR)
