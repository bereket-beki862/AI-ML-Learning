
# Project 3: SMS Spam Detection
# Step 5: Naive Bayes Classification Model

import joblib

from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# File paths
DATA_PATH = BASE_DIR / "models" / "tfidf_data.pkl"
MODEL_PATH = BASE_DIR / "models" / "naive_bayes_model.pkl"

# Load TF-IDF data
data = joblib.load(DATA_PATH)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("TF-IDF data loaded successfully!")

# Create Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

print("\nNaive Bayes model trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("-------------------------")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Saved location:", MODEL_PATH)
