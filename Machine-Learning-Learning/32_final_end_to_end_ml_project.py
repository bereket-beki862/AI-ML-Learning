
# FINAL END-TO-END MACHINE LEARNING PROJECT
# Wine Classification

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==================================
# 1. LOAD DATASET
# ==================================

wine = load_wine()

X = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

y = wine.target

print("WINE CLASSIFICATION PROJECT")
print("============================")

print("\nDataset Shape:", X.shape)

print("\nFirst Five Rows:")
print(X.head())

print("\nMissing Values:")
print(X.isnull().sum().sum())

print("\nWine Classes:")
print(wine.target_names)

# ==================================
# 2. EXPLORATORY DATA ANALYSIS
# ==================================

print("\nDataset Statistics:")
print(X.describe().round(2))

plt.figure(figsize=(10, 6))

X["alcohol"].hist(bins=15)

plt.title("Distribution of Alcohol Content")
plt.xlabel("Alcohol")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()

# ==================================
# 3. TRAIN TEST SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==================================
# 4. CREATE MACHINE LEARNING MODELS
# ==================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])
}

# ==================================
# 5. TRAIN AND COMPARE MODELS
# ==================================

results = []
trained_models = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    results.append({
        "Model": name,
        "Test Accuracy": accuracy,
        "CV Average": cv_scores.mean()
    })

    trained_models[name] = model

# ==================================
# 6. DISPLAY COMPARISON
# ==================================

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print("================")

print(results_df.round(3).to_string(index=False))

# ==================================
# 7. SELECT BEST MODEL
# ==================================

best_index = results_df["CV Average"].idxmax()

best_name = results_df.loc[best_index, "Model"]

best_model = trained_models[best_name]

print("\nBest Model:", best_name)

# ==================================
# 8. FINAL MODEL EVALUATION
# ==================================

final_predictions = best_model.predict(X_test)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        final_predictions,
        target_names=wine.target_names
    )
)

# ==================================
# 9. CONFUSION MATRIX
# ==================================

cm = confusion_matrix(y_test, final_predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=wine.target_names
)

display.plot(cmap="Blues")

plt.title("Wine Classification Confusion Matrix")
plt.show()

# ==================================
# 10. SAVE MODEL AND RESULTS
# ==================================

joblib.dump(best_model, "best_wine_model.pkl")

results_df.to_csv(
    "wine_model_comparison.csv",
    index=False
)

print("\nModel saved successfully!")
print("Comparison results saved successfully!")

# ==================================
# 11. TEST A NEW SAMPLE
# ==================================

new_sample = X.iloc[[0]]

prediction = best_model.predict(new_sample)

print("\nNEW WINE PREDICTION")
print("===================")

print("Predicted Wine Class:", wine.target_names[prediction[0]])

print("\nPROJECT COMPLETED SUCCESSFULLY!")
