
# Cross-Validation using K-Fold

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

# 1. Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Create machine learning pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=200))
])

# 3. Create K-Fold cross-validation
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 4. Perform cross-validation
scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

# 5. Display individual scores
print("Cross-Validation Scores:")

for i, score in enumerate(scores):
    print(f"Fold {i + 1}: {score * 100:.2f}%")

# 6. Calculate average accuracy
average_accuracy = np.mean(scores)

print("\nAverage Accuracy:")
print(round(average_accuracy * 100, 2), "%")

# 7. Calculate standard deviation
print("\nStandard Deviation:")
print(round(np.std(scores) * 100, 2), "%")
