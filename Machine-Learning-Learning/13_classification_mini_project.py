
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Features: study hours and attendance percentage
X = np.array([
    [1, 35], [2, 40], [2, 45], [3, 50], [3, 55],
    [4, 50], [4, 60], [5, 55], [5, 65], [6, 60],
    [6, 70], [7, 65], [7, 75], [8, 80], [9, 90],
    [2, 35], [4, 45], [6, 80], [8, 70], [9, 85]
])

# 0 = Fail, 1 = Pass (practice labels)
y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 0, 1, 0,
    1, 1, 1, 1, 1,
    0, 0, 1, 1, 1
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Create four classification models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression()
    ),
    "KNN": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=3)
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=3, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

# Train and compare models
print("CLASSIFICATION MODEL COMPARISON")
print("-" * 40)

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"{name}: {accuracy:.2%}")

# Predict for a new student using Random Forest
new_student = np.array([[6, 75]])
prediction = models["Random Forest"].predict(new_student)

print("\nNEW STUDENT PREDICTION")
if prediction[0] == 1:
    print("Predicted result: PASS")
else:
    print("Predicted result: FAIL")
