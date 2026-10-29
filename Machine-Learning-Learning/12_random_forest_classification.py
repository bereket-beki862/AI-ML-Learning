
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Features: [study hours, attendance percentage]
X = np.array([
    [1, 40],
    [2, 45],
    [2, 50],
    [3, 55],
    [4, 60],
    [5, 65],
    [6, 70],
    [7, 75],
    [8, 85],
    [9, 90],
    [3, 40],
    [6, 80],
    [7, 60],
    [8, 70],
    [4, 50]
])

# 0 = Fail, 1 = Pass
y = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    0, 1, 1, 1, 0
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("Actual results:", y_test)
print("Predicted results:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict for a new student
new_student = np.array([[6, 70]])
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("The student is predicted to PASS.")
else:
    print("The student is predicted to FAIL.")
