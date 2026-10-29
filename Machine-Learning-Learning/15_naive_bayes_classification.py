
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Features: [study hours, attendance percentage]
X = np.array([
    [1, 35], [2, 40], [2, 45], [3, 50], [3, 55],
    [4, 50], [4, 60], [5, 55], [5, 65], [6, 60],
    [6, 70], [7, 65], [7, 75], [8, 80], [9, 90],
    [2, 35], [4, 45], [6, 80], [8, 70], [9, 85]
])

# 0 = Fail, 1 = Pass
y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 1, 0, 0,
    1, 1, 1, 1, 1,
    0, 0, 1, 1, 1
])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Create and train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Actual results:", y_test)
print("Predicted results:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict for a new student
new_student = np.array([[6, 75]])

prediction = model.predict(new_student)
probabilities = model.predict_proba(new_student)

if prediction[0] == 1:
    print("Predicted result: PASS")
else:
    print("Predicted result: FAIL")

print("Probability of Fail:", round(probabilities[0][0], 3))
print("Probability of Pass:", round(probabilities[0][1], 3))
