
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Student data
# Columns: study hours, attendance percentage
X = np.array([
    [1, 40], [2, 45], [2, 55], [3, 50],
    [3, 60], [4, 55], [4, 65], [5, 60],
    [5, 70], [6, 65], [6, 75], [7, 70],
    [7, 80], [8, 75], [8, 85], [9, 90],
    [2, 40], [3, 45], [6, 80], [9, 95]
])

# 0 = Fail, 1 = Pass (example labels)
y = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 0, 0, 1, 1
])

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Train the classification model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Make predictions
predictions = model.predict(X_test)

print("STUDENT PASS/FAIL PREDICTION")
print("----------------------------")

for i in range(len(X_test)):
    result = "Pass" if predictions[i] == 1 else "Fail"
    print(
        f"Study hours: {X_test[i][0]}, "
        f"Attendance: {X_test[i][1]}%, "
        f"Actual: {y_test[i]}, Predicted: {result}"
    )

# 5. Evaluate the model
print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    labels=[0, 1],
    target_names=["Fail", "Pass"],
    zero_division=0
))

# 6. Predict for a new student
new_student = np.array([[6, 75]])
prediction = model.predict(new_student)[0]
probability = model.predict_proba(new_student)[0][1]

print("\nNEW STUDENT")
print("Predicted result:", "Pass" if prediction == 1 else "Fail")
print("Estimated pass probability:", round(probability * 100, 2), "%")
