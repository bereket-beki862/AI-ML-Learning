
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Student data: study hours, attendance percentage
X = np.array([
    [1, 40], [2, 45], [2, 55], [3, 50],
    [3, 60], [4, 55], [4, 65], [5, 60],
    [5, 70], [6, 65], [6, 75], [7, 70],
    [7, 80], [8, 75], [8, 85], [9, 90],
    [2, 40], [3, 45], [6, 80], [9, 95]
])

# 0 = Fail, 1 = Pass (practice labels)
y = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 0, 0, 1, 1
])

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Scale features because KNN uses distances
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the KNN classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# 5. Predict test labels
predictions = model.predict(X_test_scaled)

print("KNN CLASSIFICATION")
print("------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 6. Predict for a new student
new_student = np.array([[6, 75]])
new_student_scaled = scaler.transform(new_student)

prediction = model.predict(new_student_scaled)[0]
probability = model.predict_proba(new_student_scaled)[0][1]

print("\nNEW STUDENT")
print("Study hours:", new_student[0][0])
print("Attendance:", new_student[0][1], "%")
print("Prediction:", "Pass" if prediction == 1 else "Fail")
print("Estimated pass probability:", round(probability * 100, 2), "%")
