
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Training data
# Columns: study hours, attendance percentage, previous exam marks
X = np.array([
    [1, 60, 40],
    [2, 65, 45],
    [3, 70, 50],
    [4, 75, 55],
    [5, 80, 65],
    [6, 85, 70],
    [7, 90, 80],
    [8, 95, 85]
])

# Final exam marks
y = np.array([35, 40, 48, 55, 65, 72, 82, 90])

# 2. Create the model
model = LinearRegression()

# 3. Train the model
model.fit(X, y)

# 4. Predict marks for a new student
# Study hours = 5, attendance = 82%, previous marks = 68
new_student = np.array([[5, 82, 68]])
predicted_marks = model.predict(new_student)

print("MULTIPLE LINEAR REGRESSION")
print("--------------------------")
print("Study hours:", new_student[0][0])
print("Attendance:", new_student[0][1], "%")
print("Previous marks:", new_student[0][2])
print("Predicted final marks:", round(predicted_marks[0], 2))

# 5. Evaluate the model on training data
training_predictions = model.predict(X)

mae = mean_absolute_error(y, training_predictions)
r2 = r2_score(y, training_predictions)

print("\nMODEL EVALUATION")
print("Mean Absolute Error:", round(mae, 2))
print("R-squared score:", round(r2, 3))

# 6. Display the learned coefficients
print("\nFEATURE COEFFICIENTS")
print("Study hours:", round(model.coef_[0], 2))
print("Attendance:", round(model.coef_[1], 2))
print("Previous marks:", round(model.coef_[2], 2))
print("Intercept:", round(model.intercept_, 2))
