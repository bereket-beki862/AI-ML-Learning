
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Create the dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    2, 4, 6, 8, 3, 7, 9, 5, 1, 10],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
                   60, 70, 78, 88, 62, 82, 92, 72, 52, 98],
    "Marks": [30, 35, 42, 48, 58, 65, 73, 82, 90, 96,
              38, 52, 68, 85, 45, 78, 92, 60, 32, 98]
}

df = pd.DataFrame(data)

print("STUDENT DATA")
print(df.head())

# 2. Prepare input features and target
X = df[["Study_Hours", "Attendance"]]
y = df["Marks"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions on test data
predictions = model.predict(X_test)

results = pd.DataFrame({
    "Actual_Marks": y_test.to_numpy(),
    "Predicted_Marks": np.round(predictions, 2)
})

print("\nTEST PREDICTIONS")
print(results.to_string(index=False))

# 6. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R-squared:", round(r2, 3))

# 7. Predict for a new student
new_student = pd.DataFrame({
    "Study_Hours": [6],
    "Attendance": [80]
})

new_prediction = model.predict(new_student)[0]

print("\nNEW STUDENT PREDICTION")
print("Study hours: 6")
print("Attendance: 80%")
print("Predicted marks:", round(new_prediction, 2))

# 8. Plot actual versus predicted marks
plt.figure(figsize=(7, 5))
plt.scatter(y_test, predictions)

# Reference line: perfect predictions
min_mark = min(y_test.min(), predictions.min())
max_mark = max(y_test.max(), predictions.max())

plt.plot(
    [min_mark, max_mark],
    [min_mark, max_mark],
    linestyle="--"
)

plt.title("Actual vs Predicted Student Marks")
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.grid(True)
plt.tight_layout()
plt.show()
