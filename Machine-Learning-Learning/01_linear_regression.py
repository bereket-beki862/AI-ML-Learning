
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Training data
# X = study hours, y = student marks
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([35, 40, 50, 55, 65, 70, 80, 85])

# 2. Create the machine learning model
model = LinearRegression()

# 3. Train the model
model.fit(X, y)

# 4. Predict marks for new study hours
new_hours = np.array([[3], [5], [9]])
predictions = model.predict(new_hours)

print("STUDENT MARKS PREDICTION")
print("------------------------")

for i in range(len(new_hours)):
    print(
        "Study hours:", new_hours[i][0],
        "| Predicted marks:", round(predictions[i], 2)
    )

# 5. Evaluate the model on its training data
training_predictions = model.predict(X)

mae = mean_absolute_error(y, training_predictions)
r2 = r2_score(y, training_predictions)

print("\nMODEL EVALUATION")
print("Mean Absolute Error:", round(mae, 2))
print("R-squared score:", round(r2, 3))

# 6. Display the learned equation
print("\nMODEL EQUATION")
print("Slope:", round(model.coef_[0], 2))
print("Intercept:", round(model.intercept_, 2))
