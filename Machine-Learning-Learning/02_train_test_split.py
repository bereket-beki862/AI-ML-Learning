
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Prepare student data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([30, 35, 45, 50, 60, 65, 75, 80, 90, 95])

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("TRAINING AND TESTING DATA")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict marks using the test data
predictions = model.predict(X_test)

print("\nTEST RESULTS")
for i in range(len(X_test)):
    print(
        "Study hours:", X_test[i][0],
        "| Actual marks:", y_test[i],
        "| Predicted marks:", round(predictions[i], 2)
    )

# 5. Evaluate predictions on unseen test data
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("Mean Absolute Error:", round(mae, 2))
print("R-squared score:", round(r2, 3))
