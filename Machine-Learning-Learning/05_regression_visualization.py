
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Prepare training data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([35, 40, 50, 55, 65, 70, 80, 85])

# 2. Train the model
model = LinearRegression()
model.fit(X, y)

# 3. Generate predictions
predictions = model.predict(X)

# 4. Evaluate the model
r2 = r2_score(y, predictions)
print("R-squared score:", round(r2, 3))

# 5. Plot actual marks and regression line
plt.scatter(X.flatten(), y, label="Actual marks")
plt.plot(X.flatten(), predictions, label="Regression line")

plt.title("Study Hours vs Student Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
