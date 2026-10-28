
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1. Create sample data with a curved relationship
X = np.linspace(-3, 3, 30).reshape(-1, 1)
y = X.ravel() ** 2 + np.random.RandomState(42).normal(0, 1.5, 30)

# 2. Create three models with different complexity
models = {
    "Underfitting (degree 1)": make_pipeline(
        PolynomialFeatures(degree=1),
        LinearRegression()
    ),
    "Good fit (degree 2)": make_pipeline(
        PolynomialFeatures(degree=2),
        LinearRegression()
    ),
    "Overfitting risk (degree 15)": make_pipeline(
        PolynomialFeatures(degree=15),
        LinearRegression()
    )
}

# 3. Plot the data and each model
X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Observed data")

for name, model in models.items():
    model.fit(X, y)
    predictions = model.predict(X_plot)
    train_predictions = model.predict(X)

    mse = mean_squared_error(y, train_predictions)

    print(name)
    print("Training MSE:", round(mse, 2))
    print()

    plt.plot(X_plot, predictions, label=name)

plt.title("Underfitting vs. Overfitting")
plt.xlabel("X")
plt.ylabel("y")
plt.ylim(-3, 15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
