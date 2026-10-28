
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create sample student data
X = np.array([
    [1, 60],
    [2, 65],
    [3, 70],
    [4, 75],
    [5, 80],
    [6, 85],
    [7, 90],
    [8, 95],
    [9, 92],
    [10, 98],
    [3, 68],
    [6, 88],
    [8, 90],
    [2, 62],
    [7, 87]
])

# Columns: study hours, attendance percentage
y = np.array([
    35, 40, 48, 55, 65,
    72, 82, 90, 93, 98,
    50, 78, 88, 38, 80
])

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Define the models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=3, random_state=42
    )
}

# 4. Train and evaluate each model
print("MODEL COMPARISON")
print("=" * 45)

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel:", name)
    print("Mean Absolute Error:", round(mae, 2))
    print("R-squared:", round(r2, 3))
