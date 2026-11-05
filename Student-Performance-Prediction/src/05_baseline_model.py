
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Find project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load original dataset
data = pd.read_csv(
    PROJECT_ROOT / "data" / "student-mat.csv",
    sep=";"
)

# Separate input features and target
X = data.drop("G3", axis=1)
y = data["G3"]

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

# Convert categorical features into numerical values
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_columns)
    ],
    remainder="passthrough"
)

# Build machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

# Display results
print("\nBASELINE MODEL: LINEAR REGRESSION")
print("----------------------------------")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nMODEL PERFORMANCE:")
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R2 Score:", round(r2, 3))

# Compare actual and predicted grades
comparison = pd.DataFrame({
    "Actual Grade": y_test.values,
    "Predicted Grade": predictions.round(2)
})

print("\nACTUAL VS PREDICTED GRADES:")
print(comparison.head(10))
