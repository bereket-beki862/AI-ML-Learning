
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Find project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load dataset
data = pd.read_csv(
    PROJECT_ROOT / "data" / "student-mat.csv",
    sep=";"
)

# Remove previous grades
X = data.drop(
    columns=["G1", "G2", "G3"]
)

# Target variable
y = data["G3"]

# Identify categorical features
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ("categorical",
     OneHotEncoder(handle_unknown="ignore"),
     categorical_columns)
], remainder="passthrough")

# Build model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

# Display results
print("\nSTUDENT PERFORMANCE WITHOUT PREVIOUS GRADES")
print("--------------------------------------------")

print("Features used:", X.shape[1])
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
print(comparison.head(10).to_string(index=False))

# Save predictions
comparison.to_csv(
    PROJECT_ROOT / "results" / "predictions_without_previous_grades.csv",
    index=False
)

print("\nPredictions saved successfully!")
