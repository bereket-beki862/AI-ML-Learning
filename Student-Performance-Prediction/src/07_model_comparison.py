
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Find project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load dataset
data = pd.read_csv(
    PROJECT_ROOT / "data" / "student-mat.csv",
    sep=";"
)

# Separate features and target
X = data.drop("G3", axis=1)
y = data["G3"]

# Identify categorical features
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_columns)
    ],
    remainder="passthrough"
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Define models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}

# Store comparison results
results = []

# Train and evaluate each model
for name, algorithm in models.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", algorithm)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "R2 Score": round(r2, 3)
    })

# Create comparison table
comparison = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
print("----------------------------")
print(comparison.to_string(index=False))

# Identify the model with the lowest MAE
best_model = comparison.loc[
    comparison["MAE"].idxmin()
]

print("\nBEST MODEL BASED ON MAE:")
print(best_model["Model"])

# Save results
output_path = PROJECT_ROOT / "results" / "model_comparison.csv"
comparison.to_csv(output_path, index=False)

print("\nComparison saved to:", output_path)
