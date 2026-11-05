
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

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

# Identify categorical columns
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ("categorical",
     OneHotEncoder(handle_unknown="ignore"),
     categorical_columns)
], remainder="passthrough")

# Define cross-validation strategy
cv = KFold(
    n_splits=5,
    shuffle=True,
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

print("\n5-FOLD CROSS-VALIDATION")
print("-----------------------")

for name, algorithm in models.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", algorithm)
    ])

    # Negative MAE is used because sklearn
    # follows a higher-is-better scoring convention
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error"
    )

    mae_scores = -scores

    print(f"\n{name}")
    print("MAE for each fold:", mae_scores.round(3))
    print("Average MAE:", round(mae_scores.mean(), 3))
    print("Standard Deviation:", round(mae_scores.std(), 3))

print("\nCROSS-VALIDATION COMPLETED!")
