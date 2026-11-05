
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
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

# Build Random Forest pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# Train model
model.fit(X, y)

# Extract feature importance
importance = model.named_steps[
    "regressor"
].feature_importances_

# Get transformed feature names
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Create importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display top 15 features
print("\nTOP 15 IMPORTANT FEATURES")
print("-------------------------")
print(importance_df.head(15).to_string(index=False))

# Save feature importance
importance_df.to_csv(
    PROJECT_ROOT / "results" / "feature_importance.csv",
    index=False
)

# Visualize top 15 features
plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance_df.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Features Influencing Student Performance")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "results" / "feature_importance.png"
)

plt.show()

print("\nFEATURE IMPORTANCE ANALYSIS COMPLETED!")
