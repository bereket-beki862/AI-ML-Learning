
import pandas as pd
from pathlib import Path

# Find the main project folder automatically
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load the original dataset
data_path = PROJECT_ROOT / "data" / "student-mat.csv"
data = pd.read_csv(data_path, sep=";")

print("ORIGINAL DATASET SHAPE:", data.shape)

# Feature 1: Average of the first and second period grades
data["previous_grade_average"] = data[["G1", "G2"]].mean(axis=1)

# Feature 2: Grade improvement or decline from G1 to G2
data["grade_progress"] = data["G2"] - data["G1"]

# Feature 3: Whether the student has experienced past failures
data["has_past_failures"] = (data["failures"] > 0).astype(int)

# Save the engineered dataset
output_path = PROJECT_ROOT / "data" / "student_features.csv"
data.to_csv(output_path, index=False, sep=";")

# Display results
print("\nFEATURE ENGINEERING COMPLETED!")

print("\nNEW FEATURES:")
print(data[[
    "G1",
    "G2",
    "previous_grade_average",
    "grade_progress",
    "failures",
    "has_past_failures",
    "G3"
]].head())

print("\nUPDATED DATASET SHAPE:", data.shape)

print("\nNEW FEATURES SUMMARY:")
print(data[[
    "previous_grade_average",
    "grade_progress",
    "has_past_failures"
]].describe())

print("\nSaved engineered dataset to:", output_path)
