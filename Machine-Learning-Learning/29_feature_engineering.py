
# Feature Engineering

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Create sample student dataset
data = {
    "Study_Hours": [2, 4, 6, 8, 3, 7],
    "Attendance": [60, 70, 80, 90, 65, 85],
    "Assignment_Score": [50, 60, 75, 90, 55, 85],
    "Exam_Score": [45, 55, 70, 88, 50, 82]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# 2. Create a new feature
df["Total_Preparation"] = (
    df["Study_Hours"] * 0.5 +
    df["Attendance"] * 0.3 +
    df["Assignment_Score"] * 0.2
)

# 3. Create another feature
df["Performance_Gap"] = (
    df["Exam_Score"] - df["Assignment_Score"]
)

print("\nDataset with New Features:")
print(df)

# 4. Apply Min-Max Scaling
minmax = MinMaxScaler()

df["Study_Hours_Scaled"] = minmax.fit_transform(
    df[["Study_Hours"]]
)

# 5. Apply Standard Scaling
standard = StandardScaler()

df["Exam_Score_Standardized"] = standard.fit_transform(
    df[["Exam_Score"]]
)

print("\nDataset After Scaling:")
print(df.round(2))

# 6. Save the dataset
df.to_csv("feature_engineering_results.csv", index=False)

print("\nDataset saved successfully!")
