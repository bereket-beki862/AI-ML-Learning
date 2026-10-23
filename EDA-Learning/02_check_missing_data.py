
import pandas as pd

# Create a dataset with missing values
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma"],
    "Study_Hours": [5, 2, None, 3, 6],
    "Attendance": [95, None, 98, 75, 90],
    "Marks": [92, 60, 96, None, 88]
}

df = pd.DataFrame(data)

print("ORIGINAL DATA")
print(df)

# Count missing values in each column
print("\nMISSING VALUES PER COLUMN")
print(df.isnull().sum())

# Calculate the percentage of missing values
print("\nMISSING VALUE PERCENTAGE")
print((df.isnull().mean() * 100).round(2))

# Fill missing numeric values with each column's mean
numeric_columns = ["Study_Hours", "Attendance", "Marks"]
df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].mean()
)

print("\nDATA AFTER FILLING MISSING VALUES")
print(df)

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())
