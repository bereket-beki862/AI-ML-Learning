
import pandas as pd

# Create a sample student dataset
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma",
             "Frank", "Grace", "Helen"],
    "Study_Hours": [5, 2, 7, 3, 6, 4, 8, 2],
    "Attendance": [95, 70, 98, 75, 90, 82, 99, 65],
    "Marks": [92, 60, 96, 68, 88, 75, 99, 55]
}

df = pd.DataFrame(data)

# Inspect the dataset
print("FIRST FIVE ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN INFORMATION")
df.info()

print("\nSUMMARY STATISTICS")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())
