import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [20, None, 19, 22],
    "Marks": [85, 78, None, 67]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

print("\nMissing values:")
print(df.isnull())

print("\nNumber of missing values:")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())

print("\nDataFrame after filling missing values:")
print(df)