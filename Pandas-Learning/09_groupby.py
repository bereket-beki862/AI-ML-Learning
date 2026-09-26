import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma", "Frank"],
    "Department": ["CSE", "ECE", "CSE", "ECE", "CSE", "ECE"],
    "Marks": [85, 78, 92, 67, 88, 75]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

print("\nAverage marks by department:")
print(df.groupby("Department")["Marks"].mean())

print("\nHighest marks by department:")
print(df.groupby("Department")["Marks"].max())