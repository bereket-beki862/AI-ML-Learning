import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma"],
    "Age": [20, 21, 19, 22, 20],
    "Marks": [85, 78, 92, 67, 88]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

print("\nStudents with marks above 80:")
print(df[df["Marks"] > 80])

print("\nStudents with age 20:")
print(df[df["Age"] == 20])

print("\nStudents with marks between 70 and 90:")
print(df[(df["Marks"] >= 70) & (df["Marks"] <= 90)])