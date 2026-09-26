import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma"],
    "Age": [20, 21, 19, 22, 20],
    "Marks": [85, 78, 92, 67, 88]
}

df = pd.DataFrame(data)

print("Names:")
print(df["Name"])

print("\nMarks:")
print(df["Marks"])

print("\nFirst row:")
print(df.iloc[0])

print("\nFirst three rows:")
print(df.iloc[0:3])

print("\nStudents with marks above 80:")
print(df[df["Marks"] > 80])