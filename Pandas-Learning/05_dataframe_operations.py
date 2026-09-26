import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma"],
    "Age": [20, 21, 19, 22, 20],
    "Marks": [85, 78, 92, 67, 88]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

print("\nAverage marks:", df["Marks"].mean())

print("Highest marks:", df["Marks"].max())

print("Lowest marks:", df["Marks"].min())

print("\nSorted by marks:")
print(df.sort_values("Marks", ascending=False))

print("\nStudents with marks above 80:")
print(df[df["Marks"] > 80])