import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma", "Frank"],
    "Age": [20, 21, 19, 22, 20, 21],
    "Marks": [85, 78, 92, 67, 88, 75]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

print("\nAverage marks:", df["Marks"].mean())

print("\nStudents scoring above 80:")
print(df[df["Marks"] > 80])

print("\nSorted by marks:")
print(df.sort_values("Marks", ascending=False))

print("\nHighest marks:", df["Marks"].max())

print("\nLowest marks:", df["Marks"].min())

print("\nNumber of students:", len(df))