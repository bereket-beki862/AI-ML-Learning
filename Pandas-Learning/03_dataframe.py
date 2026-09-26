import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma"],
    "Age": [20, 21, 19, 22, 20],
    "Marks": [85, 78, 92, 67, 88]
}

df = pd.DataFrame(data)

print("Student Data:")
print(df)

print("\nShape:", df.shape)
print("Columns:", df.columns.tolist())
print("Number of rows:", len(df))