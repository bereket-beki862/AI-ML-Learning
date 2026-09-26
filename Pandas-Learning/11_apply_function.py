import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Marks": [85, 78, 92, 67]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

df["Marks_Double"] = df["Marks"].apply(lambda x: x * 2)

print("\nAfter applying function:")
print(df)