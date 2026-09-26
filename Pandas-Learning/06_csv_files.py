import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [20, 21, 19, 22],
    "Marks": [85, 78, 92, 67]
})

df.to_csv("students.csv", index=False)

print("CSV file created successfully!")

data = pd.read_csv("students.csv")

print("\nData from CSV:")
print(data)