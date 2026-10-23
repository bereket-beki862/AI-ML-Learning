
import pandas as pd

# Create a dataset containing duplicate records
data = {
    "Name": ["Alice", "Bob", "Charlie", "Alice", "David", "Bob"],
    "Study_Hours": [5, 2, 7, 5, 3, 2],
    "Marks": [92, 60, 96, 92, 68, 60]
}

df = pd.DataFrame(data)

print("ORIGINAL DATA")
print(df)

# Count duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Display duplicate rows
print("\nDUPLICATE RECORDS")
print(df[df.duplicated()])

# Remove duplicate rows
df_cleaned = df.drop_duplicates()

print("\nDATA AFTER REMOVING DUPLICATES")
print(df_cleaned)

print("\nNumber of rows before:", len(df))
print("Number of rows after:", len(df_cleaned))
