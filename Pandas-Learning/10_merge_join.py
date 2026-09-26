import pandas as pd

students = pd.DataFrame({
    "ID": [1, 2, 3, 4],
    "Name": ["Alice", "Bob", "Charlie", "David"]
})

marks = pd.DataFrame({
    "ID": [1, 2, 3, 4],
    "Marks": [85, 78, 92, 67]
})

print("Students:")
print(students)

print("\nMarks:")
print(marks)

merged = pd.merge(students, marks, on="ID")

print("\nMerged DataFrame:")
print(merged)