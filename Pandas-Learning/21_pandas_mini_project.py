import pandas as pd

data = {
    "Name": [
        "Alice", "Bob", "Charlie", "David", "Emma",
        "Frank", "Grace", "Henry", "Ivy", "Jack"
    ],
    "Department": [
        "CSE", "ECE", "CSE", "ECE", "CSE",
        "ECE", "CSE", "ECE", "CSE", "ECE"
    ],
    "Age": [20, 21, 19, 22, 20, 21, 19, 22, 20, 21],
    "Marks": [85, 78, 92, 67, 88, 75, 95, 72, 81, 69]
}

df = pd.DataFrame(data)

print("Student Performance Dataset:")
print(df)