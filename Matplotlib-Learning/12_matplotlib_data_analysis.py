import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Student": [
        "Alice", "Bob", "Charlie", "David", "Emma",
        "Frank", "Grace", "Henry", "Ivy", "Jack",
        "Kate", "Leo", "Mia", "Noah", "Olivia"
    ],
    "Department": [
        "CSE", "ECE", "CSE", "ECE", "CSE",
        "ECE", "CSE", "ECE", "CSE", "ECE",
        "CSE", "ECE", "CSE", "ECE", "CSE"
    ],
    "Marks": [
        85, 78, 92, 67, 88,
        75, 95, 72, 81, 69,
        90, 76, 84, 71, 93
    ]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

# Basic analysis
print("\nAverage Marks:", round(df["Marks"].mean(), 2))
print("Highest Marks:", df["Marks"].max())
print("Lowest Marks:", df["Marks"].min())

# 1. Bar Chart - Student Marks
plt.bar(df["Student"], df["Marks"])

plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 2. Histogram - Marks Distribution
plt.hist(df["Marks"], bins=5)

plt.title("Distribution of Student Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Students")

plt.show()

# 3. Department-wise Average Marks
department_average = df.groupby("Department")["Marks"].mean()

print("\nAverage Marks by Department:")
print(department_average)

plt.bar(department_average.index, department_average.values)

plt.title("Average Marks by Department")
plt.xlabel("Department")
plt.ylabel("Average Marks")

plt.show()

# 4. Scatter Plot - Student Index vs Marks
student_number = range(1, len(df) + 1)

plt.scatter(student_number, df["Marks"])

plt.title("Student Performance")
plt.xlabel("Student Number")
plt.ylabel("Marks")

plt.show()

# Final summary
print("\n========== FINAL SUMMARY ==========")
print("Total Students:", len(df))
print("Average Marks:", round(df["Marks"].mean(), 2))
print("Highest Marks:", df["Marks"].max())
print("Lowest Marks:", df["Marks"].min())
print("Students scoring 80 or above:", (df["Marks"] >= 80).sum())
print("Students scoring below 70:", (df["Marks"] < 70).sum())