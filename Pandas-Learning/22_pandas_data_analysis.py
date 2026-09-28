import pandas as pd

data = {
    "Name": [
        "Alice", "Bob", "Charlie", "David", "Emma",
        "Frank", "Grace", "Henry", "Ivy", "Jack",
        "Kate", "Leo", "Mia", "Noah", "Olivia"
    ],
    "Department": [
        "CSE", "ECE", "CSE", "ECE", "CSE",
        "ECE", "CSE", "ECE", "CSE", "ECE",
        "CSE", "ECE", "CSE", "ECE", "CSE"
    ],
    "Age": [20, 21, 19, 22, 20, 21, 19, 22, 20, 21, 19, 22, 20, 21, 19],
    "Marks": [85, 78, 92, 67, 88, 75, 95, 72, 81, 69, 90, 76, 84, 71, 93]
}

df = pd.DataFrame(data)

print("Student Performance Dataset:")
print(df)
print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)
print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())
print("\nAverage Marks:")
print(df["Marks"].mean())

print("\nHighest Marks:")
print(df["Marks"].max())

print("\nLowest Marks:")
print(df["Marks"].min())

print("\nMedian Marks:")
print(df["Marks"].median())
print("\nAverage Marks by Department:")
print(df.groupby("Department")["Marks"].mean())

print("\nHighest Marks by Department:")
print(df.groupby("Department")["Marks"].max())

print("\nNumber of Students by Department:")
print(df["Department"].value_counts())
print("\nStudents scoring 80 or above:")
top_students = df[df["Marks"] >= 80]
print(top_students)

print("\nTop students sorted by marks:")
print(top_students.sort_values("Marks", ascending=False))
# Students scoring below 70
print("\nStudents scoring below 70:")
low_students = df[df["Marks"] < 70]
print(low_students)

# Students sorted by marks
print("\nAll Students Sorted by Marks:")
print(df.sort_values("Marks", ascending=False))

# Student with the highest marks
highest_student = df.loc[df["Marks"].idxmax()]

print("\nTop Performing Student:")
print(highest_student)

# Student with the lowest marks
lowest_student = df.loc[df["Marks"].idxmin()]

print("\nLowest Performing Student:")
print(lowest_student)

# Average age
print("\nAverage Age:")
print(df["Age"].mean())

# Students with marks 90 or above
print("\nStudents Scoring 90 or Above:")
print(df[df["Marks"] >= 90])

# Final summary
print("\n========== FINAL ANALYSIS SUMMARY ==========")
print("Total Students:", len(df))
print("Average Marks:", round(df["Marks"].mean(), 2))
print("Highest Marks:", df["Marks"].max())
print("Lowest Marks:", df["Marks"].min())
print("Average Age:", round(df["Age"].mean(), 2))
print("CSE Students:", (df["Department"] == "CSE").sum())
print("ECE Students:", (df["Department"] == "ECE").sum())
print("Students scoring 80 or above:", (df["Marks"] >= 80).sum())
print("Students scoring below 70:", (df["Marks"] < 70).sum())