
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create a sample student dataset
data = {
    "Name": [
        "Alice", "Bob", "Charlie", "David", "Emma",
        "Frank", "Grace", "Helen", "Ivan", "Julia"
    ],
    "Study_Hours": [5, 2, 7, 3, 6, 4, 8, 2, 5, 6],
    "Attendance": [95, 70, 98, 75, 90, 82, 99, 65, 85, 92],
    "Marks": [92, 60, 96, 68, 88, 75, 99, 55, 82, 90]
}

df = pd.DataFrame(data)

# 2. Inspect the dataset
print("FIRST FIVE ROWS")
print(df.head())

print("\nDATASET INFORMATION")
df.info()

# 3. Check missing values and duplicates
print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS:", df.duplicated().sum())

# 4. Summary statistics
print("\nSUMMARY STATISTICS")
print(df[["Study_Hours", "Attendance", "Marks"]].describe().round(2))

# 5. Correlation analysis
correlation = df[["Study_Hours", "Attendance", "Marks"]].corr()

print("\nCORRELATION MATRIX")
print(correlation.round(2))

# 6. Identify the top student by marks
top_student = df.loc[df["Marks"].idxmax()]

print("\nTOP STUDENT")
print(top_student)

# 7. Visualize marks distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Marks", bins=6, kde=True)
plt.title("Distribution of Student Marks")
plt.tight_layout()
plt.show()

# 8. Compare study hours and marks
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Study_Hours", y="Marks", s=100)
plt.title("Study Hours vs Marks")
plt.tight_layout()
plt.show()

# 9. Visualize correlations
plt.figure(figsize=(7, 5))
sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Student Data Correlation")
plt.tight_layout()
plt.show()

# 10. Save the analyzed dataset
df.to_csv("student_eda_results.csv", index=False)

print("\nAnalysis complete!")
print("Dataset saved as student_eda_results.csv")
