import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Create the dataset
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
    "Study_Hours": [2, 3, 4, 2, 5, 3, 6, 4, 5, 7, 6, 8, 4, 5, 7],
    "Marks": [55, 60, 70, 58, 82, 65, 92, 75, 85, 95, 90, 98, 72, 78, 94]
}

df = pd.DataFrame(data)

# 2. Inspect the data
print("STUDENT PERFORMANCE DATA")
print(df)

print("\nDataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df[["Study_Hours", "Marks"]].describe())

# 3. Analyze department performance
department_summary = df.groupby("Department").agg(
    Student_Count=("Student", "count"),
    Average_Marks=("Marks", "mean"),
    Highest_Marks=("Marks", "max"),
    Average_Study_Hours=("Study_Hours", "mean")
)

print("\nDepartment Analysis:")
print(department_summary.round(2))

# 4. Find top-performing students
print("\nTop 5 Students:")
print(df.nlargest(5, "Marks")[["Student", "Department", "Marks"]])

# 5. Calculate correlation
correlation = df[["Study_Hours", "Marks"]].corr()

print("\nCorrelation Matrix:")
print(correlation.round(2))

# 6. Visualize the results
sns.set_theme(style="whitegrid")

# Graph 1: Marks distribution
sns.histplot(data=df, x="Marks", bins=6, kde=True)
plt.title("Distribution of Student Marks")
plt.tight_layout()
plt.show()

# Graph 2: Average marks by department
sns.barplot(data=df, x="Department", y="Marks", errorbar=None)
plt.title("Average Marks by Department")
plt.tight_layout()
plt.show()

# Graph 3: Study hours vs marks
sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Marks",
    hue="Department",
    s=100
)
plt.title("Study Hours vs Marks")
plt.tight_layout()
plt.show()

# Graph 4: Marks distribution by department
sns.boxplot(data=df, x="Department", y="Marks")
plt.title("Marks Distribution by Department")
plt.tight_layout()
plt.show()

# Graph 5: Correlation heatmap
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
plt.title("Study Hours and Marks Correlation")
plt.tight_layout()
plt.show()

# 7. Final report
print("\n========== FINAL REPORT ==========")
print("Total Students:", len(df))
print("Overall Average Marks:", round(df["Marks"].mean(), 2))
print("Highest Marks:", df["Marks"].max())
print("Lowest Marks:", df["Marks"].min())
print("Students scoring 80 or above:", (df["Marks"] >= 80).sum())
print("Students scoring below 70:", (df["Marks"] < 70).sum())
print("Average Study Hours:", round(df["Study_Hours"].mean(), 2))