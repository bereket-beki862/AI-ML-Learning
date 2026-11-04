import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("data/student-mat.csv", sep=";")

# Set visualization style
sns.set_theme(style="whitegrid")


# 1. Distribution of final grades
plt.figure(figsize=(8, 5))
sns.histplot(data["G3"], bins=20, kde=True)

plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.savefig("results/final_grade_distribution.png")
plt.show()


# 2. Average final grade by gender
plt.figure(figsize=(7, 5))
sns.barplot(data=data, x="sex", y="G3")

plt.title("Average Final Grade by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Final Grade")

plt.savefig("results/grade_by_gender.png")
plt.show()


# 3. Study time vs final grade
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="studytime", y="G3")

plt.title("Study Time vs Final Grade")
plt.xlabel("Weekly Study Time")
plt.ylabel("Final Grade")

plt.savefig("results/studytime_vs_grade.png")
plt.show()


# 4. Previous grades vs final grade
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x="G1", y="G3", label="G1")
sns.scatterplot(data=data, x="G2", y="G3", label="G2")

plt.title("Previous Grades vs Final Grade")
plt.xlabel("Previous Grades")
plt.ylabel("Final Grade")
plt.legend()

plt.savefig("results/previous_grades_vs_final.png")
plt.show()


# 5. Correlation heatmap
numeric_data = data.select_dtypes(include="number")
correlation = numeric_data.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Numerical Features")
plt.savefig("results/correlation_heatmap.png")
plt.show()


# 6. Display important statistics
print("\nFINAL GRADE STATISTICS:")
print(data["G3"].describe())

print("\nAVERAGE GRADE BY GENDER:")
print(data.groupby("sex")["G3"].mean())

print("\nAVERAGE GRADE BY STUDY TIME:")
print(data.groupby("studytime")["G3"].mean())

print("\nTOP CORRELATIONS WITH FINAL GRADE:")
print(correlation["G3"].sort_values(ascending=False))

print("\nEDA COMPLETED SUCCESSFULLY!")