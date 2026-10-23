
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create student data
data = {
    "Name": [
        "Alice", "Bob", "Charlie", "David",
        "Emma", "Frank", "Grace", "Helen"
    ],
    "Study_Group": [
        "Morning", "Evening", "Morning", "Evening",
        "Morning", "Evening", "Morning", "Evening"
    ],
    "Study_Hours": [5, 2, 7, 3, 6, 4, 8, 2],
    "Marks": [92, 60, 96, 68, 88, 75, 99, 55]
}

df = pd.DataFrame(data)

# Compare average marks by study group
group_means = df.groupby("Study_Group")["Marks"].mean()

print("AVERAGE MARKS BY STUDY GROUP")
print(group_means.round(2))

# Summarize each group
print("\nGROUP SUMMARY")
print(
    df.groupby("Study_Group")[["Study_Hours", "Marks"]]
    .mean()
    .round(2)
)

# Visualize average marks
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="Study_Group", y="Marks", errorbar=None)
plt.title("Marks by Study Group")
plt.xlabel("Study Group")
plt.ylabel("Marks")
plt.tight_layout()
plt.show()

# Compare the distribution of marks
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Study_Group", y="Marks")
plt.title("Mark Distribution by Study Group")
plt.xlabel("Study Group")
plt.ylabel("Marks")
plt.tight_layout()
plt.show()
