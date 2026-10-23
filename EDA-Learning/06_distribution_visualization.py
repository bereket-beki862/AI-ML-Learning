
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Student marks
data = {
    "Marks": [
        45, 50, 55, 60, 62, 65, 68, 70,
        72, 75, 78, 80, 82, 85, 88, 92, 98
    ]
}

df = pd.DataFrame(data)

# Display summary statistics
print("SUMMARY STATISTICS")
print(df["Marks"].describe())

# Plot a histogram
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Marks", bins=8, kde=True)
plt.title("Distribution of Student Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Plot a box plot
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="Marks")
plt.title("Box Plot of Student Marks")
plt.xlabel("Marks")
plt.tight_layout()
plt.show()
