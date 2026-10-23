
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a student dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 95],
    "Marks": [45, 50, 55, 65, 70, 78, 85, 95]
}

df = pd.DataFrame(data)

print("STUDENT DATA")
print(df)

# Calculate correlation between numeric columns
correlation_matrix = df.corr()

print("\nCORRELATION MATRIX")
print(correlation_matrix.round(2))

# Visualize correlations
plt.figure(figsize=(7, 5))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Student Data Correlation Heatmap")
plt.tight_layout()
plt.show()

# Plot study hours against marks
plt.figure(figsize=(7, 5))
plt.scatter(df["Study_Hours"], df["Marks"])
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.grid(True)
plt.tight_layout()
plt.show()
