
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Create student data
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma",
             "Frank", "Grace", "Helen", "Ivan", "Julia"],
    "Math": [85, 72, 90, 65, 88, 76, 95, 60, 82, 91],
    "Python": [92, 78, 88, 70, 95, 80, 98, 65, 85, 90],
    "Statistics": [80, 75, 92, 68, 90, 82, 96, 62, 84, 94]
}

df = pd.DataFrame(data)

# 2. Calculate each student's average
subjects = ["Math", "Python", "Statistics"]
df["Average"] = df[subjects].mean(axis=1)

# 3. Calculate overall statistics
all_marks = df[subjects].to_numpy().flatten()

print("STUDENT PERFORMANCE")
print(df.to_string(index=False))

print("\nOVERALL STATISTICS")
print("Mean:", round(np.mean(all_marks), 2))
print("Median:", round(np.median(all_marks), 2))
print("Standard deviation:", round(np.std(all_marks), 2))
print("Minimum mark:", np.min(all_marks))
print("Maximum mark:", np.max(all_marks))
print("75th percentile:", np.percentile(all_marks, 75))

# 4. Find the highest-performing student
top_index = df["Average"].idxmax()
print("\nTop student:", df.loc[top_index, "Name"])
print("Top student's average:", round(df.loc[top_index, "Average"], 2))

# 5. Plot average marks by student
plt.figure(figsize=(10, 5))
plt.bar(df["Name"], df["Average"])
plt.title("Average Student Performance")
plt.xlabel("Student")
plt.ylabel("Average Mark")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Plot mark distribution
plt.figure(figsize=(8, 5))
plt.hist(all_marks, bins=8, edgecolor="black")
plt.title("Distribution of Student Marks")
plt.xlabel("Mark")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 7. Compare average marks by subject
subject_means = df[subjects].mean()

plt.figure(figsize=(7, 5))
plt.bar(subject_means.index, subject_means.values)
plt.title("Average Marks by Subject")
plt.xlabel("Subject")
plt.ylabel("Average Mark")
plt.tight_layout()
plt.show()
