import seaborn as sns
import matplotlib.pyplot as plt

data = {
    "Department": ["CSE", "CSE", "CSE", "CSE", "ECE", "ECE", "ECE", "ECE"],
    "Marks": [65, 75, 80, 95, 55, 70, 85, 90]
}

sns.boxplot(data=data, x="Department", y="Marks")

plt.title("Marks Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Marks")

plt.show()