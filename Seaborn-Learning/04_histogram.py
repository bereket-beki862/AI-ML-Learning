import seaborn as sns
import matplotlib.pyplot as plt

marks = [45, 50, 55, 60, 62, 65, 68, 70, 72, 75,
         78, 80, 82, 85, 88, 90, 92, 95, 96, 98]

sns.histplot(x=marks, bins=5)

plt.title("Distribution of Student Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Students")

plt.show()