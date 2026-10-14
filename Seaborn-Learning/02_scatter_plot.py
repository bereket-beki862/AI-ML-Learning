import seaborn as sns
import matplotlib.pyplot as plt

hours = [1, 2, 3, 4, 5, 6, 7, 8]
marks = [50, 55, 60, 65, 72, 78, 85, 90]

sns.scatterplot(x=hours, y=marks)

plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")

plt.show()