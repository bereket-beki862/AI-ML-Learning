import seaborn as sns
import matplotlib.pyplot as plt

subjects = ["Math", "Physics", "Chemistry", "Biology", "Programming"]
marks = [85, 78, 92, 88, 95]

sns.barplot(x=subjects, y=marks)

plt.title("Student Marks by Subject")
plt.xlabel("Subject")
plt.ylabel("Marks")

plt.show()