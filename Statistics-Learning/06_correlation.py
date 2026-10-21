
import numpy as np
import matplotlib.pyplot as plt

# Hours studied and exam marks
hours_studied = [1, 2, 3, 4, 5, 6]
exam_marks = [40, 45, 55, 60, 70, 80]

# Calculate correlation coefficient
correlation = np.corrcoef(hours_studied, exam_marks)[0, 1]

print("Hours Studied:", hours_studied)
print("Exam Marks:", exam_marks)
print("Correlation Coefficient:", round(correlation, 2))

# Visualize the relationship
plt.scatter(hours_studied, exam_marks)
plt.title("Study Hours vs Exam Marks")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Marks")
plt.grid(True)
plt.show()
