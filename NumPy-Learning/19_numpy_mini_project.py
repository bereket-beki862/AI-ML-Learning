import numpy as np

marks = np.array([78, 85, 92, 67, 88, 74, 95, 81, 69, 90])

print("Student marks:", marks)

print("Average marks:", np.mean(marks))
print("Highest marks:", np.max(marks))
print("Lowest marks:", np.min(marks))

print("Students scoring above 80:", marks[marks > 80])

print("Number of students scoring above 80:", np.sum(marks > 80))

print("Sorted marks:", np.sort(marks))