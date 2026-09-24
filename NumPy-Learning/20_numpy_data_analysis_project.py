import numpy as np

marks = np.array([
    78, 85, 92, 67, 88,
    74, 95, 81, 69, 90,
    56, 73, 84, 91, 63,
    77, 89, 68, 94, 82
])

print("===== STUDENT MARKS ANALYSIS =====")

print("Total students:", len(marks))

print("Average marks:", np.mean(marks))

print("Highest marks:", np.max(marks))

print("Lowest marks:", np.min(marks))

print("Median marks:", np.median(marks))

print("Standard deviation:", np.std(marks))

passed = marks[marks >= 40]
failed = marks[marks < 40]

print("Number of passed students:", len(passed))
print("Number of failed students:", len(failed))

above_80 = marks[marks >= 80]

print("Students scoring 80 or above:", above_80)
print("Number scoring 80 or above:", len(above_80))

print("Sorted marks:", np.sort(marks))

print("===== ANALYSIS COMPLETE =====")