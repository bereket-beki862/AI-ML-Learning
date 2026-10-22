
import numpy as np

# Student exam marks
marks = [45, 55, 60, 65, 70, 75, 80, 85, 90, 95]

# Calculate percentiles
p25 = np.percentile(marks, 25)
p50 = np.percentile(marks, 50)
p75 = np.percentile(marks, 75)

# Calculate quartiles
q1 = p25
q2 = p50
q3 = p75

# Calculate interquartile range
iqr = q3 - q1

# Display results
print("Student Marks:", marks)
print("25th Percentile:", p25)
print("50th Percentile:", p50)
print("75th Percentile:", p75)

print("First Quartile (Q1):", q1)
print("Second Quartile (Q2):", q2)
print("Third Quartile (Q3):", q3)
print("Interquartile Range:", iqr)
