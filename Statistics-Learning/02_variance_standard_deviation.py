
import statistics

# Student marks
marks = [85, 78, 92, 67, 88, 78, 95, 80, 78]

# Calculate variance and standard deviation
variance_value = statistics.variance(marks)
std_deviation = statistics.stdev(marks)

print("Student Marks:", marks)
print("Variance:", round(variance_value, 2))
print("Standard Deviation:", round(std_deviation, 2))
