
import statistics

# Student marks
marks = [85, 78, 92, 67, 88, 78, 95, 80, 78]

# Calculate mean, median, and mode
mean_value = statistics.mean(marks)
median_value = statistics.median(marks)
mode_value = statistics.mode(marks)

# Display the results
print("Student Marks:", marks)
print("Mean:", mean_value)
print("Median:", median_value)
print("Mode:", mode_value)
