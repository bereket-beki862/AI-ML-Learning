
import statistics

# Student marks
marks = [60, 70, 80, 90, 100]

# Calculate mean and standard deviation
mean_value = statistics.mean(marks)
std_deviation = statistics.pstdev(marks)

# Choose one student's mark
student_mark = 90

# Calculate Z-score
z_score = (student_mark - mean_value) / std_deviation

print("Student Marks:", marks)
print("Mean:", mean_value)
print("Standard Deviation:", round(std_deviation, 2))
print("Student Mark:", student_mark)
print("Z-score:", round(z_score, 2))
