
import statistics

# Population: marks of all students in a class
population = [60, 70, 75, 80, 85, 90, 95, 100]

# Sample: marks of four selected students
sample = [70, 80, 90, 100]

# Calculate the population mean
population_mean = statistics.mean(population)

# Calculate the sample mean
sample_mean = statistics.mean(sample)

print("Population:", population)
print("Sample:", sample)
print("Population size:", len(population))
print("Sample size:", len(sample))
print("Population mean:", population_mean)
print("Sample mean:", sample_mean)
