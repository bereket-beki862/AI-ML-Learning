
import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 data points from a normal distribution
data = np.random.normal(loc=50, scale=10, size=1000)

# Calculate statistics
mean_value = np.mean(data)
std_deviation = np.std(data)

print("Mean:", round(mean_value, 2))
print("Standard Deviation:", round(std_deviation, 2))

# Plot the distribution
plt.hist(data, bins=30, edgecolor="black")
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
