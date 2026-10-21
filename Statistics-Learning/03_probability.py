
import random

# Total possible outcomes when rolling a die
total_outcomes = 6

# Outcomes that give an even number
favorable_outcomes = 3

# Calculate probability
probability = favorable_outcomes / total_outcomes

print("Total outcomes:", total_outcomes)
print("Favorable outcomes:", favorable_outcomes)
print("Probability of getting an even number:", round(probability, 2))

# Simulate rolling a die
roll = random.randint(1, 6)
print("Dice roll result:", roll)
