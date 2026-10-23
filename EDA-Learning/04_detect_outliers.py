
import pandas as pd

# Student marks, including an unusually high value
data = {
    "Name": [
        "Alice", "Bob", "Charlie", "David",
        "Emma", "Frank", "Grace", "Helen"
    ],
    "Marks": [75, 80, 85, 78, 82, 79, 81, 150]
}

df = pd.DataFrame(data)

print("ORIGINAL DATA")
print(df)

# Calculate quartiles
q1 = df["Marks"].quantile(0.25)
q3 = df["Marks"].quantile(0.75)

# Calculate interquartile range
iqr = q3 - q1

# Calculate outlier boundaries
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("\nQ1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

# Identify outliers
outliers = df[
    (df["Marks"] < lower_bound) |
    (df["Marks"] > upper_bound)
]

print("\nOUTLIERS")
print(outliers)

# Display data without detected outliers
cleaned_df = df[
    (df["Marks"] >= lower_bound) &
    (df["Marks"] <= upper_bound)
]

print("\nDATA WITHOUT DETECTED OUTLIERS")
print(cleaned_df)
