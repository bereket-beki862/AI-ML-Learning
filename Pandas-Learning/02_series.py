import pandas as pd

data = [10, 20, 30, 40, 50]

series = pd.Series(data)

print("Series:")
print(series)

print("First value:", series[0])
print("Last value:", series[4])

print("Values greater than 25:")
print(series[series > 25])

print("Mean:", series.mean())
print("Maximum:", series.max())
print("Minimum:", series.min())