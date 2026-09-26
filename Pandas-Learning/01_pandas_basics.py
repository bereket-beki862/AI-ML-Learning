import pandas as pd

print("Pandas version:", pd.__version__)

data = [10, 20, 30, 40, 50]

series = pd.Series(data)

print("Pandas Series:")
print(series)