import numpy as np

arr = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("First row:", arr[0, :])
print("First column:", arr[:, 0])
print("First two rows:")
print(arr[0:2, :])