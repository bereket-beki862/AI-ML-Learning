import numpy as np

arr = np.array([50, 20, 40, 10, 30])

print("Original array:", arr)

print("Sorted array:", np.sort(arr))

print("Positions of elements greater than 30:", np.where(arr > 30))

print("Positions of 20:", np.where(arr == 20))