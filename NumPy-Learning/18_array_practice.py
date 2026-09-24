import numpy as np

arr = np.array([12, 5, 30, 18, 45, 22, 9, 35])

print("Original array:", arr)

print("Sorted array:", np.sort(arr))

print("Even numbers:", arr[arr % 2 == 0])

print("Numbers greater than 20:", arr[arr > 20])

print("Sum:", np.sum(arr))

print("Mean:", np.mean(arr))

print("Maximum:", np.max(arr))

print("Minimum:", np.min(arr))