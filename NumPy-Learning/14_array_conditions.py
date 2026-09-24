import numpy as np

arr = np.array([10, 20, 30, 40, 50])

print("Array:", arr)

print("Greater than 30:", arr > 30)

print("Elements greater than 30:", arr[arr > 30])

print("Even numbers:", arr[arr % 2 == 0])