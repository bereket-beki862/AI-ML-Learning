import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

print("Original array:", arr)

new_arr = arr.reshape(2, 3)

print("Reshaped array:")
print(new_arr)

print("New shape:", new_arr.shape)