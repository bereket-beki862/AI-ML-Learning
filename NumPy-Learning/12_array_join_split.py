import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print("Array 1:", arr1)
print("Array 2:", arr2)

joined = np.concatenate((arr1, arr2))

print("Joined array:", joined)

split_arrays = np.array_split(joined, 2)

print("Split arrays:")
print(split_arrays[0])
print(split_arrays[1])