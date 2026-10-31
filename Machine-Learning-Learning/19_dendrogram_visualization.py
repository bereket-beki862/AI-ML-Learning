
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Student dataset
X = np.array([
    [1, 35],
    [2, 40],
    [2, 45],
    [3, 50],
    [4, 55],
    [5, 60],
    [6, 65],
    [7, 70],
    [8, 80],
    [9, 90],
    [3, 42],
    [6, 68]
])

# Scale the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform hierarchical clustering
linked = linkage(X_scaled, method="ward")

# Create dendrogram
plt.figure(figsize=(10, 6))

dendrogram(
    linked,
    labels=np.arange(1, len(X) + 1),
    leaf_rotation=0,
    leaf_font_size=10
)

plt.title("Student Hierarchical Clustering Dendrogram")
plt.xlabel("Student Number")
plt.ylabel("Distance")
plt.grid(axis="y")

plt.show()
