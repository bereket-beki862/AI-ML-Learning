
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Student dataset: study hours and attendance
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

# Create the clustering model
model = AgglomerativeClustering(n_clusters=3)

# Train and assign clusters
clusters = model.fit_predict(X_scaled)

# Display results
print("Hierarchical Clustering Results:")

for i in range(len(X)):
    print(
        f"Student {i + 1}: "
        f"Study Hours = {X[i][0]}, "
        f"Attendance = {X[i][1]}%, "
        f"Cluster = {clusters[i]}"
    )

# Visualize clusters
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=clusters,
    cmap="viridis",
    s=100
)

plt.xlabel("Study Hours")
plt.ylabel("Attendance Percentage")
plt.title("Hierarchical Clustering")
plt.colorbar(label="Cluster")
plt.show()
