
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
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
    [6, 68],
    [7, 75],
    [8, 85],
    [4, 50]
])

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Store WCSS values
wcss = []

# Test different numbers of clusters
for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

    print(f"K = {k}, WCSS = {model.inertia_:.2f}")

# Plot the Elbow graph
plt.plot(range(1, 11), wcss, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.xticks(range(1, 11))
plt.grid(True)

plt.show()
