
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Student dataset
# Features: study hours, attendance, assignment score, exam score

X = np.array([
    [2, 40, 45, 50],
    [3, 45, 50, 55],
    [4, 50, 55, 60],
    [5, 55, 60, 65],
    [6, 60, 65, 70],
    [7, 65, 70, 75],
    [8, 70, 75, 80],
    [9, 80, 85, 90],
    [3, 42, 48, 52],
    [6, 68, 72, 78],
    [7, 75, 80, 85],
    [8, 85, 90, 95]
])

# Step 1: Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA for 2D
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

print("2D Dataset Shape:", X_2d.shape)
print("2D Variance Preserved:",
      round(sum(pca_2d.explained_variance_ratio_) * 100, 2), "%")

# Step 3: Apply PCA for 3D
pca_3d = PCA(n_components=3)
X_3d = pca_3d.fit_transform(X_scaled)

print("\n3D Dataset Shape:", X_3d.shape)
print("3D Variance Preserved:",
      round(sum(pca_3d.explained_variance_ratio_) * 100, 2), "%")

# Step 4: Create 2D visualization
plt.figure(figsize=(8, 5))

plt.scatter(X_2d[:, 0], X_2d[:, 1], s=100)

for i in range(len(X_2d)):
    plt.annotate(f"S{i + 1}", (X_2d[i, 0], X_2d[i, 1]))

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA 2D Visualization")
plt.grid(True)
plt.show()

# Step 5: Create 3D visualization
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X_3d[:, 0],
    X_3d[:, 1],
    X_3d[:, 2],
    s=100
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA 3D Visualization")

plt.show()
