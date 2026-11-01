
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

print("Original Dataset Shape:", X.shape)

# Step 1: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 3: Display reduced dataset
print("\nReduced Dataset:")
print(X_pca)

print("\nReduced Dataset Shape:", X_pca.shape)

# Step 4: Explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(
    "Total Variance Preserved:",
    round(sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)

# Step 5: Visualize PCA results
plt.figure(figsize=(8, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    s=100
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Student Dataset")
plt.grid(True)

plt.show()
