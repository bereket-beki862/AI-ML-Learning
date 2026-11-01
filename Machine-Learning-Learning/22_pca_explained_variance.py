
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Student dataset
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

# Step 1: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA with all components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Step 3: Display explained variance
print("Explained Variance Ratio:")

for i, variance in enumerate(pca.explained_variance_ratio_):
    print(f"Component {i + 1}: {variance:.4f}")

# Step 4: Calculate cumulative variance
cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

print("\nCumulative Explained Variance:")

for i, variance in enumerate(cumulative_variance):
    print(f"First {i + 1} components: {variance:.4f}")

# Step 5: Find components needed for 95% variance
optimal_components = np.argmax(
    cumulative_variance >= 0.95
) + 1

print("\nComponents needed for 95% variance:")
print(optimal_components)

# Step 6: Plot explained variance
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.axhline(y=0.95, linestyle="--", label="95% Variance")

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Cumulative Explained Variance")
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.legend()
plt.grid(True)

plt.show()
