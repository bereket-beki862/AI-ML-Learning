
# PCA Mini Project
# Reduce the Iris dataset from 4 features to 2 dimensions

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Display original data
print("Original dataset shape:", X.shape)

# 3. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Display results
print("\nReduced dataset shape:", X_pca.shape)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

total_variance = sum(pca.explained_variance_ratio_)

print("\nTotal variance preserved:")
print(round(total_variance * 100, 2), "%")

# 6. Create a DataFrame
df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Species": iris.target_names[y]
})

print("\nFirst five rows:")
print(df.head())

# 7. Visualize PCA results
plt.figure(figsize=(8, 6))

for species in iris.target_names:
    subset = df[df["Species"] == species]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=species
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Mini Project - Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()
