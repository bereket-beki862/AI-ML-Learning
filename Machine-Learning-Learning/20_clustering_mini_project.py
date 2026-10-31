
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# Step 1: Create student dataset
data = {
    "Study_Hours": [
        1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
        6, 7, 7, 8, 8, 9, 3, 5, 7, 9
    ],
    "Attendance": [
        35, 40, 45, 50, 55, 50, 60, 55, 65, 60,
        70, 65, 75, 80, 85, 90, 42, 68, 72, 95
    ]
}

df = pd.DataFrame(data)

print("STUDENT DATASET")
print(df.head())

# Step 2: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Step 3: K-Means Clustering
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# Step 4: Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X_scaled)

# Step 5: Evaluate clustering
kmeans_score = silhouette_score(
    X_scaled, df["KMeans_Cluster"]
)

hierarchical_score = silhouette_score(
    X_scaled, df["Hierarchical_Cluster"]
)

print("\nCLUSTERING PERFORMANCE")
print(f"K-Means Silhouette Score: {kmeans_score:.3f}")
print(f"Hierarchical Silhouette Score: {hierarchical_score:.3f}")

# Step 6: Visualize K-Means clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Study_Hours"],
    df["Attendance"],
    c=df["KMeans_Cluster"],
    cmap="viridis",
    s=100
)

plt.xlabel("Study Hours")
plt.ylabel("Attendance (%)")
plt.title("K-Means Student Segmentation")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# Step 7: Visualize Hierarchical clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Study_Hours"],
    df["Attendance"],
    c=df["Hierarchical_Cluster"],
    cmap="plasma",
    s=100
)

plt.xlabel("Study Hours")
plt.ylabel("Attendance (%)")
plt.title("Hierarchical Student Segmentation")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# Step 8: Create dendrogram
plt.figure(figsize=(10, 5))

linked = linkage(X_scaled, method="ward")
dendrogram(linked)

plt.title("Student Clustering Dendrogram")
plt.xlabel("Student Index")
plt.ylabel("Ward Linkage Distance")
plt.tight_layout()
plt.show()

# Step 9: Save results
df.to_csv("student_clustering_results.csv", index=False)

print("\nFINAL DATASET")
print(df)

print("\nProject completed successfully!")
print("Results saved to student_clustering_results.csv")
