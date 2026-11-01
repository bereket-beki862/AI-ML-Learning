
# Anomaly Detection using Isolation Forest

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. Create sample data
np.random.seed(42)

normal_data = np.random.normal(
    loc=50,
    scale=5,
    size=(100, 2)
)

# 2. Add unusual data points
anomalies = np.array([
    [10, 10],
    [90, 90],
    [15, 85],
    [95, 20],
    [5, 60]
])

# 3. Combine normal data and anomalies
X = np.vstack((normal_data, anomalies))

# 4. Create Isolation Forest model
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

# 5. Train the model and predict
predictions = model.fit_predict(X)

# 6. Separate normal points and anomalies
normal_points = X[predictions == 1]
anomaly_points = X[predictions == -1]

# 7. Display results
print("Total data points:", len(X))
print("Normal points:", len(normal_points))
print("Anomalies detected:", len(anomaly_points))

print("\nAnomaly coordinates:")
print(anomaly_points)

# 8. Visualize results
plt.figure(figsize=(8, 6))

plt.scatter(
    normal_points[:, 0],
    normal_points[:, 1],
    label="Normal Data"
)

plt.scatter(
    anomaly_points[:, 0],
    anomaly_points[:, 1],
    label="Anomalies",
    marker="x",
    s=100
)

plt.title("Anomaly Detection using Isolation Forest")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

plt.show()
