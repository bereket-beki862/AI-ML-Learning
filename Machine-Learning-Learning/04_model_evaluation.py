
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Actual student marks
actual = np.array([50, 60, 70, 80, 90])

# Marks predicted by our model
predicted = np.array([52, 58, 72, 75, 88])

# 1. Mean Absolute Error (MAE)
mae = mean_absolute_error(actual, predicted)

# 2. Mean Squared Error (MSE)
mse = mean_squared_error(actual, predicted)

# 3. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# 4. R-squared score
r2 = r2_score(actual, predicted)

# Display results
print("MODEL EVALUATION RESULTS")
print("-------------------------")
print("Actual marks:", actual)
print("Predicted marks:", predicted)

print("\nEvaluation Metrics")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R-squared:", round(r2, 3))
