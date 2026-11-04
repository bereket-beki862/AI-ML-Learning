import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/student-mat.csv", sep=";")

# Separate features and target
X = data.drop("G3", axis=1)
y = data["G3"]

# Convert categorical variables into numerical values
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Display dataset information
print("\nDATA PREPROCESSING COMPLETED!")

print("\nOriginal dataset shape:", data.shape)
print("Features shape:", X.shape)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

print("\nFirst five training samples:")
print(X_train.head())

print("\nFirst five target values:")
print(y_train.head())

# Check data types
print("\nFEATURE DATA TYPES:")
print(X.dtypes.value_counts())