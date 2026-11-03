
# Machine Learning Pipeline

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 3. Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=200))
])

# 4. Train the complete pipeline
pipeline.fit(X_train, y_train)

# 5. Make predictions
predictions = pipeline.predict(X_test)

# 6. Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("Machine Learning Pipeline")
print("-------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Model accuracy:", round(accuracy * 100, 2), "%")

# 7. Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = pipeline.predict(new_flower)

print("\nNew flower prediction:")
print(iris.target_names[prediction[0]])
