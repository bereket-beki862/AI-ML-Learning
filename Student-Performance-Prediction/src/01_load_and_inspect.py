import pandas as pd

# Load the student performance dataset
data = pd.read_csv("data/student-mat.csv", sep=";")

# Display the first five rows
print("\nFIRST FIVE STUDENTS:")
print(data.head())

# Display dataset dimensions
print("\nDATASET SHAPE:")
print(data.shape)

# Display column names
print("\nCOLUMN NAMES:")
print(data.columns.tolist())

# Display dataset information
print("\nDATASET INFORMATION:")
data.info()

# Check missing values
print("\nMISSING VALUES:")
print(data.isnull().sum())

# Display statistical summary
print("\nSTATISTICAL SUMMARY:")
print(data.describe())

# Display target variable
print("\nFINAL GRADES (G3):")
print(data["G3"].head())