
# Project 3: SMS Spam Detection
# Step 1: Load and inspect the dataset

import pandas as pd
from pathlib import Path

# Find the project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset location
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"

# Load dataset
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

# Display basic information
print("Dataset loaded successfully!")

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nSpam and ham percentages:")
print(df["label"].value_counts(normalize=True) * 100)
