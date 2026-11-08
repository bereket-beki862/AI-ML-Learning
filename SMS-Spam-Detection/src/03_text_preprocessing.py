# Project 3: SMS Spam Detection
# Step 3: Text Preprocessing

import pandas as pd
import re
from pathlib import Path

# Find project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset paths
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
OUTPUT_PATH = BASE_DIR / "data" / "cleaned_sms.csv"


# Load dataset
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

print("Original dataset loaded successfully!")
print("Total messages:", len(df))


# Text cleaning function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning
df["clean_message"] = df["message"].apply(clean_text)


# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nText preprocessing completed successfully!")

print("\nOriginal message:")
print(df["message"].iloc[0])

print("\nCleaned message:")
print(df["clean_message"].iloc[0])

print("\nCleaned dataset saved to:")
print(OUTPUT_PATH)

print("\nFinal dataset shape:")
print(df.shape)

print("\nFirst five cleaned messages:")
print(df[["label", "clean_message"]].head())