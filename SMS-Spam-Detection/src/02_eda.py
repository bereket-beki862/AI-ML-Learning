
# Project 3: SMS Spam Detection
# Step 2: Exploratory Data Analysis (EDA)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import re

# Find project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# File paths
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
RESULTS_DIR = BASE_DIR / "results"

# Create results folder if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

# Calculate message length
df["message_length"] = df["message"].apply(len)

print("EDA started successfully!")

# 1. Display dataset information
print("\nDataset shape:")
print(df.shape)

print("\nMessage length statistics:")
print(df["message_length"].describe())

# 2. Count spam and ham messages
print("\nMessage distribution:")
print(df["label"].value_counts())

# 3. Plot spam and ham distribution
plt.figure(figsize=(7, 5))
df["label"].value_counts().plot(kind="bar")

plt.title("Spam vs Ham Message Distribution")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(RESULTS_DIR / "01_message_distribution.png")
plt.show()

# 4. Plot message length distribution
plt.figure(figsize=(8, 5))

for label in ["ham", "spam"]:
    df[df["label"] == label]["message_length"].hist(
        bins=40, alpha=0.6, label=label
    )

plt.title("Message Length Distribution")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

plt.savefig(RESULTS_DIR / "02_message_length.png")
plt.show()

# 5. Compare average message length
print("\nAverage message length:")
print(df.groupby("label")["message_length"].mean())

# 6. Find common words
def get_common_words(messages, top_n=15):
    words = []

    for message in messages:
        words.extend(re.findall(r"\b[a-zA-Z]{2,}\b", message.lower()))

    return Counter(words).most_common(top_n)

spam_words = get_common_words(
    df[df["label"] == "spam"]["message"]
)

ham_words = get_common_words(
    df[df["label"] == "ham"]["message"]
)

print("\nMost common spam words:")
print(spam_words)

print("\nMost common ham words:")
print(ham_words)

# 7. Plot common spam words
spam_word_df = pd.DataFrame(
    spam_words, columns=["word", "count"]
)

plt.figure(figsize=(9, 5))
plt.bar(spam_word_df["word"], spam_word_df["count"])

plt.title("Most Common Words in Spam Messages")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(RESULTS_DIR / "03_common_spam_words.png")
plt.show()

print("\nEDA completed successfully!")
print(f"Charts saved in: {RESULTS_DIR}")
