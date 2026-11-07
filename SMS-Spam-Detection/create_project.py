
from pathlib import Path

# Use the folder where this Python file is saved
project = Path(__file__).resolve().parent

# Project folders
folders = [
    "data",
    "notebooks",
    "src",
    "models",
    "results"
]

# Starter files
files = [
    "README.md",
    "requirements.txt",
    ".gitignore"
]

# Create project folders
for folder in folders:
    (project / folder).mkdir(parents=True, exist_ok=True)

# Create starter files
for file in files:
    (project / file).touch(exist_ok=True)

print("Project structure created successfully!")
print(f"Project location: {project}")

print("\nCreated folders:")
for folder in folders:
    print(f"- {folder}")

print("\nCreated files:")
for file in files:
    print(f"- {file}")
