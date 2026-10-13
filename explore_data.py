
import pandas as pd
from sklearn.datasets import fetch_california_housing

# 1. Download and load the real dataset
housing = fetch_california_housing(as_frame=True)

# 2. Convert it into a pandas DataFrame
df = housing.frame.copy()

# 3. Save a local CSV file
df.to_csv("california_housing.csv", index=False)

# 4. Explore the dataset
print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATASET SIZE ---")
print(df.shape)

print("\n--- COLUMN INFORMATION ---")
df.info()

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())
