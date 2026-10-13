
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the dataset
df = pd.read_csv("california_housing.csv")

# 2. Separate input features from the target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Display the sizes of each set
print("Total rows:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Number of input features:", X.shape[1])
