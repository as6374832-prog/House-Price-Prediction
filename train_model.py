
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
df = pd.read_csv("california_housing.csv")

# 2. Separate features (X) and target (y)
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create the machine-learning model
model = LinearRegression()

# 5. Train the model
model.fit(X_train, y_train)

# 6. Predict house values for the test data
predictions = model.predict(X_test)

# 7. Evaluate the predictions
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model: Linear Regression")
print(f"Mean Absolute Error: {mae:.3f}")
print(f"R² Score: {r2:.3f}")
