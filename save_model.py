
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
df = pd.read_csv("california_housing.csv")

# 2. Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the model
model = RandomForestRegressor(
    n_estimators=20,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 5. Evaluate it
predictions = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, predictions):.3f}")
print(f"R²: {r2_score(y_test, predictions):.3f}")

# 6. Save the trained model
joblib.dump(model, "house_price_model.joblib")
print("Model saved successfully!")
