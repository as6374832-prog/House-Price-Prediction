
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Load the data
df = pd.read_csv("california_housing.csv")

# 2. Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# 5. Get feature importance scores
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

# 6. Plot the results
importance.plot(kind="barh", figsize=(9, 6))
plt.title("Feature Importance in House Value Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
