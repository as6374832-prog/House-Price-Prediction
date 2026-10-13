
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Sample housing dataset
data = {
    "area_sqft": [
        500, 650, 800, 900, 1000,
        1100, 1200, 1350, 1500, 1700,
        1800, 2000, 2200, 2400, 2600
    ],
    "bedrooms": [
        1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5
    ],
    "price_lakhs": [
        20, 25, 32, 35, 40, 43, 48, 55, 60, 68,
        72, 80, 88, 96, 105
    ]
}

df = pd.DataFrame(data)

# Input features and target
X = df[["area_sqft", "bedrooms"]]
y = df["price_lakhs"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("R² Score:", round(r2_score(y_test, predictions), 3))

# Predict a sample house price
new_house = pd.DataFrame({
    "area_sqft": [1200],
    "bedrooms": [3]
})

predicted_price = model.predict(new_house)[0]
print(f"Predicted house price: ₹{predicted_price:.2f} lakhs")

# Visualize actual versus predicted prices
plt.scatter(y_test, predictions)
plt.xlabel("Actual price (lakhs)")
plt.ylabel("Predicted price (lakhs)")
plt.title("House Price Prediction: Actual vs Predicted")
plt.tight_layout()
plt.show()
