
import joblib
import pandas as pd

# 1. Load the saved model
model = joblib.load("house_price_model.joblib")

# 2. Enter sample housing data
house = pd.DataFrame([{
    "MedInc": 5.0,
    "HouseAge": 25.0,
    "AveRooms": 5.0,
    "AveBedrms": 1.0,
    "Population": 1000.0,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0
}])

# 3. Predict the house value
predicted_value = model.predict(house)[0]

# 4. Convert the result into dollars
predicted_dollars = predicted_value * 100000

print(f"Predicted value: ${predicted_dollars:,.0f}")
