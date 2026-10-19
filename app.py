
import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("house_price_model.joblib")
# Display feature importance
st.subheader("What influences house values?")

if hasattr(model, "feature_importances_"):
    features = [
        "Median income",
        "Median house age",
        "Average rooms",
        "Average bedrooms",
        "Area population",
        "Average occupancy",
        "Latitude",
        "Longitude",
    ]

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "Importance", ascending=True
    )

    st.bar_chart(
        importance.set_index("Feature")["Importance"]
    )

st.title("California House Value Predictor")
st.write("Estimate a census area's median house value.")
st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("MAE", "0.349")

with col2:
    st.metric("R² Score", "0.790")

st.caption(
    "Metrics are from the model's evaluation run. "
    "The dataset contains historical California housing data."
)

# Collect input features
med_inc = st.number_input("Median income", 0.5, 20.0, 5.0)
house_age = st.number_input("Median house age", 1.0, 52.0, 25.0)
ave_rooms = st.number_input("Average rooms", 1.0, 20.0, 5.0)
ave_bedrms = st.number_input("Average bedrooms", 0.5, 10.0, 1.0)
population = st.number_input("Area population", 1.0, 40000.0, 1000.0)
ave_occup = st.number_input("Average occupancy", 1.0, 20.0, 3.0)
latitude = st.number_input("Latitude", 32.0, 42.0, 34.0)
longitude = st.number_input("Longitude", -125.0, -114.0, -118.0)

if st.button("Predict house value"):
    house = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    prediction = model.predict(house)[0] * 100000
    st.success(f"Predicted median value: ${prediction:,.0f}")
    st.caption("Educational estimate from historical California data, not a current property valuation.")
