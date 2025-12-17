
import streamlit as st
import pickle
import numpy as np
import pandas as pd


# ================= LOAD MODEL =================

load_model = pickle.load(
    open("Crop_Yield_Model.sav","rb")
)

# ================= ENCODING MAPS =================
region_map = {
    "North": 0,
    "South": 1,
    "East": 2,
    "West": 3
}

soil_map = {
    "Sandy": 0,
    "Loamy": 1,
    "Clay": 2
}

crop_map = {
    "Maize": 0,
    "Rice": 1,
    "Wheat": 2
}

weather_map = {
    "Sunny": 0,
    "Rainy": 1,
    "Cloudy": 2
}

# ================= SHOW ENCODING TABLE =================
def show_encoding_table():
    st.subheader("📊 Data Encoding Reference")

    encoding_df = pd.DataFrame({
        "Feature": ["Region", "Soil Type", "Crop", "Weather Condition"],
        "Encoding": [
            str(region_map),
            str(soil_map),
            str(crop_map),
            str(weather_map)
        ]
    })

    st.table(encoding_df)

# ================= PREDICTION FUNCTION =================
def crop_yield_prediction(test_data):
    test_data = np.asarray(test_data, dtype=float)
    reshaped_array = test_data.reshape(1, -1)
    prediction = load_model.predict(reshaped_array)
    return f"🌾 Predicted Crop Yield: {prediction[0]:.2f} tons/hectare"

# ================= MAIN APP =================
def main():
    st.title("🌾 Crop Yield Prediction Web App 🌾")
    st.write("Developed by *GROUP 4*")
    st.write("Predict crop yield using encoded agricultural data.")

    show_encoding_table()

    st.subheader("🔢 Enter Input Data")

    Region = st.selectbox("Region", list(region_map.keys()))
    Soil_Type = st.selectbox("Soil Type", list(soil_map.keys()))
    Crop = st.selectbox("Crop", list(crop_map.keys()))
    Weather_Condition = st.selectbox("Weather Condition", list(weather_map.keys()))

    Rainfall_mm = st.number_input("Rainfall (mm)", min_value=0.0)
    Temperature_Celsius = st.number_input("Temperature (°C)")
    Fertilizer_Used = st.number_input("Fertilizer Used (0 = No, 1 = Yes)", min_value=0, max_value=1)
    Irrigation_Used = st.number_input("Irrigation Used (0 = No, 1 = Yes)", min_value=0, max_value=1)
    Days_to_Harvest = st.number_input("Days to Harvest", min_value=1)

    if st.button("🌾 CROP YIELD TEST RESULT"):
        input_data = [
            region_map[Region],
            soil_map[Soil_Type],
            crop_map[Crop],
            Rainfall_mm,
            Temperature_Celsius,
            Fertilizer_Used,
            Irrigation_Used,
            weather_map[Weather_Condition],
            Days_to_Harvest
        ]

        result = crop_yield_prediction(input_data)
        st.success(result)

# ================= RUN APP =================
if __name__ == "__main__":
    main()