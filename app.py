
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
    "East": 0,
    "North": 1,
    "South": 2,
    "West": 3
}

soil_map = {
    "Chalky": 0,
    "Clay": 1,
    "Loam": 2,
    "Peaty": 3,
    "Sandy": 4,
    "Silt": 5
}

crop_map = {
    "Barley": 0,
    "Cotton": 1,
    "Maize": 2,
    "Rice": 3,
    "Soybean": 4,
    "Wheate": 5
}

weather_map = {
    "Cloudy": 0,
    "Rainy": 1,
    "Sunny": 2
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

    Region = st.number_input("Region", min_value=0, max_value=3)
    Soil_Type = st.number_input("Soil Type", min_value=0, max_value=5)
    Crop = st.number_input("Crop", min_value=0, max_value=5)
    Weather_Condition = st.number_input("Weather Condition", min_value=0, max_value=2)

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