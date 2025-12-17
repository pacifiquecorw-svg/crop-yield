
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 02:06:31 2025

@author: MIND-HACKER
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd


load_model = pickle.load(open('Crop_Yield_Model.sav', 'rb'))


def crop_yeild_prediction(test_data):
    
    Array_test_data = np.asarray(test_data)
    reshaped_array = Array_test_data.reshape(1,-1)
    prediction = load_model.predict(reshaped_array)
    return('Crop yield = ',prediction)

def main():
    st.title("🌾 Crop Yield Prediction Web App🌾")
    st.write("Developed by *GROUP 4*")
    st.write("Predict crop yield based on different parameter using a trained model.")
    
    
    Region = st.text_input('NUMBER OF REGION')
    Soil_Type = st.text_input('NUMBER OF SOIL TYPE')
    Crop = st.text_input('NUMBER OF CROP')
    Rainfall_mm = st.text_input('VALUE OF RAINFALL_MM')
    Temperature_Celsius = st.text_input('VALUE OF TEMPERATURE_CELICIUS')
    Fertilizer_Used = st.text_input('NUMBER OF FERTILIZER_USED')
    Irrigation_Used = st.text_input('NUMBER OF IRRIGATION_USED')
    Weather_Condition = st.text_input('NUMBER OF WEATHER CONDITION')
    Days_to_Harvest = st.text_input('VALUE OF DAY TO HARVEST')
    
    diagnosis = ''
    
    if st.button('CROP YIELD TEST RESULT'):
        diagnosis = crop_yeild_prediction([Region,Soil_Type,Crop,Rainfall_mm,Temperature_Celsius,Fertilizer_Used,Irrigation_Used,Weather_Condition,Days_to_Harvest])
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()

