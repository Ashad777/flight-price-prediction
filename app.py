import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load the trained model and feature columns
model = joblib.load("flight_price_model.pkl")
feature_columns = joblib.load("model_features.pkl")

# App title
st.title("✈️ Flight Price Prediction App")
st.write("Predict the flight ticket price based on travel details.")

# --- User Inputs ---
# Airline
airlines = ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Vistara', 'GoAir', 'Multiple carriers']
airline = st.selectbox("Select Airline", airlines)

# Source
sources = ['Delhi', 'Kolkata', 'Banglore', 'Mumbai', 'Chennai']
source = st.selectbox("Source", sources)

# Destination
destinations = ['Cochin', 'Banglore', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata']
destination = st.selectbox("Destination", destinations)

# Journey Date
journey_date = st.date_input("Journey Date", min_value=datetime(2019,1,1), max_value=datetime(2019,12,31))
day = journey_date.day
month = journey_date.month

# Departure Time
dep_time = st.time_input("Departure Time")
dep_hour = dep_time.hour
dep_minute = dep_time.minute

# Arrival Time
arr_time = st.time_input("Arrival Time")
arr_hour = arr_time.hour
arr_minute = arr_time.minute

# Duration in minutes
duration_mins = st.slider("Duration (in minutes)", min_value=30, max_value=1800, value=180)

# Total Stops
stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])

# --- Feature Preparation ---
# Create an empty feature array
input_data = np.zeros(len(feature_columns))

# Assign numeric features
input_data[feature_columns.get_loc("Journey_Day")] = day
input_data[feature_columns.get_loc("Journey_Month")] = month
input_data[feature_columns.get_loc("Dep_Hour")] = dep_hour
input_data[feature_columns.get_loc("Dep_Minute")] = dep_minute
input_data[feature_columns.get_loc("Arrival_Hour")] = arr_hour
input_data[feature_columns.get_loc("Arrival_Minute")] = arr_minute
input_data[feature_columns.get_loc("Duration_mins")] = duration_mins
input_data[feature_columns.get_loc("Total_Stops")] = stops

# One-hot encoding for categorical values
airline_col = "Airline_" + airline
source_col = "Source_" + source
destination_col = "Destination_" + destination

if airline_col in feature_columns:
    input_data[feature_columns.get_loc(airline_col)] = 1
if source_col in feature_columns:
    input_data[feature_columns.get_loc(source_col)] = 1
if destination_col in feature_columns:
    input_data[feature_columns.get_loc(destination_col)] = 1

# Predict Button
if st.button("Predict Price"):
    input_data = input_data.reshape(1, -1)
    predicted_price = model.predict(input_data)[0]
    st.success(f"Estimated Flight Price: ₹ {round(predicted_price, 2)}")
