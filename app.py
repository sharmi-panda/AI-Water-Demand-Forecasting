import streamlit as st
import pandas as pd
import joblib
import datetime
import os
import numpy as np

# 1. Get the directory where app.py is sitting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Try looking for 'processed_data' in the same folder (Cloud style)
# If it's not there, look one level up (Your local laptop style)
processed_path = os.path.join(BASE_DIR, "processed_data")
if not os.path.exists(processed_path):
    processed_path = os.path.join(BASE_DIR, "..", "processed_data")

# 3. Load the models using the verified path
model = joblib.load(os.path.join(processed_path, 'water_model.pkl'))
le = joblib.load(os.path.join(processed_path, 'location_encoder.pkl'))
le_status = joblib.load(os.path.join(processed_path, 'status_encoder.pkl'))

# --- DASHBOARD UI ---
st.set_page_config(page_title="Campus Water AI", layout="wide")
st.title("AI-Based Water Demand Forecasting")
st.markdown("### Decision Support System for Campus Facilities")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Current Conditions (IoT Simulation)")
location = st.sidebar.selectbox("Select Campus Block", le.classes_)
temp = st.sidebar.slider("Ambient Temperature (°C)", 10, 45, 25)
status = st.sidebar.selectbox("Academic Status", le_status.classes_)
current_tank = st.sidebar.progress(45, text="Current Tank Level: 45%") # Simulated IoT feed

# --- AI PREDICTION LOGIC ---
today = datetime.date.today()
day_of_week = today.weekday()
month = today.month
is_weekend = 1 if day_of_week >= 5 else 0
tank_val = st.sidebar.slider("Simulate Tank Level (%)", 0, 100, 45)
st.sidebar.progress(tank_val, text=f"Current Tank Level: {tank_val}%")

# Add a Smart Alert immediately below it in the sidebar
if tank_val < 20:
    st.sidebar.error("CRITICAL: Tank level is too low!")
elif tank_val < 50:
    st.sidebar.warning("Low Water: Consider pumping soon.")
# Prepare data for prediction
loc_id = le.transform([location])[0]
status_id = le_status.transform([status])[0]
# Inside your app.py prediction logic:
# 1. Prepare ONLY the 5 features your model was trained on
# --- AI PREDICTION LOGIC ---
input_data = pd.DataFrame([[
    temp,           # Ambient_Temp_C
    day_of_week,    # Day_of_Week
    month,          # Month
    is_weekend,     # Is_Weekend
    loc_id       # Location 
]], columns=['Ambient_Temp_C', 'Day_of_Week', 'Month', 'Is_Weekend', 'Location_ID'])

# 1. Get the EXACT list of names the model is looking for
model_features = model.get_booster().feature_names 
# Based on your previous logs, this is: 
# ['Ambient_Temp_C', 'Day_of_Week', 'Month', 'Is_Weekend', 'Location_ID', 'Status_ID']

# 2. Create the data values - you MUST have 6 items here to match the 6 features
data_values = [[
    float(temp), 
    float(day_of_week), 
    float(month), 
    float(is_weekend), 
    float(loc_id), 
 # <--- This was the missing 6th item!
]]

# 3. Create the DataFrame using the model's own names
input_data = pd.DataFrame(data_values, columns=model_features)

# 4. Predict
prediction = model.predict(input_data)[0]

SAVINGS_PER_LITER = 0.002  # Estimated cost saved by pumping off-peak
CO2_OFFSET_FACTOR = 0.8    # kg of CO2 saved per 1000L by using night-grid power

impact_cost = prediction * SAVINGS_PER_LITER
impact_co2 = (prediction / 1000) * CO2_OFFSET_FACTOR

# --- DISPLAY RESULTS ---
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Next-Day Predicted Demand", value=f"{prediction:.2f} Liters")
    st.info(f"The AI expects a {'spike' if prediction > 15000 else 'normal'} demand for {location}.")

with col2:
    st.subheader("Optimal Pumping Schedule")
    # Optimization Logic: If demand is high and it's a cheap electricity hour
    current_hour = datetime.datetime.now().hour
    is_off_peak = 22 <= current_hour or current_hour <= 6
    
    if prediction > 12000 and is_off_peak:
        st.success("RECOMMENDATION: START PUMPING NOW (Off-Peak Rates Active)")
    elif prediction > 12000 and not is_off_peak:
        st.warning("RECOMMENDATION: DELAY PUMPING until 10:00 PM to save electricity costs.")
    else:
        st.write("Current tank levels are sufficient for predicted demand.")

st.divider()
st.subheader("Demand Trend vs. Capacity")

# Create some dummy historical data for visualization
chart_data = pd.DataFrame({
    'Time': pd.date_range(start='today', periods=24, freq='H'),
    'Predicted Demand': np.random.randint(1000, 2000, size=24).cumsum(),
    'Tank Capacity': [20000] * 24
})

st.line_chart(chart_data, x='Time', y=['Predicted Demand', 'Tank Capacity'], color=["#2E7D32", "#FF5733"])
st.divider()
st.subheader("Sustainability Impact (Solo Project Contribution)")

# Create three clear columns for the "Green Metrics"
m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Potential Cost Savings", f"₹{impact_cost:.2f}", help="Savings from moving pumping to off-peak hours.")
with m2:
    st.metric("Estimated CO2 Offset", f"{impact_co2:.2f} kg", delta="Environmentally Friendly")
with m3:
    st.metric("Project Goal Progress", "85%", help="Based on the 30-hour SMART goals.")

st.info(f"**Human Impact:** By following this AI's advice for {location}, we reduce the daily carbon footprint by roughly the equivalent of planting {int(impact_co2/2)} small trees.")