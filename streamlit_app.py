# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import os

# —————————————————————————————————————————————
# 1. Locate your model file relative to this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "travel_cost_predictor.pkl"
MODEL_PATH = BASE_DIR / MODEL_FILENAME

# 2. Debug info (optional—you can remove these later)
st.write("Working directory:", os.getcwd())
st.write("Looking for model at:", MODEL_PATH)
st.write("Files here:", os.listdir(BASE_DIR))

# 3. Load the model (stop if missing)
if not MODEL_PATH.exists():
    st.error(f"❌ Cannot find `{MODEL_FILENAME}` in {BASE_DIR}")
    st.stop()

model = joblib.load(MODEL_PATH)

# —————————————————————————————————————————————
# 4. Build your UI
st.title("✈️ Travel Cost Predictor")

# Example feature inputs — adjust these to match your training data!
destination = st.selectbox("Destination", [
    "London", "Phuket", "Bali", "New York", "Tokyo", "Paris",
    "Sydney", "Dubai", "Bangkok"
])

n_nights = st.number_input("Number of nights", min_value=1, max_value=30, value=3)
num_travelers = st.number_input("Number of travelers", min_value=1, max_value=10, value=1)

# 5. Predict button
if st.button("Predict Cost"):
    # Assemble into a DataFrame in the same format your model expects
    X = pd.DataFrame([{
        "Destination": destination,
        "Nights": n_nights,
        "NumTravelers": num_travelers
    }])
    try:
        cost = model.predict(X)[0]
        st.success(f"Estimated total cost: RM {cost:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
