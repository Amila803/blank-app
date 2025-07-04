import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import os

# ─── 1. Resolve paths ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_FILE     = BASE_DIR / "Travel_details_dataset.csv"
MODEL_FILE    = BASE_DIR / "travel_cost_predictor.pkl"

# ─── 2. Load dataset ──────────────────────────────────────────────
if not DATA_FILE.exists():
    st.error(f"❌ Dataset not found at {DATA_FILE}")
    st.stop()

df = pd.read_csv(DATA_FILE)
st.sidebar.success(f"Loaded dataset with {len(df):,} rows")

# Use dataset to populate unique destinations
if "Destination" not in df.columns:
    st.error("❌ ‘Destination’ column not found in dataset")
    st.stop()
destinations = sorted(df["Destination"].dropna().unique())

# ─── 3. Load model ─────────────────────────────────────────────────
if not MODEL_FILE.exists():
    st.error(f"❌ Model file not found at {MODEL_FILE}")
    st.stop()

model = joblib.load(MODEL_FILE)

# ─── 4. App layout ──────────────────────────────────────────────────
st.title("✈️ Travel Cost Predictor")

with st.sidebar:
    st.header("Inputs")
    destination  = st.selectbox("Destination", destinations)
    nights       = st.number_input("Number of nights",    1, 30, 3)
    travelers    = st.number_input("Number of travelers", 1, 10, 1)
    predict_btn  = st.button("Predict Costs")

# ─── 5. Prediction logic ────────────────────────────────────────────
if predict_btn:
    X = pd.DataFrame([{
        "Destination": destination,
        "Nights":      nights,
        "NumTravelers": travelers
    }])
    try:
        accom_cost, transport_cost = model.predict(X)[0]
        st.metric("🏨 Accommodation Cost", f"RM {accom_cost:,.2f}")
        st.metric("🚌 Transport Cost",       f"RM {transport_cost:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ─── 6. (Optional) Data preview ─────────────────────────────────────
with st.expander("View raw dataset"):
    st.dataframe(df)
