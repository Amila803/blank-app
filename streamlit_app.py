import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import os

# ─── 1. Paths ───────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_FILE  = BASE_DIR / "Travel_details_dataset.csv"
MODEL_FILE = BASE_DIR / "travel_cost_predictor.pkl"

# ─── 2. Load dataset ───────────────────────────────────────────────
if not DATA_FILE.exists():
    st.error(f"❌ Dataset not found at {DATA_FILE}")
    st.stop()
df = pd.read_csv(DATA_FILE)
destinations = sorted(df["Destination"].dropna().unique())

# ─── 3. Load model ─────────────────────────────────────────────────
if not MODEL_FILE.exists():
    st.error(f"❌ Model file not found at {MODEL_FILE}")
    st.stop()

loaded = joblib.load(MODEL_FILE)

# If it’s a dict, show keys and pick the first sub‐model that has predict()
if isinstance(loaded, dict):
    st.warning(f"Loaded a dict with keys: {list(loaded.keys())}")
    # try to find a value with a predict method
    for k, v in loaded.items():
        if hasattr(v, "predict"):
            st.info(f"Using entry '{k}' from the dict as the model")
            model = v
            break
    else:
        st.error("❌ No sub‐model with a `.predict()` method found in the dict.")
        st.stop()
else:
    model = loaded

# ─── 4. UI ─────────────────────────────────────────────────────────
st.title("✈️ Travel Cost Predictor")

destination = st.sidebar.selectbox("Destination", destinations)
nights      = st.sidebar.number_input("Number of nights",    1, 30, 3)
travelers   = st.sidebar.number_input("Number of travelers", 1, 10, 1)
if st.sidebar.button("Predict Costs"):
    X = pd.DataFrame([{
        "Destination":  destination,
        "Nights":       nights,
        "NumTravelers": travelers
    }])
    try:
        # if multi-output
        out = model.predict(X)[0]
        if hasattr(out, "__len__") and len(out) == 2:
            accom_cost, transport_cost = out
            st.metric("🏨 Accommodation Cost", f"RM {accom_cost:,.2f}")
            st.metric("🚌 Transport Cost",   f"RM {transport_cost:,.2f}")
        else:
            st.success(f"Predicted cost: {out}")
    except Exception as e:
        st.error(f"Prediction error:\n{e}")

# ─── 5. Debug ───────────────────────────────────────────────────────
with st.expander("Debug info"):
    st.write("Working dir:", os.getcwd())
    st.write("Files here:", os.listdir(BASE_DIR))
    st.write("Model object type:", type(model))
