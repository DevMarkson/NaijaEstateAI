import os
import json
import joblib
import pandas as pd
import streamlit as st
from config import BEST_MODEL_PATH, METRICS_PATH, DATA_PATH, FEATURE_COLUMNS

PROPERTY_TYPES = [
    "Duplex", "Semi Detached", "Detached", "Apartment", "Flat",
    "Bungalow", "Terrace", "Terraced", "Mansion", "Studio",
    "Penthouse", "Mini Flat", "Self Contain", "Maisonette",
    "Terrace Duplex", "Terraced Duplex", "House", "Other",
]

@st.cache_resource(show_spinner=False)
def load_model():
    if not BEST_MODEL_PATH.exists():
        return None
    return joblib.load(BEST_MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_data_peek(n=5000):
    try:
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH, nrows=n)
            df.columns = [c.strip() for c in df.columns]
            return df
    except Exception:
        pass
    return None

st.set_page_config(page_title="NaijaEstateAI • Rent Predictor", page_icon="🏠", layout="centered")
st.title("NaijaEstateAI — Lagos Rent Price Predictor")
st.caption("Predict likely rent (₦) from listing attributes. Model: best of Linear/RandomForest/XGBoost.")

model = load_model()
metrics = load_metrics()
df_peek = load_data_peek()

if metrics:
    with st.expander("Model metrics", expanded=False):
        best = metrics.get("best")
        st.write(f"Best model: {best}")
        baseline = metrics.get("baseline")
        if baseline:
            st.markdown("**Baseline (Median per Neighborhood)**")
            st.write({k: round(v,2) if isinstance(v,(int,float)) else v for k,v in baseline.items()})
        st.markdown("**Models (Holdout + CV)**")
        st.json(metrics.get("results", {}))
        if metrics.get("removed_outliers") is not None:
            st.caption(f"Outliers removed: {metrics['removed_outliers']} • Target transform: {metrics.get('target_transform')}")

if model is None:
    st.error("No trained model found. Run: python train.py --data lagos-rent.csv (or schedule training).")
    st.stop()

st.subheader("Enter listing details")

col1, col2, col3 = st.columns(3)
with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)
    serviced = st.checkbox("Serviced", value=False)
    property_type = st.selectbox("Property Type", options=PROPERTY_TYPES, index=PROPERTY_TYPES.index("Apartment") if "Apartment" in PROPERTY_TYPES else 0)
with col2:
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=3, step=1)
    newly_built = st.checkbox("Newly Built", value=False)
    city = st.text_input("City", value="Lagos")
with col3:
    toilets = st.number_input("Toilets", min_value=0, max_value=20, value=3, step=1)
    furnished = st.checkbox("Furnished", value=False)
    # Neighborhood dropdown populated from data if available
    if df_peek is not None and 'Neighborhood' in df_peek.columns:
        neighborhood_options = sorted([c for c in df_peek['Neighborhood'].dropna().astype(str).unique() if c.strip()])
        default_nbhd = 'Lekki' if 'Lekki' in neighborhood_options else neighborhood_options[0] if neighborhood_options else 'Unknown'
        neighborhood_choice = st.selectbox("Neighborhood", options=neighborhood_options + ["Other (custom)"] , index=(neighborhood_options.index('Lekki') if 'Lekki' in neighborhood_options else 0))
        if neighborhood_choice == "Other (custom)":
            neighborhood = st.text_input("Enter Custom Neighborhood", value="") or "Unknown"
        else:
            neighborhood = neighborhood_choice
    else:
        neighborhood = st.text_input("Neighborhood", value="Lekki")

# Removed separate quick-pick expander since Neighborhood now dropdown above.

input_row = pd.DataFrame([{ 
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "toilets": toilets,
    "Serviced": 1 if serviced else 0,
    "Newly Built": 1 if newly_built else 0,
    "Furnished": 1 if furnished else 0,
    "property_type": property_type,
    "City": city if city.strip() else "Unknown",
    "Neighborhood": neighborhood if neighborhood.strip() else "Unknown",
}])[FEATURE_COLUMNS]

if st.button("Predict Rent (₦)"):
    try:
        pred = model.predict(input_row)[0]
        # Guardrails: non-negative and formatted
        pred = max(0.0, float(pred))
        st.success(f"Estimated rent: ₦{pred:,.0f}")
        with st.expander("Model input sent", expanded=False):
            st.dataframe(input_row)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Tip: Better accuracy often comes from including location granularity and log-transforming prices; consider retraining with target log(₦) and more features like property size, period, and amenities.")
