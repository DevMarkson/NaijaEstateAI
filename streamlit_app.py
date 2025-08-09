import os
import json
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join("models", "best_model.joblib")
METRICS_PATH = os.path.join("artifacts", "metrics.json")
DATA_PATH = "lagos-rent.csv"

# Keep in sync with training features
FEATURE_COLUMNS = [
    "bedrooms", "bathrooms", "toilets",
    "Serviced", "Newly Built", "Furnished",
    "property_type", "City", "Neighborhood",
]

PROPERTY_TYPES = [
    "Duplex", "Semi Detached", "Detached", "Apartment", "Flat",
    "Bungalow", "Terrace", "Terraced", "Mansion", "Studio",
    "Penthouse", "Mini Flat", "Self Contain", "Maisonette",
    "Terrace Duplex", "Terraced Duplex", "House", "Other",
]

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_data_peek(n=5000):
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH, nrows=n)
            # Normalize columns (match training)
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
        st.json(metrics.get("results", {}))

if model is None:
    st.error("No trained model found. Please run the training notebook to produce models/best_model.joblib.")
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
    neighborhood = st.text_input("Neighborhood", value="Lekki")

# Optional: quick-pick from data
if df_peek is not None and {"City", "Neighborhood"}.issubset(set(df_peek.columns)):
    with st.expander("Quick-pick from dataset (optional)"):
        cities = sorted([c for c in df_peek["City"].dropna().astype(str).unique() if c and c.strip()][:1000])
        nbhds = sorted([c for c in df_peek["Neighborhood"].dropna().astype(str).unique() if c and c.strip()][:1000])
        c1, c2 = st.columns(2)
        with c1:
            pick_city = st.selectbox("Pick City", options=[city] + [c for c in cities if c != city])
        with c2:
            pick_nbhd = st.selectbox("Pick Neighborhood", options=[neighborhood] + [c for c in nbhds if c != neighborhood])
        if st.button("Use quick-picks"):
            city = pick_city
            neighborhood = pick_nbhd
            st.success("Applied selections.")

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
