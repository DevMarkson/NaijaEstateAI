"""Central configuration for NaijaEstateAI ML pipeline.

This keeps feature lists and default paths in one place so that
training and inference (Streamlit app) stay in sync.
"""
from pathlib import Path

# Paths
DATA_PATH = Path("lagos-rent.csv")
ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Target column after cleaning
TARGET_COLUMN = "price_ngn"  # numeric annual rent in NGN

# Raw dataset columns (original CSV)
RAW_COLS = [
    "Title","More Info","Price","Serviced","Newly Built","Furnished",
    "Bedrooms","Bathrooms","Toilets","City","Neighborhood"
]

# Engineered / modeling feature columns (must match Streamlit app expectations)
FEATURE_COLUMNS = [
    "bedrooms","bathrooms","toilets",
    "Serviced","Newly Built","Furnished",
    "property_type","City","Neighborhood"
]

CATEGORICAL_FEATURES = ["Serviced","Newly Built","Furnished","property_type","City","Neighborhood"]
NUMERIC_FEATURES = ["bedrooms","bathrooms","toilets"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Keywords to infer property type from title/more info
PROPERTY_TYPE_KEYWORDS = {
    "duplex": "Duplex",
    "semi": "Semi Detached",
    "detached": "Detached",
    "apartment": "Apartment",
    "flat": "Flat",
    "bungalow": "Bungalow",
    "terrace": "Terrace",
    "terraced": "Terraced",
    "mansion": "Mansion",
    "studio": "Studio",
    "mini": "Mini Flat",
    "self contain": "Self Contain",
    "maisonette": "Maisonette",
    "penthouse": "Penthouse",
}

DEFAULT_PROPERTY_TYPE = "Other"

# Models to train (name -> constructor kwargs)
MODEL_SPECS = {
    "LinearRegression": {"module": "sklearn.linear_model", "class": "LinearRegression", "init": {}},
    "RandomForest": {"module": "sklearn.ensemble", "class": "RandomForestRegressor", "init": {"n_estimators": 200, "random_state": RANDOM_STATE}},
    "XGBoost": {"module": "xgboost", "class": "XGBRegressor", "init": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": RANDOM_STATE, "tree_method": "hist"}},
}

# Some environments may not have xgboost; allow graceful skip
OPTIONAL_MODELS = {"XGBoost"}
