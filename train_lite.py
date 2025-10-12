"""Lightweight training script for NaijaEstateAI - optimized for Heroku's memory limits.

This is a simplified version that uses minimal memory and only trains LinearRegression.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import gc

# Configuration
DATA_PATH = "lagos-rent.csv"
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

FEATURE_COLUMNS = ['bedrooms', 'bathrooms', 'toilets', 'Serviced',
                   'Newly Built', 'Furnished', 'property_type', 'City', 'Neighborhood']
CATEGORICAL_FEATURES = ['property_type', 'City', 'Neighborhood']
NUMERIC_FEATURES = ['bedrooms', 'bathrooms',
                    'toilets', 'Serviced', 'Newly Built', 'Furnished']
TARGET_COLUMN = 'price_ngn'

PROPERTY_TYPE_KEYWORDS = {
    "duplex": "Duplex",
    "detached": "Detached House",
    "semi": "Semi-detached House",
    "terraced": "Terraced House",
    "bungalow": "Bungalow",
    "apartment": "Apartment",
    "flat": "Apartment",
    "mini": "Mini Flat",
    "studio": "Studio Apartment",
    "penthouse": "Penthouse"
}
DEFAULT_PROPERTY_TYPE = "Apartment"


def memory_efficient_load_and_clean(data_path: str) -> pd.DataFrame:
    """Load and clean data with minimal memory usage."""
    print(f"Loading data: {data_path}")

    # Read only needed columns to save memory
    try:
        # First, read just a few rows to see what columns exist
        sample = pd.read_csv(data_path, nrows=5)
        available_cols = sample.columns.tolist()
        print(f"Available columns: {len(available_cols)}")

        # Map columns we need
        col_mapping = {}
        for col in available_cols:
            lower_col = col.lower()
            if 'bedroom' in lower_col:
                col_mapping[col] = 'bedrooms'
            elif 'bathroom' in lower_col:
                col_mapping[col] = 'bathrooms'
            elif 'toilet' in lower_col:
                col_mapping[col] = 'toilets'
            elif 'price' in lower_col or 'amount' in lower_col:
                col_mapping[col] = 'price_ngn'
            elif col in ['Serviced', 'Newly Built', 'Furnished', 'City', 'Neighborhood']:
                col_mapping[col] = col
            elif 'title' in lower_col:
                col_mapping[col] = 'Title'

        # Read only the columns we need
        usecols = list(col_mapping.keys())
        df = pd.read_csv(data_path, usecols=usecols)
        df.rename(columns=col_mapping, inplace=True)

        print(f"Raw rows: {len(df)}")

        # Clean data with minimal memory usage
        # Convert price to numeric - handle formatted strings like "5,000,000/year"
        if 'price_ngn' in df.columns:
            def clean_price(price_str):
                if pd.isna(price_str):
                    return None
                # Convert to string and remove quotes, commas, '/year', etc.
                price_clean = str(price_str).replace('"', '').replace(',', '')
                price_clean = price_clean.replace(
                    '/year', '').replace('/month', '')
                price_clean = price_clean.strip()
                try:
                    return float(price_clean)
                except:
                    return None

            df['price_ngn'] = df['price_ngn'].apply(clean_price)

        # Convert numeric features - handle formats like "4 beds", "3 baths"
        for col in ['bedrooms', 'bathrooms', 'toilets']:
            if col in df.columns:
                def extract_number(val):
                    if pd.isna(val):
                        return 0
                    # Extract number from strings like "4 beds", "3 baths"
                    import re
                    match = re.search(r'(\d+)', str(val))
                    return int(match.group(1)) if match else 0

                df[col] = df[col].apply(extract_number).astype('int8')

        # Convert boolean features
        for col in ['Serviced', 'Newly Built', 'Furnished']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col], errors='coerce').fillna(0).astype('int8')
            else:
                df[col] = 0

        # Infer property type (simplified)
        def simple_infer_prop(row):
            title = str(row.get('Title', '')).lower()
            for key, val in PROPERTY_TYPE_KEYWORDS.items():
                if key in title:
                    return val
            return DEFAULT_PROPERTY_TYPE

        df['property_type'] = df.apply(simple_infer_prop, axis=1)

        # Fill missing categorical data
        for col in ['City', 'Neighborhood']:
            if col not in df.columns:
                df[col] = 'Unknown'
            else:
                df[col] = df[col].fillna('Unknown').astype('category')

        df['property_type'] = df['property_type'].astype('category')

        # Remove rows with missing target
        df = df.dropna(subset=['price_ngn'])

        # Remove reasonable outliers (not too aggressive)
        q5 = df['price_ngn'].quantile(0.05)
        q95 = df['price_ngn'].quantile(0.95)
        df = df[(df['price_ngn'] >= q5) & (df['price_ngn'] <= q95)]

        # Also remove obviously wrong values
        df = df[df['price_ngn'] > 100000]  # At least 100k Naira
        df = df[df['price_ngn'] < 100000000]  # Less than 100M Naira

        # Keep only needed columns
        final_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
        df = df[final_cols]

        # Debug: Show some statistics
        print(f"After cleaning: {len(df)} rows, columns: {list(df.columns)}")
        if len(df) > 0:
            print(
                f"Price range: {df['price_ngn'].min():,.0f} - {df['price_ngn'].max():,.0f} NGN")
            print(f"Price mean: {df['price_ngn'].mean():,.0f} NGN")
            print(f"Sample prices: {df['price_ngn'].head().tolist()}")

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def train_simple_model(df: pd.DataFrame):
    """Train a simple LinearRegression model with minimal memory usage."""
    print("Training LinearRegression model...")

    # Prepare features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Free memory
    del df, X, y
    gc.collect()

    # Build simple pipeline
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer([
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ("num", numeric_transformer, NUMERIC_FEATURES),
    ])

    # Create model pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(
        f"LinearRegression: RMSE={rmse:,.0f} MAE={mae:,.0f} MAPE={mape:.2f} R2={r2:.3f}")

    return model, {
        'model_name': 'LinearRegression',
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }


def main():
    """Main training function."""
    # Create directories
    MODELS_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    try:
        # Load and clean data
        df = memory_efficient_load_and_clean(DATA_PATH)

        # Train model
        model, metrics = train_simple_model(df)

        # Save model
        joblib.dump(model, BEST_MODEL_PATH)
        print(f"Model saved to {BEST_MODEL_PATH}")

        # Save metrics
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {METRICS_PATH}")

        print("Training completed successfully!")

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
