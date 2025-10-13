"""Ultra-lightweight training script for memory-constrained environments.

This version uses minimal features and aggressive memory management.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import gc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Memory monitoring
def log_memory():
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage: {memory_mb:.1f} MB")
    except ImportError:
        print("Memory monitoring not available")

def clean_price(price_str):
    """Extract numeric price from string format."""
    if pd.isna(price_str):
        return np.nan
    
    # Convert to string and clean
    price_str = str(price_str).lower().replace(',', '').replace('₦', '').replace('naira', '')
    
    # Remove non-numeric characters except decimal point
    cleaned = ''.join(c for c in price_str if c.isdigit() or c == '.')
    
    try:
        return float(cleaned) if cleaned else np.nan
    except (ValueError, TypeError):
        return np.nan

def minimal_train():
    """Train a minimal model with aggressive memory management."""
    
    print("Starting ultra-lightweight training...")
    log_memory()
    
    # Load data in chunks to save memory
    print("Loading data...")
    try:
        data = pd.read_csv("lagos-rent.csv", 
                          usecols=['Price', 'Bedrooms', 'Bathrooms', 'Location', 'Property Type'],
                          dtype={'Bedrooms': 'float32', 'Bathrooms': 'float32'})
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    print(f"Loaded {len(data)} rows")
    log_memory()
    
    # Clean price column
    print("Cleaning prices...")
    data['Price_Clean'] = data['Price'].apply(clean_price)
    data = data.dropna(subset=['Price_Clean'])
    
    # Remove extreme outliers (keep only 10th-90th percentile for memory efficiency)
    price_10 = data['Price_Clean'].quantile(0.1)
    price_90 = data['Price_Clean'].quantile(0.9)
    data = data[(data['Price_Clean'] >= price_10) & (data['Price_Clean'] <= price_90)]
    
    print(f"After cleaning: {len(data)} rows")
    log_memory()
    
    # Use only the most important features to save memory
    # Create simplified dummy variables
    print("Creating features...")
    
    # Simplified location encoding (only top locations)
    top_locations = data['Location'].value_counts().head(5).index
    for loc in top_locations:
        data[f'Location_{loc}'] = (data['Location'] == loc).astype('float32')
    
    # Simplified property type encoding
    data['IsApartment'] = data['Property Type'].str.contains('Apartment|apartment', na=False).astype('float32')
    data['IsHouse'] = data['Property Type'].str.contains('House|house|Duplex|duplex', na=False).astype('float32')
    
    # Select minimal feature set
    feature_cols = ['Bedrooms', 'Bathrooms', 'IsApartment', 'IsHouse'] + [f'Location_{loc}' for loc in top_locations]
    
    # Fill missing values with median
    for col in feature_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    # Final feature matrix
    X = data[feature_cols].fillna(0).astype('float32')
    y = data['Price_Clean'].astype('float32')
    
    print(f"Feature matrix shape: {X.shape}")
    log_memory()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use simple LinearRegression for minimal memory footprint
    print("Training LinearRegression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Clean up training data immediately
    del X_train, y_train
    gc.collect()
    log_memory()
    
    # Quick evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance - MAE: {mae:,.0f}, R²: {r2:.3f}")
    
    # Save model
    Path("models").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    
    joblib.dump(model, "models/best_model.joblib")
    
    # Save metrics
    metrics = {
        "best": "LinearRegression",
        "mae": float(mae),
        "r2": float(r2),
        "rows": len(data),
        "features": feature_cols,
        "test_size": 0.2,
        "target_transform": "none",
        "memory_optimized": True
    }
    
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("✅ Ultra-lightweight model saved successfully!")
    log_memory()
    
    # Final cleanup
    del model, X_test, y_test, y_pred, data
    gc.collect()
    
    return True

if __name__ == "__main__":
    success = minimal_train()
    if not success:
        exit(1)