"""FastAPI application exposing prediction endpoint for NaijaEstateAI.

Run locally:
    uvicorn api_app:app --reload --port 8000

Endpoints:
    GET /health            -> basic health check
    GET /model/info        -> model + metrics meta
    POST /predict          -> rent prediction
"""
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
import subprocess
import sys
import gc
import os

from config import BEST_MODEL_PATH, METRICS_PATH, FEATURE_COLUMNS
from settings import settings

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

app = FastAPI(title=settings.app_name, version=settings.model_version)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
REQUEST_COUNT = Counter("naijaestateai_requests_total", "Total API requests", [
                        "endpoint", "method", "status"])
REQUEST_LATENCY = Histogram(
    "naijaestateai_request_latency_seconds", "Request latency", ["endpoint", "method"])

_model = None
_metrics_cache = None


class PredictRequest(BaseModel):
    bedrooms: int = Field(ge=0, le=20)
    bathrooms: int = Field(ge=0, le=20)
    toilets: int = Field(ge=0, le=25)
    Serviced: int = Field(0, ge=0, le=1)
    Newly_Built: int = Field(0, ge=0, le=1)
    Furnished: int = Field(0, ge=0, le=1)
    property_type: str
    City: str
    Neighborhood: str


class PredictResponse(BaseModel):
    prediction: float
    rounded: int
    currency: str = "NGN"


class ModelInfo(BaseModel):
    best: Optional[str]
    rows: Optional[int]
    test_size: Optional[float]
    target_transform: Optional[str]
    model_version: Optional[str]
    feature_importance_available: bool = False


def load_model():
    global _model
    if _model is None:
        if not BEST_MODEL_PATH.exists():
            # Create necessary directories
            BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

            # Try to train the model automatically with memory optimization
            try:
                print("Model not found. Training model automatically...")
                # Use train_lite.py to avoid memory issues during auto-training
                result = subprocess.run([
                    sys.executable, "train_lite.py"
                ], capture_output=True, text=True, cwd=Path.cwd())

                if result.returncode != 0:
                    print(f"Training failed: {result.stderr}")
                    raise FileNotFoundError(
                        "Model not trained and auto-training failed.")

                print("Model training completed successfully!")

            except Exception as e:
                print(f"Auto-training error: {e}")
                raise FileNotFoundError(
                    "Model not trained. Run train.py first.")

        try:
            # Force garbage collection before loading model
            gc.collect()
            
            # Check available memory if possible
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                print(f"Available memory: {memory_info.available / 1024 / 1024:.1f} MB")
            except ImportError:
                print("psutil not available, cannot check memory")
            
            _model = joblib.load(BEST_MODEL_PATH)
            print(f"Model loaded successfully. Type: {type(_model)}")
            
            # Force garbage collection after loading
            gc.collect()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # If model loading fails, clear the global variable to prevent issues
            _model = None
            # Force cleanup
            gc.collect()
            raise FileNotFoundError("Model file corrupted or incompatible.")
    return _model


def load_metrics():
    global _metrics_cache
    if _metrics_cache is None and METRICS_PATH.exists():
        _metrics_cache = json.loads(METRICS_PATH.read_text())
    return _metrics_cache or {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    m = load_metrics()
    return {
        "best": m.get("best"),
        "rows": m.get("rows"),
        "test_size": m.get("test_size"),
        "target_transform": m.get("target_transform"),
        "model_version": settings.model_version,
        "feature_importance_available": bool(Path("artifacts/feature_importance.json").exists())
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Memory-efficient prediction that manages memory carefully."""
    start_time = time.time()
    
    try:
        # Force garbage collection before loading
        gc.collect()
        
        # Load model on demand (don't cache it permanently for memory efficiency)
        if not BEST_MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Load model temporarily
        model = joblib.load(BEST_MODEL_PATH)
        print(f"Model loaded for prediction. Type: {type(model)}")
        
        # Make prediction
        data = pd.DataFrame([{f: getattr(req, f if f != 'Newly Built' else 'Newly_Built')
                            if f != 'Newly Built' else req.Newly_Built for f in FEATURE_COLUMNS}])
        
        # Ensure consistent feature naming
        if 'Newly Built' in FEATURE_COLUMNS and 'Newly Built' not in data.columns:
            data['Newly Built'] = req.Newly_Built
        
        # Make prediction
        pred = float(model.predict(data)[0])
        
        # Clean up model from memory immediately after use
        del model
        gc.collect()
        
        # Record metrics
        REQUEST_COUNT.labels(endpoint="predict").inc()
        PREDICTION_TIME.observe(time.time() - start_time)
        
        return PredictResponse(prediction=pred, rounded=int(max(0, round(pred))), currency="NGN")
        
    except Exception as e:
        # Clean up on error
        gc.collect()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@app.get("/metrics")
def metrics():
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.get("/model/feature_importance")
def feature_importance():
    fi_path = Path("artifacts/feature_importance.json")
    if not fi_path.exists():
        raise HTTPException(
            status_code=404, detail="Feature importance not computed")
    return json.loads(fi_path.read_text())
