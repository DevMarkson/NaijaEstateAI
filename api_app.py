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
REQUEST_COUNT = Counter("naijaestateai_requests_total", "Total API requests", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram("naijaestateai_request_latency_seconds", "Request latency", ["endpoint", "method"])

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
            raise FileNotFoundError("Model not trained. Run train.py first.")
        _model = joblib.load(BEST_MODEL_PATH)
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
    model = load_model()
    data = pd.DataFrame([{f: getattr(req, f if f != 'Newly Built' else 'Newly_Built') if f != 'Newly Built' else req.Newly_Built for f in FEATURE_COLUMNS}])
    # Ensure New Feature column naming consistent
    if 'Newly Built' in FEATURE_COLUMNS and 'Newly Built' not in data.columns:
        data['Newly Built'] = req.Newly_Built
    try:
        pred = float(model.predict(data)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    return PredictResponse(prediction=pred, rounded=int(max(0, round(pred))), currency="NGN")


@app.get("/metrics")
def metrics():
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.get("/model/feature_importance")
def feature_importance():
    fi_path = Path("artifacts/feature_importance.json")
    if not fi_path.exists():
        raise HTTPException(status_code=404, detail="Feature importance not computed")
    return json.loads(fi_path.read_text())
