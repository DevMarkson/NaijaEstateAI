# NaijaEstateAI

Machine Learning–driven Real Estate Management System focused on the Nigerian property market (starting with Lagos rent data). The goal is to improve transparency and pricing accuracy by:

* Cleaning and structuring fragmented listing data
* Predicting likely annual rent (₦) from property attributes
* Laying foundations for recommendation, trend analysis & geospatial insights

## Features (Current)
* Training script (`train.py`) trains Linear Regression, RandomForest, optionally XGBoost and auto-selects the best (RMSE)
* Persisted best model (`models/best_model.joblib`) + metrics JSON (`artifacts/metrics.json`)
* Streamlit app (`streamlit_app.py`) for interactive rent prediction
* Config centralization (`config.py`) keeping features & paths consistent
* Basic test (`tests/test_training.py`) to ensure training produces artifacts

## Roadmap (Planned)
1. Property recommendation engine (content + hybrid)
2. Market trend time-series module (price evolution by location)
3. Geospatial enrichment (lat/long, distance to POIs, clustering)
4. Explainability (SHAP values, feature importance visualization)
5. Data versioning & experiment tracking (DVC / MLflow)
6. Scheduled retraining workflow (GitHub Actions cron)
7. Deployment (Streamlit Cloud / Render) + API endpoints
8. Drift monitoring & simple analytics dashboard

## Setup

Install dependencies (ideally in a virtual environment):

```bash
pip install -r requirements.txt
```

## Train Models

```bash
python train.py --data lagos-rent.csv --test-size 0.2 --skip-xgb   # skip-xgb if environment lacks xgboost
```

Artifacts:
* `models/best_model.joblib`
* `artifacts/metrics.json`

## Run Streamlit App

```bash
streamlit run streamlit_app.py
```

## Run FastAPI Service (basic web API)

```bash
uvicorn api_app:app --reload --port 8000
```

Then test:

```bash
curl -X POST http://127.0.0.1:8000/predict \
	-H 'Content-Type: application/json' \
	-d '{
				"bedrooms":3,
				"bathrooms":3,
				"toilets":4,
				"Serviced":1,
				"Newly_Built":1,
				"Furnished":0,
				"property_type":"Apartment",
				"City":"Lagos",
				"Neighborhood":"Lekki"
			}'
```

Swagger UI: http://127.0.0.1:8000/docs

Prometheus Metrics (if enabled): http://127.0.0.1:8000/metrics

Feature Importance (if available): http://127.0.0.1:8000/model/feature_importance


### Makefile Shortcuts
```bash
make train      # quick training
make api        # run API with reload
make streamlit  # run Streamlit UI
make test       # run test suite
make docker-build && make docker-run
```

### Docker
```bash
docker build -t naijaestateai:latest .
docker run -p 8000:8000 naijaestateai:latest
```

### Environment Configuration
Configure via .env or environment variables:
```
LOG_LEVEL=info
ENABLE_METRICS=true
MODEL_VERSION=1.0.0
```

### Static Frontend
Simple static HTML page is in `static/index.html` (served if you put behind a web server or mount). For local dev you can open it and point fetch calls to the API origin if different.

## Next.js Frontend (Optional)

A React/Next.js client lives in `frontend/`.

Install & run (in separate shell):
```bash
cd frontend
npm install   # or yarn / pnpm
npm run dev
```

It defaults to calling `http://localhost:8000` for the API. Override:
```bash
export NEXT_PUBLIC_API_BASE="https://your-api-host"
npm run dev
```

Production build:
```bash
npm run build && npm start
```

The page provides a form for predictions and a cURL snippet generator.

## Testing

```bash
pytest -q
```

## Data Notes
Raw Lagos rent dataset columns are parsed into engineered features:
* Numerical: bedrooms, bathrooms, toilets
* Binary: Serviced, Newly Built, Furnished
* Categorical: property_type (inferred), City, Neighborhood

Target: annual rent price (₦) converted to numeric (price_ngn).

## Contributing
Open an issue / PR with improvements (e.g., better feature extraction, additional data sources, or deployment scripts).

## License
MIT License
