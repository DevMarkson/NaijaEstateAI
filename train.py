"""Training script for NaijaEstateAI.

Usage (basic):
    python train.py --data lagos-rent.csv

Outputs:
    models/best_model.joblib
    artifacts/metrics.json

Steps:
    1. Load & clean raw CSV
    2. Feature engineering (parse numeric counts, infer property type)
    3. Train/test split
    4. Train multiple models (Linear, RandomForest, XGBoost if available)
    5. Evaluate (RMSE, MAE, R2) & select best
    6. Persist best model + metrics
"""
from __future__ import annotations
import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor

from config import (
    DATA_PATH, ARTIFACTS_DIR, MODELS_DIR, BEST_MODEL_PATH, METRICS_PATH,
    FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    PROPERTY_TYPE_KEYWORDS, DEFAULT_PROPERTY_TYPE, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE, MODEL_SPECS, OPTIONAL_MODELS
)
from settings import settings

import joblib


def parse_args():
    p = argparse.ArgumentParser(description="Train NaijaEstateAI models")
    p.add_argument("--data", type=Path, default=DATA_PATH, help="Path to raw CSV")
    p.add_argument("--test-size", type=float, default=TEST_SIZE)
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument("--skip-xgb", action="store_true", help="Skip XGBoost even if installed")
    p.add_argument("--cv-folds", type=int, default=5, help="KFold splits for CV (reduce for speed)")
    p.add_argument("--max-rows", type=int, default=None, help="Optional row cap for fast experiments")
    return p.parse_args()


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Keep original columns for reference; create helper mapping with safe keys
    original_cols = {c: c for c in df.columns}
    # Parse price into numeric annual NGN. Many entries have "/year" but some not.
    def parse_price(s: str) -> float | None:
        if not isinstance(s, str):
            return None
        s = s.strip().replace(",", "")
        # remove currency symbols and trailing parts
        s = s.replace("₦", "")
        # If contains '/' keep only before
        if "/" in s:
            s = s.split("/")[0]
        # Extract digits
        import re
        m = re.search(r"(\d+)", s)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    price_col = original_cols.get("Price")
    df["price_ngn"] = df[price_col].apply(parse_price)

    # Bedrooms etc are like "4 beds"
    def parse_count(s: str) -> float | None:
        if not isinstance(s, str):
            return None
        import re
        m = re.search(r"(\d+)", s)
        if m:
            return float(m.group(1))
        return None

    df["bedrooms"] = df[original_cols.get("Bedrooms")].apply(parse_count)
    df["bathrooms"] = df[original_cols.get("Bathrooms")].apply(parse_count)
    df["toilets"] = df[original_cols.get("Toilets")].apply(parse_count)

    # Booleans already 0/1? Ensure numeric 0/1 ints.
    for raw_name, std_name in [("Serviced","Serviced"),("Newly Built","Newly Built"),("Furnished","Furnished")]:
        if raw_name in df.columns:
            df[std_name] = pd.to_numeric(df[raw_name], errors="coerce").fillna(0).astype(int)
        elif std_name in df.columns:
            df[std_name] = pd.to_numeric(df[std_name], errors="coerce").fillna(0).astype(int)
        else:
            df[std_name] = 0

    # Copy original City/Neighborhood (rename to match feature list capitalization)
    # Ensure City / Neighborhood exist (retain names used in FEATURE_COLUMNS)
    if "City" not in df.columns and "city" in df.columns:
        df.rename(columns={"city": "City"}, inplace=True)
    if "Neighborhood" not in df.columns and "neighborhood" in df.columns:
        df.rename(columns={"neighborhood": "Neighborhood"}, inplace=True)

    # Infer property_type from Title / More_Info
    def infer_prop(row):
        title_val = row.get("Title")
        more_val = row.get("More Info") or row.get("More_Info")
        for key, val in PROPERTY_TYPE_KEYWORDS.items():
            if isinstance(title_val, str) and key in title_val.lower():
                return val
            if isinstance(more_val, str) and key in more_val.lower():
                return val
        return DEFAULT_PROPERTY_TYPE

    df["property_type"] = df.apply(infer_prop, axis=1)

    # Keep only rows with target
    df = df[df["price_ngn"].notnull()]

    # Select modeling columns + target
    cols_needed = FEATURE_COLUMNS + [TARGET_COLUMN]
    return df[cols_needed]


def build_pipeline() -> ColumnTransformer:
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    preprocessor = ColumnTransformer([
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ("num", numeric_transformer, NUMERIC_FEATURES),
    ])
    return preprocessor


def compute_rmse(y_true, y_pred) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > 0
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]))


def load_model_class(spec: Dict[str, Any]):
    module = importlib.import_module(spec["module"])
    cls = getattr(module, spec["class"])
    return cls(**spec.get("init", {}))


def train_and_evaluate(X, y, X_train, X_test, y_train, y_test, preprocessor, skip_xgb=False, n_splits: int = 5):
    """Train models with log-target transform, evaluate on holdout and CV.

    Returns best model info and metrics dict:
        best_name, best_pipeline, results
    results[model] = {
        'rmse': holdout_rmse,
        'mae': holdout_mae,
        'r2': holdout_r2,
        'mape': holdout_mape,
        'cv': {metric_mean/std}
    }
    """
    results = {}
    best_name = None
    best_rmse = float("inf")
    best_pipeline = None

    do_cv = n_splits and n_splits > 1
    if do_cv:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for name, spec in MODEL_SPECS.items():
        if name in OPTIONAL_MODELS and skip_xgb:
            continue
        if name in OPTIONAL_MODELS:
            try:
                base_model = load_model_class(spec)
            except Exception:
                print(f"Skipping optional model {name} (import failed)")
                continue
        else:
            base_model = load_model_class(spec)

        # Log-transform target
        wrapped_model = TransformedTargetRegressor(
            regressor=base_model, func=np.log1p, inverse_func=np.expm1
        )

        pipe = Pipeline(steps=[
            ("preproc", preprocessor),
            ("model", wrapped_model)
        ])

        # Fit on train, evaluate holdout
        pipe.fit(X_train, y_train)
        holdout_pred = pipe.predict(X_test)
        rmse = compute_rmse(y_test, holdout_pred)
        mae = float(mean_absolute_error(y_test, holdout_pred))
        r2 = float(r2_score(y_test, holdout_pred))
        mape = compute_mape(y_test, holdout_pred)

        # Cross-validation on full dataset (could be expensive)
        cv_rmses, cv_maes, cv_r2s, cv_mapes = [], [], [], []
        if do_cv:
            for tr_idx, va_idx in kf.split(X):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
                cv_pipe = Pipeline(steps=[
                    ("preproc", preprocessor),
                    ("model", TransformedTargetRegressor(
                        regressor=load_model_class(spec), func=np.log1p, inverse_func=np.expm1
                    ))
                ])
                try:
                    cv_pipe.fit(X_tr, y_tr)
                    va_pred = cv_pipe.predict(X_va)
                    cv_rmses.append(compute_rmse(y_va, va_pred))
                    cv_maes.append(float(mean_absolute_error(y_va, va_pred)))
                    cv_r2s.append(float(r2_score(y_va, va_pred)))
                    cv_mapes.append(compute_mape(y_va, va_pred))
                except Exception as e:
                    print(f"CV fold failed for {name}: {e}")

        results[name] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "cv": {
                "rmse_mean": float(np.mean(cv_rmses)) if cv_rmses else None,
                "rmse_std": float(np.std(cv_rmses)) if cv_rmses else None,
                "mae_mean": float(np.mean(cv_maes)) if cv_maes else None,
                "mae_std": float(np.std(cv_maes)) if cv_maes else None,
                "r2_mean": float(np.mean(cv_r2s)) if cv_r2s else None,
                "r2_std": float(np.std(cv_r2s)) if cv_r2s else None,
                "mape_mean": float(np.mean(cv_mapes)) if cv_mapes else None,
                "mape_std": float(np.std(cv_mapes)) if cv_mapes else None,
            }
        }
        cv_rmse_mean = results[name]['cv']['rmse_mean']
        cv_rmse_std = results[name]['cv']['rmse_std']
        cv_rmse_mean_disp = f"{cv_rmse_mean:,.0f}" if cv_rmse_mean is not None else "nan"
        cv_rmse_std_disp = f"{cv_rmse_std:,.0f}" if cv_rmse_std is not None else "nan"
        print(f"{name}: HOLDOUT RMSE={rmse:,.0f} MAE={mae:,.0f} MAPE={mape:.2f} R2={r2:.3f} | CV RMSE={cv_rmse_mean_disp} ± {cv_rmse_std_disp}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_pipeline = pipe

    return best_name, best_pipeline, results


def main():
    args = parse_args()
    raw_path = args.data
    if not raw_path.exists():
        raise SystemExit(f"Data file not found: {raw_path}")

    print(f"Loading data: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    if args.max_rows:
        raw_df = raw_df.sample(n=min(args.max_rows, len(raw_df)), random_state=args.random_state)
    print(f"Raw rows: {len(raw_df)}")
    df = clean_raw(raw_df)
    # Outlier trimming
    lower_bound = 100000
    upper_bound = df['price_ngn'].quantile(0.99)
    before = len(df)
    df = df[(df['price_ngn'] >= lower_bound) & (df['price_ngn'] <= upper_bound)]
    removed = before - len(df)
    print(f"After cleaning: {len(df)} rows, columns: {list(df.columns)}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    preprocessor = build_pipeline()
    best_name, best_pipeline, results = train_and_evaluate(
        X, y, X_train, X_test, y_train, y_test, preprocessor,
        skip_xgb=args.skip_xgb, n_splits=args.cv_folds
    )

    # Baseline: median price by Neighborhood (fallback to global median)
    train_df = X_train.copy()
    train_df['target'] = y_train
    neighborhood_median = train_df.groupby('Neighborhood')['target'].median().to_dict()
    global_median = float(train_df['target'].median())
    baseline_preds = [neighborhood_median.get(nbhd, global_median) for nbhd in X_test['Neighborhood']]
    baseline_rmse = compute_rmse(y_test, baseline_preds)
    baseline_mae = float(mean_absolute_error(y_test, baseline_preds))
    baseline_r2 = float(r2_score(y_test, baseline_preds))
    baseline_mape = compute_mape(y_test, baseline_preds)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # Persist metrics
    metrics_payload = {
        "best": best_name,
        "results": results,  # holdout + cv per model
        "baseline": {
            "rmse": baseline_rmse,
            "mae": baseline_mae,
            "r2": baseline_r2,
            "mape": baseline_mape
        },
        "rows": len(df),
        "removed_outliers": int(removed),
        "price_bounds": {"min_kept": lower_bound, "p99_kept": float(upper_bound)},
        "test_size": args.test_size,
        "target_transform": "log1p",
        "model_version": settings.model_version
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics -> {METRICS_PATH}")

    if best_pipeline is None:
        raise SystemExit("No model trained successfully")

    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"Saved best model ({best_name}) -> {BEST_MODEL_PATH}")

    # Feature importance (only for models exposing feature_importances_)
    try:
        model_step = best_pipeline.named_steps.get("model")
        reg = getattr(model_step, 'regressor', None)
        inner = getattr(reg, 'regressor', reg)  # handle wrapped models
        if hasattr(inner, 'feature_importances_'):
            # Need expanded OHE feature names
            preproc = best_pipeline.named_steps['preproc']
            ohe = preproc.named_transformers_['cat'].named_steps['ohe']
            cat_features = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
            all_features = list(cat_features) + NUMERIC_FEATURES
            importances = inner.feature_importances_
            fi = sorted([
                {"feature": f, "importance": float(i)} for f, i in zip(all_features, importances)
            ], key=lambda x: x['importance'], reverse=True)
            fi_path = ARTIFACTS_DIR / 'feature_importance.json'
            with open(fi_path, 'w') as f:
                json.dump({"model": best_name, "feature_importance": fi}, f, indent=2)
            print(f"Saved feature importance -> {fi_path}")
    except Exception as e:
        print(f"Feature importance extraction skipped: {e}")


if __name__ == "__main__":
    main()
