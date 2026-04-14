from __future__ import annotations

import csv
from datetime import datetime, timezone
from functools import lru_cache
import os
import json
from pathlib import Path
import time
from threading import Lock
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.preprocess import WINE_FEATURE_COLUMNS, read_csv_auto_sep


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
MODELS_DIR = ROOT / "models"
PRED_LOG_PATH = ROOT / "logs" / "predictions_wine.csv"
API_LOG_PATH = ROOT / "logs" / "api_events.csv"
DATASET_PATH = ROOT / "data" / "winequality-red.csv"


class PredictRequest(BaseModel):
    fixed_acidity: float = Field(..., alias="fixed acidity", description="Fixed acidity (g/dm^3)")
    volatile_acidity: float = Field(..., alias="volatile acidity", ge=0, description="Volatile acidity (g/dm^3)")
    citric_acid: float = Field(..., alias="citric acid", ge=0, description="Citric acid (g/dm^3)")
    residual_sugar: float = Field(..., alias="residual sugar", ge=0, description="Residual sugar (g/dm^3)")
    chlorides: float = Field(..., ge=0, description="Chlorides (g/dm^3)")
    free_sulfur_dioxide: float = Field(
        ..., alias="free sulfur dioxide", ge=0, description="Free sulfur dioxide (mg/dm^3)"
    )
    total_sulfur_dioxide: float = Field(
        ..., alias="total sulfur dioxide", ge=0, description="Total sulfur dioxide (mg/dm^3)"
    )
    density: float = Field(..., ge=0, description="Density (g/cm^3)")
    pH: float = Field(..., alias="pH", description="pH")
    sulphates: float = Field(..., ge=0, description="Sulphates (g/dm^3)")
    alcohol: float = Field(..., ge=0, description="Alcohol (% by volume)")

    # Pydantic v2
    model_config = {"populate_by_name": True, "extra": "forbid"}


class PredictResponse(BaseModel):
    prediction: int
    probability_good: float | None = None
    model_version: str
    timestamp_utc: str


app = FastAPI(title="VelvetVine API", version="0.1.0")
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

_stats_lock = Lock()
_stats = {
    "requests_total": 0,
    "predict_total": 0,
    "errors_total": 0,
    "predict_success_total": 0,
    "predict_error_total": 0,
    "latency_ms_count": 0,
    "latency_ms_sum": 0.0,
    "latency_ms_max": 0.0,
}


def _append_api_event(*, route: str, status_code: int, latency_ms: float, error: str | None) -> None:
    API_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not API_LOG_PATH.exists()
    with API_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp_utc", "route", "status_code", "latency_ms", "error"])
        w.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                route,
                status_code,
                round(float(latency_ms), 3),
                error,
            ]
        )


@app.middleware("http")
async def _timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    route = request.url.path
    status_code = 500
    err: str | None = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000.0
        with _stats_lock:
            _stats["requests_total"] += 1
            _stats["latency_ms_count"] += 1
            _stats["latency_ms_sum"] += latency_ms
            _stats["latency_ms_max"] = max(_stats["latency_ms_max"], latency_ms)
            if status_code >= 400:
                _stats["errors_total"] += 1
        _append_api_event(route=route, status_code=status_code, latency_ms=latency_ms, error=err)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(WEB_DIR / "index.html"))


def _available_versions() -> list[str]:
    versions: dict[int, str] = {}
    for p in MODELS_DIR.glob("model_v*.pkl"):
        name = p.stem
        if not name.startswith("model_v"):
            continue
        try:
            n = int(name.split("model_v", 1)[1])
        except Exception:
            continue
        versions[n] = f"v{n}"
    return [versions[n] for n in sorted(versions)]


def _resolve_active_version() -> str:
    env_v = (os.getenv("TERRAFLOW_MODEL_VERSION") or "").strip()
    if env_v:
        return env_v
    versions = _available_versions()
    return versions[-1] if versions else "v1"


def _artifact_paths(version: str) -> tuple[Path, Path]:
    v = version.strip()
    return (MODELS_DIR / f"model_{v}.pkl", MODELS_DIR / f"scaler_{v}.pkl")


def _info_path(version: str) -> Path:
    v = version.strip()
    return MODELS_DIR / f"model_info_{v}.json"


@lru_cache(maxsize=8)
def _load_artifacts(version: str):
    model_path, scaler_path = _artifact_paths(version)
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found for version {version}. Run `python -m src.train` to create them."
        )
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to load model artifacts for {version}: {e}") from e


@lru_cache(maxsize=1)
def _load_wine_dataset() -> list[dict]:
    """
    Load the wine dataset rows as dicts. Used only for demo sample presets in the UI.
    """
    df, _sep = read_csv_auto_sep(DATASET_PATH)
    # Normalize to the exact feature columns + quality if present
    cols = [*WINE_FEATURE_COLUMNS]
    if "quality" in df.columns:
        cols.append("quality")
    df = df[cols].dropna()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    return df.to_dict(orient="records")


_FEATURE_ATTR_MAP = {
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "chlorides": "chlorides",
    "free sulfur dioxide": "free_sulfur_dioxide",
    "total sulfur dioxide": "total_sulfur_dioxide",
    "density": "density",
    "pH": "pH",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
}


def _vectorize_request(req: PredictRequest) -> np.ndarray:
    """
    Convert validated request -> model input in the exact training feature order.
    """
    values: list[float] = []
    for feature_name in WINE_FEATURE_COLUMNS:
        attr = _FEATURE_ATTR_MAP.get(feature_name)
        if not attr:
            raise RuntimeError(f"Missing feature mapping for: {feature_name}")
        values.append(float(getattr(req, attr)))
    return np.array([values], dtype=float)


def _append_prediction_log(req: PredictRequest, prediction: int, probability_good: float | None) -> None:
    PRED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not PRED_LOG_PATH.exists()
    with PRED_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp_utc", "model_version", *WINE_FEATURE_COLUMNS, "prediction", "probability_good"])

        x = _vectorize_request(req)[0].tolist()
        w.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                _resolve_active_version(),
                *x,
                prediction,
                probability_good,
            ]
        )


@app.get("/health")
def health():
    active = _resolve_active_version()
    model_path, scaler_path = _artifact_paths(active)
    model_ready = model_path.exists() and scaler_path.exists()
    return {"status": "ok", "model_ready": model_ready}


@app.get("/metrics")
def metrics():
    with _stats_lock:
        snapshot = dict(_stats)
    avg = (
        snapshot["latency_ms_sum"] / snapshot["latency_ms_count"]
        if snapshot["latency_ms_count"]
        else 0.0
    )
    return {
        "requests_total": snapshot["requests_total"],
        "errors_total": snapshot["errors_total"],
        "predict_total": snapshot["predict_total"],
        "predict_success_total": snapshot["predict_success_total"],
        "predict_error_total": snapshot["predict_error_total"],
        "latency_ms": {"avg": round(avg, 3), "max": round(snapshot["latency_ms_max"], 3)},
    }


@app.get("/monitor")
def monitor(last_n: int = 200):
    last_n = max(1, min(int(last_n), 5000))

    with _stats_lock:
        snapshot = dict(_stats)

    recent: list[dict] = []
    if PRED_LOG_PATH.exists():
        try:
            with PRED_LOG_PATH.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                recent = list(reader)[-last_n:]
        except Exception:
            recent = []

    pred_ones = 0
    pred_zeros = 0
    prob_sum = 0.0
    prob_count = 0
    for row in recent:
        try:
            pred = int(float(row.get("prediction", "0")))
            if pred == 1:
                pred_ones += 1
            else:
                pred_zeros += 1
        except Exception:
            continue
        p = row.get("probability_good")
        if p not in (None, "", "None"):
            try:
                prob_sum += float(p)
                prob_count += 1
            except Exception:
                pass

    total_recent = pred_ones + pred_zeros
    pos_rate = (pred_ones / total_recent) if total_recent else None
    avg_prob = (prob_sum / prob_count) if prob_count else None

    avg_latency = (
        snapshot["latency_ms_sum"] / snapshot["latency_ms_count"]
        if snapshot["latency_ms_count"]
        else 0.0
    )

    return {
        "now_utc": datetime.now(timezone.utc).isoformat(),
        "active_model_version": _resolve_active_version(),
        "available_model_versions": _available_versions(),
        "latency_ms": {"avg": round(avg_latency, 3), "max": round(snapshot["latency_ms_max"], 3)},
        "errors_total": snapshot["errors_total"],
        "recent_predictions": {
            "n": total_recent,
            "pred_1_good": pred_ones,
            "pred_0_bad": pred_zeros,
            "pred_1_rate": round(pos_rate, 4) if pos_rate is not None else None,
            "avg_probability_good": round(avg_prob, 4) if avg_prob is not None else None,
        },
    }


@app.get("/model")
def model_status():
    active = _resolve_active_version()
    model_path, scaler_path = _artifact_paths(active)
    return {
        "active_version": active,
        "available_versions": _available_versions(),
        "active_artifacts_present": (model_path.exists() and scaler_path.exists()),
        "env_var": "TERRAFLOW_MODEL_VERSION",
    }


@app.get("/model-info")
def model_info():
    active = _resolve_active_version()
    info_path = _info_path(active)
    if not info_path.exists():
        raise HTTPException(status_code=404, detail=f"Model info not found for {active}")
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to read model info: {e}") from e


@app.get("/sample")
def sample(kind: Literal["premium", "balanced", "low"] = "balanced"):
    """
    Returns a sample payload (feature dict) derived from the real dataset.
    - premium: a row with quality >= 7
    - low: a row with quality <= 5
    - balanced: a row with quality == 6 (fallback to any row)
    """
    rows = _load_wine_dataset()
    if not rows:
        raise HTTPException(status_code=500, detail="Dataset not available for samples")

    premium_rows = [r for r in rows if r.get("quality", 0) >= 7]
    low_rows = [r for r in rows if r.get("quality", 0) <= 5]
    balanced_rows = [r for r in rows if r.get("quality", 0) == 6]

    if kind == "premium" and premium_rows:
        row = premium_rows[0]
    elif kind == "low" and low_rows:
        row = low_rows[0]
    elif kind == "balanced" and balanced_rows:
        row = balanced_rows[0]
    else:
        row = rows[0]

    payload = {k: float(row[k]) for k in WINE_FEATURE_COLUMNS}
    return {"kind": kind, "features": payload, "quality": int(row.get("quality", -1))}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    with _stats_lock:
        _stats["predict_total"] += 1
    active_version = _resolve_active_version()
    try:
        model, scaler = _load_artifacts(active_version)
    except FileNotFoundError as e:
        with _stats_lock:
            _stats["predict_error_total"] += 1
        raise HTTPException(status_code=503, detail=str(e)) from e
    except RuntimeError as e:
        with _stats_lock:
            _stats["predict_error_total"] += 1
        raise HTTPException(status_code=500, detail=str(e)) from e

    X = _vectorize_request(req)
    Xs = scaler.transform(X)
    pred = int(model.predict(Xs)[0])
    probability_good: float | None = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)
        probability_good = float(proba[0][1]) if proba is not None else None

    _append_prediction_log(req, pred, probability_good)
    with _stats_lock:
        _stats["predict_success_total"] += 1

    return PredictResponse(
        prediction=pred,
        probability_good=probability_good,
        model_version=active_version,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
