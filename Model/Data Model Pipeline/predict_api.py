from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from feature_engineering import FeatureSpec, build_features_from_months


# -----------------------------
# Paths / artifacts
# -----------------------------
ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

MODEL_PATH = ART / "best_cluster_model.pkl"
NORM_PATH = ART / "norm_obj.pkl"
OUTLIER_PATH = ART / "outlier_bounds.pkl"
SAMPLER_PATH = ART / "sampler_obj.pkl"
SCALER_PATH = ART / "scaler_obj.pkl"
STANDARD_PATH = ART / "standardization_obj.pkl"
FEATURE_COLS_PATH = ART / "feature_cols.json"
PIPELINE_CFG_PATH = ART / "pipeline_config.json"


# -----------------------------
# Load all 8 artifacts
# -----------------------------
model = joblib.load(MODEL_PATH)
norm_obj = joblib.load(NORM_PATH)
outlier_bounds = joblib.load(OUTLIER_PATH)          # may be unused if config says "none"
sampler_obj = joblib.load(SAMPLER_PATH)             # training-only, still loaded
scaler_obj = joblib.load(SCALER_PATH)
standardization_obj = joblib.load(STANDARD_PATH)

feature_cols = json.loads(FEATURE_COLS_PATH.read_text(encoding="utf-8"))
pipeline_config = json.loads(PIPELINE_CFG_PATH.read_text(encoding="utf-8"))

spec = FeatureSpec(feature_cols=feature_cols)


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="FinGrowth Local Inference API", version="1.0")

# CORS for local dev (webpack / react dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictIn(BaseModel):
    months: list[dict[str, Any]]


class PredictOut(BaseModel):
    top: int
    probs: list[float]


def _maybe_clip_outliers(X: np.ndarray) -> np.ndarray:
    """
    pipeline_config.json says outlier_method = "none" (current config),
    so this is intentionally a no-op by default. Still keep the hook,
    so outlier_bounds.pkl is involved.
    """
    method = str(pipeline_config.get("outlier_method", "none")).lower()
    if method == "none":
        return X

    # If switch outlier_method away from "none", implement the chosen method here.
    # For now: optional simple clipping if outlier_bounds.
    if isinstance(outlier_bounds, dict):
        Xc = X.copy()
        for j, col in enumerate(feature_cols):
            b = outlier_bounds.get(col, None)
            if b is None:
                continue
            lo = None
            hi = None
            if isinstance(b, (list, tuple)) and len(b) >= 2:
                lo, hi = b[0], b[1]
            elif isinstance(b, dict):
                lo = b.get("lo", None)
                hi = b.get("hi", None)
            if lo is not None:
                Xc[:, j] = np.maximum(Xc[:, j], float(lo))
            if hi is not None:
                Xc[:, j] = np.minimum(Xc[:, j], float(hi))
        return Xc

    return X


def _transform_features(X_df) -> np.ndarray:
    """
    Apply transforms in the same order as trained/saved them.
    To involve all artifacts, run:
      norm_obj -> standardization_obj -> scaler_obj
    """
    X = X_df.to_numpy(dtype=float)

    X = _maybe_clip_outliers(X)

    # norm (sqrt per pipeline_config)
    if hasattr(norm_obj, "transform"):
        X = norm_obj.transform(X)

    # standardization (pipeline_config says none, but object exists)
    if hasattr(standardization_obj, "transform"):
        X = standardization_obj.transform(X)

    # scaler (robust per pipeline_config)
    if hasattr(scaler_obj, "transform"):
        X = scaler_obj.transform(X)

    return X


@app.get("/health")
def health():
    return {
        "ok": True,
        "artifacts_found": {
            "best_cluster_model.pkl": MODEL_PATH.exists(),
            "norm_obj.pkl": NORM_PATH.exists(),
            "outlier_bounds.pkl": OUTLIER_PATH.exists(),
            "sampler_obj.pkl": SAMPLER_PATH.exists(),
            "scaler_obj.pkl": SCALER_PATH.exists(),
            "standardization_obj.pkl": STANDARD_PATH.exists(),
            "feature_cols.json": FEATURE_COLS_PATH.exists(),
            "pipeline_config.json": PIPELINE_CFG_PATH.exists(),
        },
        "pipeline_config": pipeline_config,
        "n_features": len(feature_cols),
    }


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    # 1) UI months -> engineered feature vector
    X_df = build_features_from_months(payload.months, spec)

    # 2) Apply transforms
    X = _transform_features(X_df)

    # 3) Predict probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    else:
        # fallback: some models only have decision_function
        if hasattr(model, "decision_function"):
            logits = model.decision_function(X)
            # Convert to probabilities (softmax)
            logits = np.array(logits).reshape(-1)
            ex = np.exp(logits - np.max(logits))
            probs = ex / np.sum(ex)
        else:
            # final fallback: hard class only
            pred = int(model.predict(X)[0])
            probs = np.zeros(6, dtype=float)
            probs[pred] = 1.0

    probs = np.array(probs, dtype=float).reshape(-1)
    top = int(np.argmax(probs))

    return PredictOut(top=top, probs=probs.tolist())
