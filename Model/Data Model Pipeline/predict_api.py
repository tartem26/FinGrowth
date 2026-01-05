from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
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
SAMPLER_PATH = ART / "sampler_obj.pkl"  # training-only, ok if missing deps
SCALER_PATH = ART / "scaler_obj.pkl"
STANDARD_PATH = ART / "standardization_obj.pkl"
FEATURE_COLS_PATH = ART / "feature_cols.json"
PIPELINE_CFG_PATH = ART / "pipeline_config.json"

# Cache files
CLUSTER_CACHE_JOBLIB = ART / "cluster_space_cache.joblib"
CLUSTER_CACHE_JSON = ART / "cluster_space_cache.json"
CLUSTER_CACHE_PKL = ART / "cluster_space_cache.pkl"  # allow older name too


# -----------------------------
# Safe loaders
# -----------------------------
def _safe_load(path: Path, *, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing artifact: {path}")
        return None, f"Optional artifact missing: {path.name}"
    try:
        return joblib.load(path), None
    except Exception as e:
        if required:
            raise
        return None, f"Optional artifact failed to load ({path.name}): {type(e).__name__}: {e}"


def _safe_read_json(path: Path, *, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing artifact: {path}")
        return None, f"Optional artifact missing: {path.name}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as e:
        if required:
            raise
        return None, f"Optional artifact failed to read ({path.name}): {type(e).__name__}: {e}"


def _load_cluster_cache() -> Tuple[Optional[dict], Optional[str]]:
    """
    Supported:
      - cluster_space_cache.joblib (recommended)
      - cluster_space_cache.pkl
      - cluster_space_cache.json
    """
    if CLUSTER_CACHE_JOBLIB.exists():
        return _safe_load(CLUSTER_CACHE_JOBLIB, required=False)
    if CLUSTER_CACHE_PKL.exists():
        return _safe_load(CLUSTER_CACHE_PKL, required=False)
    if CLUSTER_CACHE_JSON.exists():
        return _safe_read_json(CLUSTER_CACHE_JSON, required=False)
    return None, "Optional artifact missing: cluster_space_cache.(joblib|pkl|json)"


# -----------------------------
# Load artifacts
# -----------------------------
model, _ = _safe_load(MODEL_PATH, required=True)
norm_obj, _ = _safe_load(NORM_PATH, required=True)
outlier_bounds, _ = _safe_load(OUTLIER_PATH, required=True)
scaler_obj, _ = _safe_load(SCALER_PATH, required=True)
standardization_obj, _ = _safe_load(STANDARD_PATH, required=True)

sampler_obj, sampler_warn = _safe_load(SAMPLER_PATH, required=False)

cluster_cache, cluster_cache_warn = _load_cluster_cache()

feature_cols = json.loads(FEATURE_COLS_PATH.read_text(encoding="utf-8"))
pipeline_config = json.loads(PIPELINE_CFG_PATH.read_text(encoding="utf-8"))
spec = FeatureSpec(feature_cols=feature_cols)


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="FinGrowth Local Inference API", version="2.2")

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


# -----------------------------
# Schemas
# -----------------------------
class PredictIn(BaseModel):
    months: list[dict[str, Any]]


class RadarPoint(BaseModel):
    metric: str
    v: float


class NetworthPoint(BaseModel):
    month: int
    networth: float


class ScatterPoint(BaseModel):
    x: float
    y: float
    k: int
    isUser: bool = False


class ClusterSpace(BaseModel):
    points: list[ScatterPoint]
    user_point: Optional[ScatterPoint] = None


class Conclusion(BaseModel):
    text: str
    drivers: list[str] = []
    missing_fields: list[str] = []


class PredictOut(BaseModel):
    top: int
    probs: list[float]
    radars: Optional[dict[str, list[RadarPoint]]] = None
    networth: Optional[list[NetworthPoint]] = None
    cluster_space: Optional[ClusterSpace] = None
    conclusion: Optional[Conclusion] = None
    warnings: list[str] = []


# -----------------------------
# Helpers: input parsing
# -----------------------------
UI_CATEGORIES_17 = [
    "Income_Deposits",
    "Housing",
    "Utilities_Telecom",
    "Groceries_FoodAtHome",
    "Dining_FoodAway",
    "Transportation_Variable",
    "Auto_Costs",
    "Healthcare_OOP",
    "Insurance_All",
    "Debt_Payments",
    "Savings_Investments",
    "Education_Childcare",
    "Entertainment",
    "Subscriptions_Memberships",
    "Cash_ATM_MiscTransfers",
    "Pets",
    "Travel",
]


def _to_float_or_nan(x: Any) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _build_month_df(months: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str], list[str]]:
    warnings: list[str] = []
    missing_fields: list[str] = []

    if not isinstance(months, list) or len(months) == 0:
        months = [{}]
        warnings.append("No months provided; using a single empty month (this will reduce accuracy).")

    df = pd.DataFrame(months)

    for k in UI_CATEGORIES_17:
        if k not in df.columns:
            df[k] = np.nan
        df[k] = df[k].map(_to_float_or_nan)

    for k in UI_CATEGORIES_17:
        miss = int(df[k].isna().sum())
        if miss > 0:
            missing_fields.append(f"{k} (missing in {miss}/{len(df)} months)")

    if df["Income_Deposits"].isna().all():
        warnings.append("Income_Deposits is missing for all months — prediction will be unreliable.")
    else:
        n_zero_or_missing_income = int((df["Income_Deposits"].fillna(0) <= 0).sum())
        if n_zero_or_missing_income > 0:
            warnings.append(
                f"{n_zero_or_missing_income}/{len(df)} months have Income_Deposits ≤ 0 (shares / rates may collapse)."
            )

    return df, warnings, missing_fields


# -----------------------------
# Inference transforms
#   norm -> outlier_clip -> scaler -> standardization
# -----------------------------
def _coerce_finite(X: np.ndarray) -> tuple[np.ndarray, int]:
    bad = ~np.isfinite(X)
    n_bad = int(bad.sum())
    if n_bad:
        X = X.copy()
        X[bad] = 0.0
    return X, n_bad


def _apply_norm(X: np.ndarray, warnings: list[str]) -> np.ndarray:
    method = str(pipeline_config.get("norm_method", "none") or "none").lower().strip()
    if method in ("none", "", "null"):
        return X

    if method == "log":
        return np.log1p(np.clip(X, 0, None))
    if method == "sqrt":
        return np.sqrt(np.clip(X, 0, None))
    if method == "square":
        return np.square(X)
    if method == "inverse":
        eps = 1e-3
        return 1.0 / (np.clip(X, 0, None) + eps)

    if method == "zscore":
        # zscore requires fitted stats; typically handled by scaler_obj
        warnings.append("norm_method=zscore requested; skipping here (use scaler_obj stage).")
        return X

    if method in ("yeo_johnson", "quantile_normal"):
        if not hasattr(norm_obj, "transform"):
            raise RuntimeError(f"norm_method={method} but norm_obj has no transform().")
        return norm_obj.transform(X)

    warnings.append(f"Unknown norm_method='{method}'; skipping.")
    return X


def _apply_outlier_clip(X: np.ndarray) -> np.ndarray:
    method = str(pipeline_config.get("outlier_method", "none") or "none").lower().strip()
    if method in ("none", "", "null"):
        return X

    if not isinstance(outlier_bounds, dict):
        return X

    # Format A: {"lo": {...}, "hi": {...}}
    if "lo" in outlier_bounds and "hi" in outlier_bounds:
        lo = pd.Series(outlier_bounds["lo"])
        hi = pd.Series(outlier_bounds["hi"])
        Xdf = pd.DataFrame(X, columns=feature_cols)
        common = [c for c in feature_cols if c in lo.index and c in hi.index]
        if common:
            Xdf[common] = Xdf[common].clip(lower=lo[common], upper=hi[common], axis=1)
        return Xdf.to_numpy(dtype=float)

    # Format B: bounds per feature col
    Xc = X.copy()
    for j, col in enumerate(feature_cols):
        b = outlier_bounds.get(col, None)
        if b is None:
            continue

        lo = hi = None
        if isinstance(b, (list, tuple)) and len(b) >= 2:
            lo, hi = b[0], b[1]
        elif isinstance(b, dict):
            lo = b.get("lo", None)
            hi = b.get("hi", None)

        try:
            if lo is not None and np.isfinite(float(lo)):
                Xc[:, j] = np.maximum(Xc[:, j], float(lo))
            if hi is not None and np.isfinite(float(hi)):
                Xc[:, j] = np.minimum(Xc[:, j], float(hi))
        except Exception:
            continue

    return Xc


def _transform_features(X_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    MUST match training:
      1) norm_method
      2) outlier_method
      3) scaler_method (fitted scaler_obj)
      4) standardization_method (fitted standardization_obj)
    """
    warnings: list[str] = []
    X = X_df.to_numpy(dtype=float)

    # check everything first
    X, n_bad = _coerce_finite(X)
    if n_bad:
        warnings.append(f"Coerced {n_bad} non-finite values (NaN/Inf) to 0 before transforms.")

    # 1) norm
    X = _apply_norm(X, warnings)

    # 2) outlier clipping
    X = _apply_outlier_clip(X)

    # 3) scaler
    scaler_method = str(pipeline_config.get("scaler_method", "none") or "none").lower().strip()
    if scaler_method not in ("none", "", "null"):
        if not hasattr(scaler_obj, "transform"):
            raise RuntimeError("scaler_method is set but scaler_obj has no transform().")
        X = scaler_obj.transform(X)

    # 4) standardization (row-wise normalization / fitted normalizer)
    std_method = str(pipeline_config.get("standardization_method", "none") or "none").lower().strip()
    if std_method not in ("none", "", "null"):
        if not hasattr(standardization_obj, "transform"):
            raise RuntimeError("standardization_method is set but standardization_obj has no transform().")
        X = standardization_obj.transform(X)

    # final safety check
    X, n_bad2 = _coerce_finite(np.asarray(X, dtype=float))
    if n_bad2:
        warnings.append(f"Coerced {n_bad2} non-finite values (NaN/Inf) to 0 after transforms.")

    return np.asarray(X, dtype=float), warnings


# -----------------------------
# Model prob helper
# -----------------------------
def _predict_probs(X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        return np.array(probs, dtype=float).reshape(-1)

    if hasattr(model, "decision_function"):
        logits = model.decision_function(X)
        logits = np.array(logits, dtype=float).reshape(-1)
        ex = np.exp(logits - np.max(logits))
        probs = ex / np.sum(ex)
        return probs

    pred = int(model.predict(X)[0])
    probs = np.zeros(6, dtype=float)
    probs[pred] = 1.0
    return probs


# -----------------------------
# Backend charts
# -----------------------------
def _clamp01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(max(0.0, min(1.0, x)))


def _score_0_100(x01: float) -> float:
    return 100.0 * _clamp01(x01)


def _radar_from_month_df(df: pd.DataFrame) -> dict[str, list[dict[str, float]]]:
    """
    IMPORTANT: keys match frontend consumption:
      radars.snapshot / radars.trend / radars.risk / radars.growth

    Definitions:
      - snapshot: latest month only (level)
      - trend: mean level of last W_trend months
      - risk: conservative stability level across ALL months using (mean - std) of the score series
      - growth: momentum; compares last W_growth months vs the W_growth months before that (50 = no change)
    """
    inc = df["Income_Deposits"].to_numpy(dtype=float)

    # Sum of all outflows (includes savings and debt as uses of money)
    outflows = np.zeros_like(inc)
    for k in UI_CATEGORIES_17:
        if k == "Income_Deposits":
            continue
        outflows += np.nan_to_num(df[k].to_numpy(dtype=float), nan=0.0)

    # Pull categories (NaN -> 0 for amounts)
    housing = np.nan_to_num(df["Housing"].to_numpy(dtype=float), nan=0.0)
    utilities = np.nan_to_num(df["Utilities_Telecom"].to_numpy(dtype=float), nan=0.0)
    groceries = np.nan_to_num(df["Groceries_FoodAtHome"].to_numpy(dtype=float), nan=0.0)
    trans = np.nan_to_num(df["Transportation_Variable"].to_numpy(dtype=float), nan=0.0)
    auto = np.nan_to_num(df["Auto_Costs"].to_numpy(dtype=float), nan=0.0)
    ins = np.nan_to_num(df["Insurance_All"].to_numpy(dtype=float), nan=0.0)
    med = np.nan_to_num(df["Healthcare_OOP"].to_numpy(dtype=float), nan=0.0)
    edu = np.nan_to_num(df["Education_Childcare"].to_numpy(dtype=float), nan=0.0)
    pets = np.nan_to_num(df["Pets"].to_numpy(dtype=float), nan=0.0)

    dining = np.nan_to_num(df["Dining_FoodAway"].to_numpy(dtype=float), nan=0.0)
    ent = np.nan_to_num(df["Entertainment"].to_numpy(dtype=float), nan=0.0)
    subs = np.nan_to_num(df["Subscriptions_Memberships"].to_numpy(dtype=float), nan=0.0)
    travel = np.nan_to_num(df["Travel"].to_numpy(dtype=float), nan=0.0)
    cash = np.nan_to_num(df["Cash_ATM_MiscTransfers"].to_numpy(dtype=float), nan=0.0)

    debt = np.nan_to_num(df["Debt_Payments"].to_numpy(dtype=float), nan=0.0)
    savings = np.nan_to_num(df["Savings_Investments"].to_numpy(dtype=float), nan=0.0)

    eps = 1e-9
    # income must be positive to define rates; otherwise drop that month via NaN
    inc_pos = np.where(np.isfinite(inc) & (inc > 0), inc, np.nan)

    # ---- Rates (0..~) ----
    # Broader definitions so radar is visually interpretable
    essentials_amt = housing + utilities + groceries + trans + auto + ins + med + edu + pets
    discretionary_amt = dining + ent + subs + travel + cash

    essentials_rate = essentials_amt / (inc_pos + eps)
    debt_rate = debt / (inc_pos + eps)
    savings_rate = savings / (inc_pos + eps)
    discretionary_rate = discretionary_amt / (inc_pos + eps)

    # Net flow here means: remaining share after ALL outflows (including savings and debt)
    net_flow_rate = (inc - outflows) / (inc_pos + eps)

    # ---- Scores in [0,1], higher = better ----
    net_flow_score = np.clip((net_flow_rate + 1.0) / 2.0, 0.0, 1.0)
    essentials_score = np.clip(1.0 - np.clip(essentials_rate, 0.0, 1.0), 0.0, 1.0)
    debt_score = np.clip(1.0 - np.clip(debt_rate, 0.0, 1.0), 0.0, 1.0)
    discretionary_score = np.clip(1.0 - np.clip(discretionary_rate, 0.0, 1.0), 0.0, 1.0)
    savings_score = np.clip(np.clip(savings_rate, 0.0, 1.0), 0.0, 1.0)

    def _safe_mean(a: np.ndarray) -> float:
        return float(np.nanmean(a)) if a.size else 0.0

    def _safe_std(a: np.ndarray) -> float:
        # population std (ddof=0)
        return float(np.nanstd(a, ddof=0)) if a.size > 1 else 0.0

    def pack_level(vals01) -> list[dict[str, float]]:
        # vals01: list of 5 values in [0,1]
        return [
            {"metric": "Essentials", "v": float(_score_0_100(vals01[0]))},
            {"metric": "Debt", "v": float(_score_0_100(vals01[1]))},
            {"metric": "Savings", "v": float(_score_0_100(vals01[2]))},
            {"metric": "Fun", "v": float(_score_0_100(vals01[3]))},
            {"metric": "Left", "v": float(_score_0_100(vals01[4]))},
        ]

    def pack_growth(deltas01) -> list[dict[str, float]]:
        """
        deltas01 are in [-1, +1] (since score is [0,1]).
        Map to [0,100] with 50 = no change.
        """
        def to_0_100(delta: float) -> float:
            if not np.isfinite(delta):
                delta = 0.0
            v = 50.0 + 50.0 * float(delta)
            return float(np.clip(v, 0.0, 100.0))

        return [
            {"metric": "Essentials", "v": to_0_100(deltas01[0])},
            {"metric": "Debt", "v": to_0_100(deltas01[1])},
            {"metric": "Savings", "v": to_0_100(deltas01[2])},
            {"metric": "Fun", "v": to_0_100(deltas01[3])},
            {"metric": "Left", "v": to_0_100(deltas01[4])},
        ]

    n = len(df)

    # ---------- Snapshot: latest month ----------
    if n > 0:
        i = n - 1
        snapshot = pack_level([
            essentials_score[i],
            debt_score[i],
            savings_score[i],
            discretionary_score[i],
            net_flow_score[i],
        ])
    else:
        snapshot = pack_level([0, 0, 0, 0, 0])

    # Window sizes (configurable)
    W_trend = int(pipeline_config.get("radar_trend_window", 3))
    W_growth = int(pipeline_config.get("radar_growth_window", 3))

    w_tr = min(max(W_trend, 1), n) if n else 0
    w_g = max(W_growth, 1)

    # ---------- Trend: recent average (last w_tr months) ----------
    if w_tr > 0:
        trend = pack_level([
            _safe_mean(essentials_score[-w_tr:]),
            _safe_mean(debt_score[-w_tr:]),
            _safe_mean(savings_score[-w_tr:]),
            _safe_mean(discretionary_score[-w_tr:]),
            _safe_mean(net_flow_score[-w_tr:]),
        ])
    else:
        trend = pack_level([0, 0, 0, 0, 0])

    # ---------- Risk: stability across all months ----------
    # Conservative level = mean(score) - std(score)
    if n > 0:
        risk_vals = [
            float(np.clip(_safe_mean(essentials_score) - _safe_std(essentials_score), 0.0, 1.0)),
            float(np.clip(_safe_mean(debt_score) - _safe_std(debt_score), 0.0, 1.0)),
            float(np.clip(_safe_mean(savings_score) - _safe_std(savings_score), 0.0, 1.0)),
            float(np.clip(_safe_mean(discretionary_score) - _safe_std(discretionary_score), 0.0, 1.0)),
            float(np.clip(_safe_mean(net_flow_score) - _safe_std(net_flow_score), 0.0, 1.0)),
        ]
        risk = pack_level(risk_vals)
    else:
        risk = pack_level([0, 0, 0, 0, 0])

    # ---------- Growth: momentum ----------
    # Compare last w_g months vs the w_g months before them (if possible).
    # Otherwise fall back to last month vs first month.
    if n >= 2 * w_g:
        recent = (
            _safe_mean(essentials_score[-w_g:]),
            _safe_mean(debt_score[-w_g:]),
            _safe_mean(savings_score[-w_g:]),
            _safe_mean(discretionary_score[-w_g:]),
            _safe_mean(net_flow_score[-w_g:]),
        )
        prior = (
            _safe_mean(essentials_score[-2 * w_g: -w_g]),
            _safe_mean(debt_score[-2 * w_g: -w_g]),
            _safe_mean(savings_score[-2 * w_g: -w_g]),
            _safe_mean(discretionary_score[-2 * w_g: -w_g]),
            _safe_mean(net_flow_score[-2 * w_g: -w_g]),
        )
        deltas = [recent[i] - prior[i] for i in range(5)]
        growth = pack_growth(deltas)
    elif n >= 2:
        deltas = [
            float(essentials_score[-1] - essentials_score[0]),
            float(debt_score[-1] - debt_score[0]),
            float(savings_score[-1] - savings_score[0]),
            float(discretionary_score[-1] - discretionary_score[0]),
            float(net_flow_score[-1] - net_flow_score[0]),
        ]
        growth = pack_growth(deltas)
    else:
        growth = pack_growth([0, 0, 0, 0, 0])

    return {"snapshot": snapshot, "trend": trend, "risk": risk, "growth": growth}


def _networth_series_from_month_df(df: pd.DataFrame, months_out: int = 24) -> list[dict[str, float]]:
    base = float(pipeline_config.get("networth_baseline", 14500.0))

    inc = df["Income_Deposits"].to_numpy(dtype=float)

    outflows = np.zeros_like(inc)
    for k in UI_CATEGORIES_17:
        if k == "Income_Deposits":
            continue
        outflows += np.nan_to_num(df[k].to_numpy(dtype=float), nan=0.0)

    savings = np.nan_to_num(df["Savings_Investments"].to_numpy(dtype=float), nan=0.0)
    debt = np.nan_to_num(df["Debt_Payments"].to_numpy(dtype=float), nan=0.0)

    consumption = np.maximum(0.0, outflows - savings - debt)
    net_change = np.nan_to_num(inc, nan=0.0) - consumption

    series: list[dict[str, float]] = []
    net = base
    for i in range(len(df)):
        net += float(net_change[i])
        series.append({"month": i + 1, "networth": float(round(net, 2))})

    if len(series) == 0:
        series.append({"month": 1, "networth": float(round(base, 2))})

    last_change = float(net_change[-1]) if len(net_change) else 0.0
    while len(series) < months_out:
        m = len(series) + 1
        drift = 0.015 * last_change * (1.0 if (m % 6) else 0.5)
        net += last_change + drift
        series.append({"month": m, "networth": float(round(net, 2))})

    return series[:months_out]


# -----------------------------
# Cluster space: cached cloud and PCA params
# -----------------------------
def _pca_project(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        Xc = X - mean
        return Xc @ components.T
    Xc = X - mean.reshape(1, -1)
    return Xc @ components.T


def _cluster_space_from_cache(X_user_trans: np.ndarray, warnings: list[str]) -> ClusterSpace:
    if not isinstance(cluster_cache, dict):
        warnings.append("Cluster cache not loaded. Rebuild in Colab and copy into artifacts/.")
        return ClusterSpace(points=[], user_point=None)

    pts_raw = cluster_cache.get("points", None)
    pca_mean = cluster_cache.get("pca_mean", None)
    pca_components = cluster_cache.get("pca_components", None)

    cache_cols = cluster_cache.get("feature_cols", None)
    if isinstance(cache_cols, list) and cache_cols != feature_cols:
        warnings.append("Cluster cache feature_cols != current feature_cols. Rebuild cache to match.")

    if not isinstance(pts_raw, list) or pca_mean is None or pca_components is None:
        warnings.append("Cluster cache missing 'points' or 'pca_mean'/'pca_components'. Rebuild cache.")
        return ClusterSpace(points=[], user_point=None)

    try:
        mean = np.asarray(pca_mean, dtype=float).reshape(-1)
        comps = np.asarray(pca_components, dtype=float)
        if comps.shape[0] != 2:
            raise ValueError(f"pca_components first dim must be 2, got {comps.shape}")

        user_xy = _pca_project(X_user_trans.reshape(-1), mean, comps).reshape(2)
        user_point = ScatterPoint(x=float(user_xy[0]), y=float(user_xy[1]), k=-1, isUser=True)
    except Exception as e:
        warnings.append(f"Failed to project user point with cached PCA params: {type(e).__name__}: {e}")
        user_point = None

    points: list[ScatterPoint] = []
    max_pts = int(pipeline_config.get("cluster_space_samples", 1500))

    for p in pts_raw[:max_pts]:
        try:
            points.append(ScatterPoint(x=float(p["x"]), y=float(p["y"]), k=int(p["k"]), isUser=False))
        except Exception:
            continue

    if user_point is not None:
        points.append(user_point)

    return ClusterSpace(points=points, user_point=user_point)


# -----------------------------
# Conclusion
# -----------------------------
def _build_conclusion(
    df_months: pd.DataFrame,
    top: int,
    probs: np.ndarray,
    missing_fields: list[str],
) -> Conclusion:
    top_prob = float(np.max(probs)) if probs.size else 0.0

    last = df_months.iloc[-1] if len(df_months) else pd.Series(dtype=float)
    inc = float(last.get("Income_Deposits", np.nan))

    drivers: list[str] = []
    if np.isfinite(inc) and inc > 0:
        shares = []
        for k in UI_CATEGORIES_17:
            if k == "Income_Deposits":
                continue
            v = float(last.get(k, np.nan))
            if np.isfinite(v) and v > 0:
                shares.append((k, v / inc))
        shares.sort(key=lambda x: x[1], reverse=True)
        for k, sh in shares[:4]:
            drivers.append(f"{k} is {sh*100:.1f}% of income (latest month)")
    else:
        drivers.append("Income_Deposits is missing or ≤ 0 in the latest month; drivers are less reliable.")

    txt = (
        f"Model predicts cluster C{top + 1} with {top_prob*100:.1f}% confidence "
        f"based on your provided months. The prediction reflects the engineered features "
        f"used during training (shares/rates + aggregation across months)."
    )

    return Conclusion(text=txt, drivers=drivers, missing_fields=missing_fields)


# -----------------------------
# Routes
# -----------------------------
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
            "cluster_space_cache.joblib": CLUSTER_CACHE_JOBLIB.exists(),
            "cluster_space_cache.json": CLUSTER_CACHE_JSON.exists(),
            "cluster_space_cache.pkl": CLUSTER_CACHE_PKL.exists(),
        },
        "warnings": [w for w in [sampler_warn, cluster_cache_warn] if w],
        "pipeline_config": pipeline_config,
        "n_features": len(feature_cols),
    }


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    warnings: list[str] = []
    if sampler_warn:
        warnings.append(sampler_warn)
    if cluster_cache_warn:
        warnings.append(cluster_cache_warn)

    # 1) For charts and missingness warnings
    months_df, w1, missing_fields = _build_month_df(payload.months)
    warnings.extend(w1)

    # 2) Exact model feature vector
    X_raw_df = build_features_from_months(payload.months, spec)

    # 3) Apply inference transforms
    X_trans, w2 = _transform_features(X_raw_df)
    warnings.extend(w2)

    # 4) Predict
    probs = _predict_probs(X_trans)
    probs = np.array(probs, dtype=float).reshape(-1)

    s = float(np.sum(probs)) if probs.size else 0.0
    if probs.size and (not np.isfinite(s) or s <= 0):
        warnings.append("Model returned invalid probabilities; falling back to uniform distribution.")
        probs = np.ones(6, dtype=float) / 6.0
    elif probs.size and abs(s - 1.0) > 1e-6:
        probs = probs / s

    top = int(np.argmax(probs)) if probs.size else 0

    # 5) Charts
    radars = _radar_from_month_df(months_df)
    networth = _networth_series_from_month_df(months_df, months_out=int(pipeline_config.get("networth_months_out", 24)))

    # 6) Cluster space
    cluster_space = _cluster_space_from_cache(X_trans, warnings)

    # 7) Conclusion
    conclusion = _build_conclusion(months_df, top, probs, missing_fields)

    return PredictOut(
        top=top,
        probs=probs.tolist(),
        radars=radars,
        networth=[NetworthPoint(**p) for p in networth],
        cluster_space=cluster_space,
        conclusion=conclusion,
        warnings=warnings,
    )
