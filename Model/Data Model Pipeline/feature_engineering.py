from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import pandas as pd

# -------------------------
# Columns: 17 categories
# -------------------------
CATEGORIES_17 = [
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

INCOME_COL = "Income_Deposits"
OUTFLOW_COLS = [c for c in CATEGORIES_17 if c != INCOME_COL]


@dataclass(frozen=True)
class FeatureSpec:
    feature_cols: list[str]


def _to_float(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _safe_div(a: float, b: float) -> float:
    if b is None or b <= 0:
        return 0.0
    return a / b


def _mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else 0.0


def _median(arr: np.ndarray) -> float:
    return float(np.median(arr)) if arr.size else 0.0


def _std_pop(arr: np.ndarray) -> float:
    # population std (ddof=0)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=0))


def _share_series(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numer, dtype=float)
    for i in range(len(numer)):
        out[i] = _safe_div(float(numer[i]), float(denom[i]))
    return out


def _normalize_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    OPTIONAL:
    If the frontend accidentally sends older keys, map them into the new merged schema.
    This prevents all zeros features when keys mismatch.
    """
    # Transportation_Variable = Gas + PublicTransit (if not provided)
    if "Transportation_Variable" not in df.columns:
        gas = df["Transportation_Gas"] if "Transportation_Gas" in df.columns else 0.0
        pub = df["Transportation_PublicTransit"] if "Transportation_PublicTransit" in df.columns else 0.0
        df["Transportation_Variable"] = gas + pub

    # Insurance_All = Health + Auto (if not provided)
    if "Insurance_All" not in df.columns:
        ih = df["Insurance_Health"] if "Insurance_Health" in df.columns else 0.0
        ia = df["Insurance_Auto"] if "Insurance_Auto" in df.columns else 0.0
        df["Insurance_All"] = ih + ia

    # Healthcare_OOP from Medical_OutOfPocket (if not provided)
    if "Healthcare_OOP" not in df.columns and "Medical_OutOfPocket" in df.columns:
        df["Healthcare_OOP"] = df["Medical_OutOfPocket"]

    # Auto_Costs fallback: if Auto_Costs missing but Insurance_Auto exists
    if "Auto_Costs" not in df.columns and "Insurance_Auto" in df.columns:
        df["Auto_Costs"] = df["Insurance_Auto"]

    # Cash_ATM_MiscTransfers fallback if older name exists
    if "Cash_ATM_MiscTransfers" not in df.columns and "Cash_ATM_Misc" in df.columns:
        df["Cash_ATM_MiscTransfers"] = df["Cash_ATM_Misc"]

    return df


def build_features_from_months(months: list[dict], spec: FeatureSpec) -> pd.DataFrame:
    """
    Main logic:
      - monthly shares: category / Income_Deposits
      - NetCashflowRate = (Income - sum(outflows)) / Income
      - EssentialRate = (Housing + Utilities + Groceries) / Income
      - DiscretionaryRate = (Dining + Entertainment + Travel + Subscriptions) / Income
      - aggregate across months using mean/median/std (ddof=0), then select feature_cols
    """
    if not isinstance(months, list) or len(months) == 0:
        months = [{}]

    df = pd.DataFrame(months)
    df = _normalize_legacy_columns(df)

    # ensure all expected cols exist and numeric
    for k in CATEGORIES_17:
        if k not in df.columns:
            df[k] = 0.0
        df[k] = df[k].map(_to_float)

    income = df[INCOME_COL].to_numpy(dtype=float)

    # per-category arrays
    housing = df["Housing"].to_numpy(dtype=float)
    utilities = df["Utilities_Telecom"].to_numpy(dtype=float)
    groceries = df["Groceries_FoodAtHome"].to_numpy(dtype=float)
    dining = df["Dining_FoodAway"].to_numpy(dtype=float)
    trans_var = df["Transportation_Variable"].to_numpy(dtype=float)
    auto_costs = df["Auto_Costs"].to_numpy(dtype=float)
    healthcare_oop = df["Healthcare_OOP"].to_numpy(dtype=float)
    insurance_all = df["Insurance_All"].to_numpy(dtype=float)
    debt = df["Debt_Payments"].to_numpy(dtype=float)
    edu = df["Education_Childcare"].to_numpy(dtype=float)
    entertainment = df["Entertainment"].to_numpy(dtype=float)
    subs = df["Subscriptions_Memberships"].to_numpy(dtype=float)
    cash_atm_misc = df["Cash_ATM_MiscTransfers"].to_numpy(dtype=float)
    pets = df["Pets"].to_numpy(dtype=float)
    travel = df["Travel"].to_numpy(dtype=float)

    # shares
    housing_sh = _share_series(housing, income)
    utilities_sh = _share_series(utilities, income)
    groceries_sh = _share_series(groceries, income)
    dining_sh = _share_series(dining, income)
    trans_var_sh = _share_series(trans_var, income)
    auto_costs_sh = _share_series(auto_costs, income)
    healthcare_oop_sh = _share_series(healthcare_oop, income)
    insurance_all_sh = _share_series(insurance_all, income)
    debt_sh = _share_series(debt, income)
    edu_sh = _share_series(edu, income)
    entertainment_sh = _share_series(entertainment, income)
    subs_sh = _share_series(subs, income)
    cash_atm_misc_sh = _share_series(cash_atm_misc, income)
    pets_sh = _share_series(pets, income)
    travel_sh = _share_series(travel, income)

    # ---- rates ----
    essential_amt = housing + utilities + groceries
    discretionary_amt = dining + entertainment + travel + subs

    essential_rate = _share_series(essential_amt, income)
    debt_rate = debt_sh
    discretionary_rate = _share_series(discretionary_amt, income)

    # NetCashflowRate = (Income - sum(outflows)) / Income
    outflows_amt = np.zeros_like(income, dtype=float)
    for c in OUTFLOW_COLS:
        outflows_amt += df[c].to_numpy(dtype=float)

    net_cashflow_rate = np.zeros_like(income, dtype=float)
    for i in range(len(income)):
        inc = float(income[i])
        if inc <= 0:
            net_cashflow_rate[i] = 0.0
        else:
            net_cashflow_rate[i] = (inc - float(outflows_amt[i])) / inc

    # aggregate into exact expected feature vector
    feats: Dict[str, float] = {}

    def put_mean(prefix: str, arr: np.ndarray):
        feats[f"{prefix}__mean"] = _mean(arr)

    def put_median(prefix: str, arr: np.ndarray):
        feats[f"{prefix}__median"] = _median(arr)

    def put_std(prefix: str, arr: np.ndarray):
        feats[f"{prefix}__std"] = _std_pop(arr)

    # Shares (match feature_cols.json)
    put_std("Housing__share", housing_sh)
    put_std("Utilities_Telecom__share", utilities_sh)

    put_median("Groceries_FoodAtHome__share", groceries_sh)
    put_std("Groceries_FoodAtHome__share", groceries_sh)

    put_std("Dining_FoodAway__share", dining_sh)

    put_median("Transportation_Variable__share", trans_var_sh)
    put_std("Transportation_Variable__share", trans_var_sh)

    put_mean("Auto_Costs__share", auto_costs_sh)
    put_median("Auto_Costs__share", auto_costs_sh)
    put_std("Auto_Costs__share", auto_costs_sh)

    put_mean("Healthcare_OOP__share", healthcare_oop_sh)
    put_median("Healthcare_OOP__share", healthcare_oop_sh)
    put_std("Healthcare_OOP__share", healthcare_oop_sh)

    put_mean("Insurance_All__share", insurance_all_sh)
    put_std("Insurance_All__share", insurance_all_sh)

    put_mean("Debt_Payments__share", debt_sh)
    put_median("Debt_Payments__share", debt_sh)
    put_std("Debt_Payments__share", debt_sh)

    put_std("Education_Childcare__share", edu_sh)

    put_median("Entertainment__share", entertainment_sh)
    put_std("Entertainment__share", entertainment_sh)

    put_median("Subscriptions_Memberships__share", subs_sh)
    put_std("Subscriptions_Memberships__share", subs_sh)

    put_mean("Cash_ATM_MiscTransfers__share", cash_atm_misc_sh)
    put_median("Cash_ATM_MiscTransfers__share", cash_atm_misc_sh)
    put_std("Cash_ATM_MiscTransfers__share", cash_atm_misc_sh)

    put_mean("Pets__share", pets_sh)
    put_median("Pets__share", pets_sh)
    put_std("Pets__share", pets_sh)

    put_median("Travel__share", travel_sh)

    # Rates (match feature_cols.json)
    put_std("EssentialRate", essential_rate)
    put_mean("DebtRate", debt_rate)
    put_median("DebtRate", debt_rate)
    put_mean("DiscretionaryRate", discretionary_rate)
    put_std("DiscretionaryRate", discretionary_rate)
    put_mean("NetCashflowRate", net_cashflow_rate)
    put_std("NetCashflowRate", net_cashflow_rate)

    X = pd.DataFrame([feats])

    missing = [c for c in spec.feature_cols if c not in X.columns]
    if missing:
        raise ValueError(
            f"Feature engineering mismatch: {len(missing)} expected features are missing. "
            f"Example: {missing[:10]}"
        )

    return X[spec.feature_cols]
