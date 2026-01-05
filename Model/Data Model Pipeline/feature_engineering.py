from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CATEGORIES_17 = [
    "Income_Deposits",
    "Housing",
    "Utilities_Telecom",
    "Groceries_FoodAtHome",
    "Dining_FoodAway",
    "Transportation_Gas",
    "Transportation_PublicTransit",
    "Insurance_Health",
    "Insurance_Auto",
    "Medical_OutOfPocket",
    "Debt_Payments",
    "Savings_Investments",
    "Education_Childcare",
    "Entertainment",
    "Subscriptions_Memberships",
    "Pets",
    "Travel",
]


@dataclass(frozen=True)
class FeatureSpec:
    feature_cols: list[str]


def _to_float(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    try:
        s = str(x).strip()
        if s == "":
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def _safe_div(a: float, b: float) -> float:
    if b is None or b <= 0:
        return 0.0
    return a / b


def _std_pop(arr: np.ndarray) -> float:
    # population std (ddof=0) so it stays stable even for small n
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=0))


def _mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def _median(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))


def _share_series(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numer, dtype=float)
    for i in range(len(numer)):
        out[i] = _safe_div(float(numer[i]), float(denom[i]))
    return out


def build_features_from_months(months: list[dict], spec: FeatureSpec) -> pd.DataFrame:
    """
    Converts UI monthly inputs into the exact engineered feature vector expected by feature_cols.json.

    IMPORTANT:
    - Uses shares relative to Income_Deposits (per-month).
    - Aggregates over months with mean / median / std as requested by feature_cols.
    - Some features (Cash_ATM_MiscTransfers) are not present in UI -> set to 0 share.
    """

    if not isinstance(months, list) or len(months) == 0:
        months = [{}]

    # Build numeric dataframe with all UI categories present
    df = pd.DataFrame(months)
    for k in CATEGORIES_17:
        if k not in df.columns:
            df[k] = 0.0
        df[k] = df[k].map(_to_float)

    income = df["Income_Deposits"].to_numpy(dtype=float)

    # ----------------------------------------
    # Group definitions to match feature names
    # ----------------------------------------
    housing = df["Housing"].to_numpy(dtype=float)
    utilities = df["Utilities_Telecom"].to_numpy(dtype=float)
    groceries = df["Groceries_FoodAtHome"].to_numpy(dtype=float)
    dining = df["Dining_FoodAway"].to_numpy(dtype=float)

    # Transportation_Variable: gas + public transit (variable transport)
    trans_var = (df["Transportation_Gas"] + df["Transportation_PublicTransit"]).to_numpy(dtype=float)

    # Auto_Costs: use auto insurance (fixed-ish auto cost)
    # NOTE: If training defined Auto_Costs differently, update this mapping.
    auto_costs = df["Insurance_Auto"].to_numpy(dtype=float)

    healthcare_oop = df["Medical_OutOfPocket"].to_numpy(dtype=float)

    # Insurance_All: health and auto
    insurance_all = (df["Insurance_Health"] + df["Insurance_Auto"]).to_numpy(dtype=float)

    debt = df["Debt_Payments"].to_numpy(dtype=float)
    edu = df["Education_Childcare"].to_numpy(dtype=float)
    entertainment = df["Entertainment"].to_numpy(dtype=float)
    subs = df["Subscriptions_Memberships"].to_numpy(dtype=float)
    pets = df["Pets"].to_numpy(dtype=float)
    travel = df["Travel"].to_numpy(dtype=float)

    # UI doesn't have this category -> keep zeros but still produce expected feature columns
    cash_atm_misc = np.zeros_like(income, dtype=float)

    # ----------------------------------------
    # Shares relative to income
    # ----------------------------------------
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

    # ----------------------------------------
    # Rate features (relative to income)
    # ----------------------------------------
    essentials_amt = housing + utilities + groceries + trans_var + insurance_all + healthcare_oop + edu + pets
    discretionary_amt = dining + entertainment + subs + travel

    essential_rate = _share_series(essentials_amt, income)
    debt_rate = debt_sh
    discretionary_rate = _share_series(discretionary_amt, income)

    # NetCashflowRate = (Income - Outflows) / Income
    outflows_amt = np.zeros_like(income, dtype=float)
    for k in CATEGORIES_17:
        if k == "Income_Deposits":
            continue
        outflows_amt += df[k].to_numpy(dtype=float)
    net_cashflow_rate = np.zeros_like(income, dtype=float)
    for i in range(len(income)):
        inc = float(income[i])
        if inc <= 0:
            net_cashflow_rate[i] = 0.0
        else:
            net_cashflow_rate[i] = (inc - float(outflows_amt[i])) / inc

    # ----------------------------------------
    # Aggregate into the exact feature vector
    # ----------------------------------------
    feats: dict[str, float] = {}

    # Helpers for writing features
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

    # Rates
    put_std("EssentialRate", essential_rate)

    put_mean("DebtRate", debt_rate)
    put_median("DebtRate", debt_rate)

    put_mean("DiscretionaryRate", discretionary_rate)
    put_std("DiscretionaryRate", discretionary_rate)

    put_mean("NetCashflowRate", net_cashflow_rate)
    put_std("NetCashflowRate", net_cashflow_rate)

    # Build df in exact column order expected by the model
    X = pd.DataFrame([feats])

    # Ensure all expected columns exist (fill missing with 0)
    for col in spec.feature_cols:
        if col not in X.columns:
            X[col] = 0.0

    return X[spec.feature_cols]
