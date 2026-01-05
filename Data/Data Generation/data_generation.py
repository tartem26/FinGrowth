"""
Synthetic personal-finance dataset generator (1-person household assumption).

Goal
- Generate 6 CSV files (one per income cluster).
- Each file contains N_people * 48 rows (1 row = 1 month).
- Each row has 17 columns: Income + 16 outflow categories.
- Values are in USD rounded to cents.
- Generation is constraint-aware:
  - hard floors/caps per category (baseline)
  - soft % of income ranges per cluster (enforced as dynamic clamps)
  - budget closure via rebalancing toward a target outflow ratio
- Person-level traits are sampled first (Latin Hypercube) and then 48 months are simulated.

Notes
- This generates synthetic data; no real customer data is used.
- For reproducibility, provide a --seed; otherwise a cryptographic seed is created.

Usage (full test for a target scale with metadata)
  python ./data_generation.py --outdir ../DataSets --people-per-cluster 20000 --months 48 --include-metadata

Usage (full test for a target scale without metadata)
  python ./data_generation.py --outdir ../DataSets --people-per-cluster 20000 --months 48

Usage (small test)
  python ./data_generation.py --outdir ../DataSets --people-per-cluster 200 --months 48 --seed 123

Usage (person per month identifiers in the CSV)
  python ./data_generation.py --outdir ../DataSets --people-per-cluster 200 --months 48 --seed 123 --include-metadata

Usage (change the top-5% max income bound)
  python ./data_generation.py --outdir ../DataSets --people-per-cluster 200 --months 48 --seed 123 --top5-max 2000000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import secrets
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# -------------------------
# Configuration: clusters
# -------------------------
# Treat these as individual annual income cutoffs for a 1-person household.
# Cluster 6 is top 5%; it needs a practical max to keep generation bounded.
CLUSTERS = [
    ("C1_low",        10_000,   34_510),
    ("C2_lower_mid",  34_510,   65_100),
    ("C3_mid",        65_100,  105_500),
    ("C4_upper_mid", 105_500,  175_700),
    ("C5_high",      175_700,  335_700),
    ("C6_top5",      335_700, 1_500_000),  # adjustable via --top5-max
]

MONTHS_PER_YEAR = 12


# -------------------------
# Columns: 17 categories
# -------------------------
# 1 income + 16 outflows = 17 total.
INCOME_COL = "Income_Deposits"

OUTFLOW_COLS = [
    "Housing",                  # rent/mortgage/HOA/property tax/parking/maintenance (basic)
    "Utilities_Telecom",        # electric/gas/water/trash/internet/phone
    "Groceries_FoodAtHome",     # groceries
    "Dining_FoodAway",          # restaurants/takeout/delivery
    "Transportation_Variable",  # gas/transit/rideshare/parking/tolls
    "Auto_Costs",               # car payment/lease + maintenance/repairs/registration
    "Healthcare_OOP",           # out-of-pocket healthcare
    "Insurance_All",            # health premiums + other insurance (auto/renters/life/etc.)
    "Debt_Payments",            # CC/student/personal/BNPL/etc.
    "Savings_Investments",      # transfers to savings/brokerage/retirement
    "Education_Childcare",      # tuition/courses/childcare (even for 1-person, keep range)
    "Entertainment",            # events/hobbies/gaming/etc.
    "Subscriptions_Memberships",# streaming/apps/gym/etc.
    "Cash_ATM_MiscTransfers",   # cash withdrawals/P2P/bank fees/uncategorized transfers
    "Pets",                     # pet expenses
    "Travel",                   # travel expenses
]

ALL_COLS = [INCOME_COL] + OUTFLOW_COLS


# -------------------------
# Boundaries (baseline hard floors/caps + soft % ranges by cluster)
# -------------------------
# The % ranges are applied as dynamic clamps relative to monthly income.
# Also keep a baseline absolute floor/cap, and use:
#   floor = max(abs_floor, pct_lo * income_monthly)
#   cap   = max(abs_cap,   pct_hi * income_monthly)
#
# That means higher incomes naturally allow higher category caps, while still guided by pct limits.

# Category baselines: abs_floor, abs_cap (monthly, USD)
ABS_BOUNDS = {
    "Housing":                 (400.0, 25_000.0),
    "Utilities_Telecom":       ( 80.0,  1_500.0),
    "Groceries_FoodAtHome":    (250.0,  2_500.0),
    "Dining_FoodAway":         (  0.0,  3_000.0),
    "Transportation_Variable": ( 50.0,  4_000.0),
    "Auto_Costs":              (  0.0,  3_000.0),
    "Healthcare_OOP":          (  0.0,  2_500.0),
    "Insurance_All":           ( 50.0,  3_000.0),
    "Debt_Payments":           (  0.0, 15_000.0),
    "Savings_Investments":     (  0.0, 25_000.0),
    "Education_Childcare":     (  0.0,  4_000.0),
    "Entertainment":           (  0.0,  3_000.0),
    "Subscriptions_Memberships":(0.0,    700.0),
    "Cash_ATM_MiscTransfers":  (  0.0,  5_000.0),
    "Pets":                    (  0.0,  1_200.0),
    "Travel":                  (  0.0, 10_000.0),
}

# Soft % ranges by cluster: per outflow category => (pct_lo, pct_hi)
# Intentionally wide for broad coverage; realistic shape comes from Dirichlet + traits + constraints.
PCT_RANGES = {
    "C1_low": {
        "Housing": (0.30, 0.60),
        "Utilities_Telecom": (0.06, 0.14),
        "Groceries_FoodAtHome": (0.08, 0.18),
        "Dining_FoodAway": (0.00, 0.08),
        "Transportation_Variable": (0.06, 0.20),
        "Auto_Costs": (0.00, 0.12),
        "Healthcare_OOP": (0.00, 0.10),
        "Insurance_All": (0.02, 0.12),
        "Debt_Payments": (0.00, 0.36),
        "Savings_Investments": (0.00, 0.08),
        "Education_Childcare": (0.00, 0.05),
        "Entertainment": (0.00, 0.08),
        "Subscriptions_Memberships": (0.00, 0.05),
        "Cash_ATM_MiscTransfers": (0.00, 0.10),
        "Pets": (0.00, 0.05),
        "Travel": (0.00, 0.05),
    },
    "C2_lower_mid": {
        "Housing": (0.25, 0.50),
        "Utilities_Telecom": (0.05, 0.12),
        "Groceries_FoodAtHome": (0.07, 0.15),
        "Dining_FoodAway": (0.00, 0.09),
        "Transportation_Variable": (0.07, 0.20),
        "Auto_Costs": (0.00, 0.12),
        "Healthcare_OOP": (0.00, 0.10),
        "Insurance_All": (0.02, 0.12),
        "Debt_Payments": (0.00, 0.36),
        "Savings_Investments": (0.00, 0.10),
        "Education_Childcare": (0.00, 0.06),
        "Entertainment": (0.00, 0.09),
        "Subscriptions_Memberships": (0.00, 0.05),
        "Cash_ATM_MiscTransfers": (0.00, 0.10),
        "Pets": (0.00, 0.05),
        "Travel": (0.00, 0.06),
    },
    "C3_mid": {
        "Housing": (0.20, 0.40),
        "Utilities_Telecom": (0.04, 0.10),
        "Groceries_FoodAtHome": (0.06, 0.12),
        "Dining_FoodAway": (0.01, 0.10),
        "Transportation_Variable": (0.06, 0.18),
        "Auto_Costs": (0.00, 0.10),
        "Healthcare_OOP": (0.00, 0.09),
        "Insurance_All": (0.02, 0.10),
        "Debt_Payments": (0.00, 0.30),
        "Savings_Investments": (0.02, 0.15),
        "Education_Childcare": (0.00, 0.07),
        "Entertainment": (0.00, 0.10),
        "Subscriptions_Memberships": (0.00, 0.04),
        "Cash_ATM_MiscTransfers": (0.00, 0.09),
        "Pets": (0.00, 0.04),
        "Travel": (0.00, 0.08),
    },
    "C4_upper_mid": {
        "Housing": (0.18, 0.35),
        "Utilities_Telecom": (0.03, 0.08),
        "Groceries_FoodAtHome": (0.05, 0.10),
        "Dining_FoodAway": (0.01, 0.12),
        "Transportation_Variable": (0.05, 0.16),
        "Auto_Costs": (0.00, 0.10),
        "Healthcare_OOP": (0.00, 0.08),
        "Insurance_All": (0.02, 0.10),
        "Debt_Payments": (0.00, 0.28),
        "Savings_Investments": (0.05, 0.20),
        "Education_Childcare": (0.00, 0.08),
        "Entertainment": (0.00, 0.12),
        "Subscriptions_Memberships": (0.00, 0.04),
        "Cash_ATM_MiscTransfers": (0.00, 0.08),
        "Pets": (0.00, 0.04),
        "Travel": (0.00, 0.10),
    },
    "C5_high": {
        "Housing": (0.15, 0.30),
        "Utilities_Telecom": (0.02, 0.06),
        "Groceries_FoodAtHome": (0.04, 0.08),
        "Dining_FoodAway": (0.01, 0.14),
        "Transportation_Variable": (0.04, 0.14),
        "Auto_Costs": (0.00, 0.08),
        "Healthcare_OOP": (0.00, 0.07),
        "Insurance_All": (0.02, 0.08),
        "Debt_Payments": (0.00, 0.25),
        "Savings_Investments": (0.08, 0.30),
        "Education_Childcare": (0.00, 0.08),
        "Entertainment": (0.00, 0.14),
        "Subscriptions_Memberships": (0.00, 0.03),
        "Cash_ATM_MiscTransfers": (0.00, 0.07),
        "Pets": (0.00, 0.03),
        "Travel": (0.00, 0.12),
    },
    "C6_top5": {
        "Housing": (0.10, 0.25),
        "Utilities_Telecom": (0.01, 0.05),
        "Groceries_FoodAtHome": (0.03, 0.07),
        "Dining_FoodAway": (0.01, 0.15),
        "Transportation_Variable": (0.03, 0.12),
        "Auto_Costs": (0.00, 0.06),
        "Healthcare_OOP": (0.00, 0.06),
        "Insurance_All": (0.01, 0.07),
        "Debt_Payments": (0.00, 0.20),
        "Savings_Investments": (0.10, 0.40),
        "Education_Childcare": (0.00, 0.08),
        "Entertainment": (0.00, 0.15),
        "Subscriptions_Memberships": (0.00, 0.03),
        "Cash_ATM_MiscTransfers": (0.00, 0.06),
        "Pets": (0.00, 0.03),
        "Travel": (0.00, 0.15),
    },
}

ESSENTIALS = {"Housing", "Utilities_Telecom", "Groceries_FoodAtHome"}


# -------------------------
# Helper dataclasses
# -------------------------
@dataclass
class ClusterSpec:
    name: str
    annual_low: float
    annual_high: float


@dataclass
class PersonSpec:
    cluster: str
    person_id: int

    annual_income: float
    base_monthly_income: float

    mode: str  # "balanced" | "debt_stress" | "aggressive_saver"

    # Traits (0..1 or booleans) that shift spending composition
    car_owner: bool
    pet_owner: bool
    travel_prop: float
    dining_prop: float
    saver_prop: float
    debt_prop: float
    cash_prop: float

    # Dirichlet alpha for 16 outflow categories
    alpha: np.ndarray

    # Month indices where spikes occur
    travel_months: set
    health_spike_months: set
    auto_repair_months: set


# -------------------------
# Randomness
# -------------------------
def choose_seed(user_seed: int | None) -> int:
    """If user_seed is None, create a cryptographic seed via OS entropy."""
    if user_seed is not None:
        return int(user_seed)
    return secrets.randbits(128)


def make_rng(seed: int) -> np.random.Generator:
    """Fast PRNG for bulk simulation."""
    return np.random.default_rng(seed)


# -------------------------
# Latin Hypercube Sampling (for person-level traits)
# -------------------------
def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simple LHS in [0,1): each dimension is stratified into n bins.
    Good for evenly covering parameter space without collapsing to the mean.
    """
    H = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        H[:, j] = (perm + rng.random(n)) / n
    return H


# -------------------------
# Income sampling inside cluster
# -------------------------
def sample_annual_income(cluster: ClusterSpec, u: float, top5_log_uniform: bool = True) -> float:
    """
    Sample annual income within cluster range.
    - For clusters 1-5: uniform.
    - For top 5: optionally log-uniform to cover very wide range more evenly.
    """
    lo, hi = cluster.annual_low, cluster.annual_high
    if cluster.name == "C6_top5" and top5_log_uniform:
        # log-uniform across [lo, hi]
        return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))
    return float(lo + u * (hi - lo))


# -------------------------
# Category bounds given income + cluster
# -------------------------
def category_bounds(cluster_name: str, cat: str, income_monthly: float) -> Tuple[float, float]:
    abs_floor, abs_cap = ABS_BOUNDS[cat]
    pct_lo, pct_hi = PCT_RANGES[cluster_name][cat]

    # Dynamic clamps (scale with income). Keep abs bounds as baseline.
    floor = max(abs_floor, pct_lo * income_monthly)
    cap = max(abs_cap, pct_hi * income_monthly)

    # Ensure cap ≥ floor
    cap = max(cap, floor)
    return float(floor), float(cap)


# -------------------------
# Rebalancing utilities
# -------------------------
def rebalance_to_target(
    amounts: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    target_total: float,
    max_iters: int = 30,
) -> np.ndarray:
    """
    Adjust amounts so sum(amounts) is as close as possible to target_total,
    respecting per-category floors/caps.

    Strategy:
    - If sum too low: add to categories with slack to cap, proportional to slack.
    - If sum too high: remove from categories above floor, proportional to excess.
    """
    x = amounts.copy()
    for _ in range(max_iters):
        total = float(x.sum())
        delta = target_total - total

        if abs(delta) < 0.01:  # within 1 cent
            break

        if delta > 0:
            slack = caps - x
            slack[slack < 0] = 0
            s = float(slack.sum())
            if s <= 1e-9:
                break
            add = slack * (delta / s)
            x = np.minimum(caps, x + add)
        else:
            excess = x - floors
            excess[excess < 0] = 0
            s = float(excess.sum())
            if s <= 1e-9:
                break
            remove = excess * ((-delta) / s)
            x = np.maximum(floors, x - remove)

    return x


def clamp_amounts(amounts: np.ndarray, floors: np.ndarray, caps: np.ndarray) -> np.ndarray:
    return np.minimum(caps, np.maximum(floors, amounts))


# -------------------------
# Person-level templates (Dirichlet concentration + mode probabilities)
# -------------------------
def cluster_mode_probs(cluster_name: str) -> Dict[str, float]:
    # More debt-stress at low incomes, more aggressive saving at high incomes.
    if cluster_name == "C1_low":
        return {"balanced": 0.65, "debt_stress": 0.30, "aggressive_saver": 0.05}
    if cluster_name == "C2_lower_mid":
        return {"balanced": 0.70, "debt_stress": 0.20, "aggressive_saver": 0.10}
    if cluster_name == "C3_mid":
        return {"balanced": 0.72, "debt_stress": 0.13, "aggressive_saver": 0.15}
    if cluster_name == "C4_upper_mid":
        return {"balanced": 0.70, "debt_stress": 0.10, "aggressive_saver": 0.20}
    if cluster_name == "C5_high":
        return {"balanced": 0.65, "debt_stress": 0.07, "aggressive_saver": 0.28}
    return {"balanced": 0.60, "debt_stress": 0.05, "aggressive_saver": 0.35}


def sample_mode(rng: np.random.Generator, probs: Dict[str, float]) -> str:
    modes = list(probs.keys())
    p = np.array([probs[m] for m in modes], dtype=np.float64)
    p = p / p.sum()
    return str(rng.choice(modes, p=p))


def outflow_ratio_distribution(cluster_name: str, mode: str, rng: np.random.Generator) -> float:
    """
    Total outflow ratio relative to income for the month.
    - balanced: around 0.70..1.05
    - debt_stress: around 1.05..1.25
    - aggressive_saver: around 0.75..1.00 but with high savings/investing share
    """
    if mode == "debt_stress":
        return float(rng.uniform(1.05, 1.25))

    if mode == "aggressive_saver":
        # High-income people often still don't spend all income; transfers to investments dominate.
        # Keep total outflow moderate; allocation does the heavy lifting.
        lo, hi = (0.75, 1.00) if cluster_name != "C6_top5" else (0.65, 0.95)
        return float(rng.uniform(lo, hi))

    # balanced
    lo, hi = (0.70, 1.05) if cluster_name != "C6_top5" else (0.60, 1.00)
    return float(rng.uniform(lo, hi))


def base_alpha_from_pct_midpoints(cluster_name: str, concentration: float) -> np.ndarray:
    """
    Convert pct-range midpoints into a Dirichlet alpha vector (length=16 outflows).
    This sets expected shares; concentration controls variability.
    """
    mids = []
    for cat in OUTFLOW_COLS:
        lo, hi = PCT_RANGES[cluster_name][cat]
        mids.append(max((lo + hi) / 2.0, 1e-6))
    mids = np.array(mids, dtype=np.float64)
    mids = mids / mids.sum()
    alpha = mids * concentration
    alpha[alpha < 1e-3] = 1e-3
    return alpha


def trait_shift_alpha(alpha: np.ndarray, idx: Dict[str, int], **mults: float) -> np.ndarray:
    """
    Multiply selected category alphas to encode person traits.
    Example: travel-heavy person => higher alpha for Travel.
    """
    a = alpha.copy()
    for cat, m in mults.items():
        if cat in idx:
            a[idx[cat]] *= float(m)
    a[a < 1e-3] = 1e-3
    return a


# -------------------------
# Person generation
# -------------------------
def build_person_spec(
    cluster: ClusterSpec,
    cluster_name: str,
    person_id: int,
    lhs_row: np.ndarray,
    rng: np.random.Generator,
    top5_log_uniform: bool,
) -> PersonSpec:
    """
    Use LHS to sample a stable set of traits for a person.
    Traits then control spending patterns across 48 months.
    """
    # LHS dims interpretation (0..1):
    # 0 income position in cluster
    # 1 saver propensity
    # 2 debt propensity
    # 3 travel propensity
    # 4 dining propensity
    # 5 cash propensity
    # 6 car owner likelihood
    # 7 pet owner likelihood
    u_income, u_saver, u_debt, u_travel, u_dining, u_cash, u_car, u_pet = lhs_row

    annual_income = sample_annual_income(cluster, u_income, top5_log_uniform=top5_log_uniform)
    base_monthly_income = annual_income / MONTHS_PER_YEAR

    saver_prop = float(u_saver)
    debt_prop = float(u_debt)
    travel_prop = float(u_travel)
    dining_prop = float(u_dining)
    cash_prop = float(u_cash)

    # Ownership as thresholded probabilities (cluster-conditioned)
    car_owner = bool(u_car < (0.35 if cluster_name in ("C1_low", "C2_lower_mid") else 0.65))
    pet_owner = bool(u_pet < (0.25 if cluster_name == "C1_low" else 0.45))

    # Choose mode
    mode = sample_mode(rng, cluster_mode_probs(cluster_name))

    # Base Dirichlet alpha (cluster template)
    # More concentration => less month-to-month variation in category shares.
    concentration = 60.0 if cluster_name in ("C4_upper_mid", "C5_high", "C6_top5") else 45.0
    alpha = base_alpha_from_pct_midpoints(cluster_name, concentration)

    idx = {c: i for i, c in enumerate(OUTFLOW_COLS)}

    # Trait shifts (multipliers are mild; variability comes from Dirichlet)
    # Increase/decrease based on propensities.
    travel_mult = 0.7 + 1.8 * travel_prop
    dining_mult = 0.7 + 1.6 * dining_prop
    saver_mult = 0.6 + 2.2 * saver_prop
    debt_mult = 0.7 + 2.0 * debt_prop
    cash_mult = 0.7 + 1.4 * cash_prop

    alpha = trait_shift_alpha(
        alpha,
        idx,
        Travel=travel_mult,
        Dining_FoodAway=dining_mult,
        Savings_Investments=saver_mult,
        Debt_Payments=debt_mult,
        Cash_ATM_MiscTransfers=cash_mult,
        Auto_Costs=(1.3 if car_owner else 0.6),
        Transportation_Variable=(1.2 if not car_owner else 1.0),
        Pets=(1.4 if pet_owner else 0.5),
    )

    # If mode is aggressive_saver, boost savings further; if debt_stress, boost debt & dining variance.
    if mode == "aggressive_saver":
        alpha = trait_shift_alpha(alpha, idx, Savings_Investments=1.8, Travel=1.1, Dining_FoodAway=0.9)
    elif mode == "debt_stress":
        alpha = trait_shift_alpha(alpha, idx, Debt_Payments=1.6, Savings_Investments=0.6)

    # Pre-schedule spikes across 48 months
    months = 48
    travel_months = set()
    health_spike_months = set()
    auto_repair_months = set()

    # Travel spikes: more likely for higher clusters and high travel_prop
    base_trips_per_year = (0.5 + 3.0 * travel_prop)
    if cluster_name in ("C5_high", "C6_top5"):
        base_trips_per_year *= 1.4
    if cluster_name in ("C1_low",):
        base_trips_per_year *= 0.6

    expected_trips_4y = base_trips_per_year * 4.0
    n_trip_months = int(rng.poisson(expected_trips_4y))
    if n_trip_months > 0:
        chosen = rng.choice(np.arange(months), size=min(n_trip_months, months), replace=False)
        travel_months = set(int(x) for x in chosen)

    # Healthcare spikes: mostly rare
    n_health = int(rng.poisson(2.0 + 2.0 * debt_prop))  # slightly correlated with stress
    if n_health > 0:
        chosen = rng.choice(np.arange(months), size=min(n_health, months), replace=False)
        health_spike_months = set(int(x) for x in chosen)

    # Auto repairs spikes if car owner
    if car_owner:
        n_auto = int(rng.poisson(1.5))
        if n_auto > 0:
            chosen = rng.choice(np.arange(months), size=min(n_auto, months), replace=False)
            auto_repair_months = set(int(x) for x in chosen)

    return PersonSpec(
        cluster=cluster_name,
        person_id=person_id,
        annual_income=annual_income,
        base_monthly_income=base_monthly_income,
        mode=mode,
        car_owner=car_owner,
        pet_owner=pet_owner,
        travel_prop=travel_prop,
        dining_prop=dining_prop,
        saver_prop=saver_prop,
        debt_prop=debt_prop,
        cash_prop=cash_prop,
        alpha=alpha,
        travel_months=travel_months,
        health_spike_months=health_spike_months,
        auto_repair_months=auto_repair_months,
    )


# -------------------------
# Month simulation
# -------------------------
def seasonal_utility_multiplier(month_index: int) -> float:
    # Simple seasonality: winter/summer higher
    # month_index 0..47; use month of year for season
    m = month_index % 12
    # peaks around Jan (0) and Jul (6)
    return float(0.92 + 0.16 * (math.cos((m - 0) * 2 * math.pi / 12) ** 2))


def simulate_person_months(
    ps: PersonSpec,
    months: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns an array shape (months, 17) => [income, 16 outflows...]
    """
    out = np.zeros((months, 1 + len(OUTFLOW_COLS)), dtype=np.float64)
    idx = {c: i for i, c in enumerate(OUTFLOW_COLS)}

    # Stable monthly income with mild noise; occasional bonus for higher clusters
    income_sigma = 0.02 if ps.cluster in ("C1_low", "C2_lower_mid") else 0.03
    bonus_prob = 0.04 if ps.cluster in ("C5_high", "C6_top5") else 0.02
    bonus_months = set()
    if rng.random() < bonus_prob:
        bonus_months.add(int(rng.integers(0, months)))

    for t in range(months):
        # Monthly income noise
        income = ps.base_monthly_income * (1.0 + float(rng.normal(0.0, income_sigma)))
        income = max(income, ps.base_monthly_income * 0.6)

        # Bonus (rare): add 10%..60% of monthly income (bigger for top clusters)
        if t in bonus_months:
            bonus_scale = float(rng.uniform(0.10, 0.60 if ps.cluster == "C6_top5" else 0.35))
            income *= (1.0 + bonus_scale)

        # Total outflow ratio
        r = outflow_ratio_distribution(ps.cluster, ps.mode, rng)
        # small month-to-month noise on outflow ratio
        r *= float(1.0 + rng.normal(0.0, 0.03))
        r = max(0.45, min(r, 1.35))

        target_outflow = r * income

        # Draw category shares
        shares = rng.dirichlet(ps.alpha)

        # Convert to amounts
        amounts = shares * target_outflow  # length=16

        # Apply seasonality/spikes BEFORE clamping
        # Utilities seasonal
        util_i = idx["Utilities_Telecom"]
        amounts[util_i] *= seasonal_utility_multiplier(t)

        # Travel spikes: most months 0-ish, some months high
        travel_i = idx["Travel"]
        if t in ps.travel_months:
            # spike multiplier (lognormal-ish)
            amounts[travel_i] *= float(np.exp(rng.normal(1.0, 0.6)))
        else:
            # suppress travel most months
            amounts[travel_i] *= float(rng.uniform(0.0, 0.25))

        # Healthcare spikes
        health_i = idx["Healthcare_OOP"]
        if t in ps.health_spike_months:
            amounts[health_i] *= float(np.exp(rng.normal(0.8, 0.7)))
        else:
            amounts[health_i] *= float(rng.uniform(0.5, 1.2))

        # Auto repairs spikes (if car owner)
        auto_i = idx["Auto_Costs"]
        if t in ps.auto_repair_months:
            amounts[auto_i] *= float(np.exp(rng.normal(0.7, 0.6)))
        else:
            amounts[auto_i] *= float(rng.uniform(0.7, 1.15))

        # Housing stability: small variation
        house_i = idx["Housing"]
        amounts[house_i] *= float(rng.uniform(0.97, 1.03))

        # Groceries stability: small variation
        groc_i = idx["Groceries_FoodAtHome"]
        amounts[groc_i] *= float(rng.uniform(0.95, 1.08))

        # Dining varies with dining_prop
        dine_i = idx["Dining_FoodAway"]
        amounts[dine_i] *= float(0.85 + 0.8 * ps.dining_prop + rng.uniform(-0.10, 0.15))

        # Build floors/caps arrays for this month (dynamic by income)
        floors = np.zeros(len(OUTFLOW_COLS), dtype=np.float64)
        caps = np.zeros(len(OUTFLOW_COLS), dtype=np.float64)
        for i, cat in enumerate(OUTFLOW_COLS):
            f, c = category_bounds(ps.cluster, cat, income)
            floors[i] = f
            caps[i] = c

        # Clamp first
        amounts = clamp_amounts(amounts, floors, caps)

        # Enforce "average case survival": essentials must meet their floors (already by floors)
        # Rebalance to hit target_outflow as closely as possible under constraints
        amounts = rebalance_to_target(amounts, floors, caps, target_outflow)

        # Round later when writing; keep float here
        out[t, 0] = income
        out[t, 1:] = amounts

    return out


# -------------------------
# Output
# -------------------------
def format_money(x: float) -> str:
    # Round to cents; format as fixed 2 decimals.
    return f"{float(round(x + 1e-9, 2)):.2f}"


def write_cluster_csv(
    outdir: str,
    cluster: ClusterSpec,
    people_per_cluster: int,
    months: int,
    seed: int,
    include_metadata: bool,
    top5_log_uniform: bool,
    chunk_people: int,
) -> Dict[str, object]:
    """
    Generate one CSV for one cluster, streaming writes.
    Returns metadata (seed, settings, etc).
    """
    os.makedirs(outdir, exist_ok=True)
    fname = f"{cluster.name}.csv"
    path = os.path.join(outdir, fname)

    # Derive a stable sub-seed per cluster so reruns are reproducible and independent.
    cluster_seed = (seed ^ (hash(cluster.name) & 0xFFFFFFFF)) & ((1 << 128) - 1)
    rng = make_rng(cluster_seed)

    # Person-level LHS trait matrix: 8 dims
    # (Small memory footprint: 20k x 8 is fine)
    H = latin_hypercube(people_per_cluster, 8, rng)

    # Header
    header = []
    if include_metadata:
        header += ["cluster", "person_id", "month_index"]
    header += ALL_COLS

    rows_written = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        # Generate in person chunks to keep peak memory low
        for base in range(0, people_per_cluster, chunk_people):
            end = min(base + chunk_people, people_per_cluster)
            for p in range(base, end):
                person_id = p  # 0..people_per_cluster-1 per cluster file

                ps = build_person_spec(
                    cluster=cluster,
                    cluster_name=cluster.name,
                    person_id=person_id,
                    lhs_row=H[p],
                    rng=rng,
                    top5_log_uniform=top5_log_uniform,
                )

                series = simulate_person_months(ps, months, rng)

                # Write 48 rows
                for t in range(months):
                    row = []
                    if include_metadata:
                        row += [cluster.name, str(person_id), str(t)]

                    # income + 16 outflows
                    row.append(format_money(series[t, 0]))
                    for j in range(1, series.shape[1]):
                        row.append(format_money(series[t, j]))

                    w.writerow(row)
                    rows_written += 1

            # Lightweight progress log (doesn't slow writing too much)
            print(f"[{cluster.name}] people {base}..{end-1} done | rows_written={rows_written}")

    meta = {
        "cluster": cluster.name,
        "file": fname,
        "rows": rows_written,
        "people": people_per_cluster,
        "months": months,
        "cluster_seed": int(cluster_seed),
        "top5_log_uniform": top5_log_uniform,
        "include_metadata": include_metadata,
        "columns": header,
    }
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for CSV files.")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility.")
    ap.add_argument("--people-per-cluster", type=int, default=20_000, help="People per cluster (default: 20000).")
    ap.add_argument("--months", type=int, default=48, help="Months per person (default: 48).")
    ap.add_argument("--top5-max", type=float, default=1_500_000, help="Max annual income for top 5% cluster.")
    ap.add_argument("--no-top5-log-uniform", action="store_true", help="Use uniform instead of log-uniform for top 5%.")
    ap.add_argument("--include-metadata", action="store_true", help="Include cluster/person_id/month_index columns.")
    ap.add_argument("--chunk-people", type=int, default=500, help="People per chunk (default: 500).")
    args = ap.parse_args()

    # Update cluster 6 max if requested
    clusters = []
    for name, lo, hi in CLUSTERS:
        if name == "C6_top5":
            hi = float(args.top5_max)
        clusters.append(ClusterSpec(name=name, annual_low=float(lo), annual_high=float(hi)))

    seed = choose_seed(args.seed)
    include_metadata = bool(args.include_metadata)

    # If user didn't explicitly request metadata columns, default is OFF,
    # but you can enable it to preserve "48 months per person" structure.
    # Pragmatic default: OFF to match "17 categories".
    # Enable with: --include-metadata
    if args.include_metadata is False:
        include_metadata = False

    top5_log_uniform = not bool(args.no_top5_log_uniform)

    # Record run metadata
    os.makedirs(args.outdir, exist_ok=True)
    run_meta_path = os.path.join(args.outdir, "generation_metadata.json")

    run_meta = {
        "master_seed": int(seed),
        "people_per_cluster": int(args.people_per_cluster),
        "months": int(args.months),
        "top5_max_annual": float(args.top5_max),
        "include_metadata": include_metadata,
        "top5_log_uniform": top5_log_uniform,
        "files": [],
    }

    for c in clusters:
        meta = write_cluster_csv(
            outdir=args.outdir,
            cluster=c,
            people_per_cluster=int(args.people_per_cluster),
            months=int(args.months),
            seed=int(seed),
            include_metadata=include_metadata,
            top5_log_uniform=top5_log_uniform,
            chunk_people=int(args.chunk_people),
        )
        run_meta["files"].append(meta)

    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print(f"\nDone. Metadata written to: {run_meta_path}")


if __name__ == "__main__":
    main()
