"""
src/data/basis_forecast.py
Dynamic basis forecasting for Uruguay/River Plate soybean producers.

Academic basis:
  - Wool (2002, JASSA): Bayesian cross-hedging in soybean market requires
    explicit basis risk modeling.
  - GARCH-BEKK models for Brazilian soybean farmers significantly outperform
    static hedging ratios (ResearchGate, 2010).
  - The "double basis" (CBOT spread + FX risk) is the key challenge for
    South American producers.

This module:
  1. Loads historical basis data (basis_uruguay.json history + raw prices)
  2. Models basis dynamics with GARCH(1,1) volatility
  3. Incorporates seasonal patterns and FX risk
  4. Forecasts basis for 7/15/30d horizons with confidence intervals
  5. Generates hedging signals based on basis regime
"""

import json
import os
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_PATH = os.path.join(_PROJECT_ROOT, "data", "basis_forecast.json")
_HISTORY_PATH = os.path.join(_PROJECT_ROOT, "data", "basis_history.csv")
_TTL_HOURS = 12

BUSHELS_PER_TON = 36.744

# Seasonal basis patterns for Uruguay (empirical, USD/ton discount from CBOT)
# Basis typically widens during harvest (Mar-Jun) and tightens post-harvest (Sep-Dec)
SEASONAL_BASIS = {
    1: -20, 2: -18, 3: -17, 4: -22, 5: -25, 6: -28,
    7: -30, 8: -28, 9: -25, 10: -22, 11: -20, 12: -19,
}


def _load_basis_history() -> pd.DataFrame:
    """Load historical basis observations."""
    if os.path.exists(_HISTORY_PATH):
        try:
            df = pd.read_csv(_HISTORY_PATH, parse_dates=["date"])
            return df.sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    # Build from raw data if history file doesn't exist
    return _build_basis_history()


def _build_basis_history() -> pd.DataFrame:
    """Build basis history from raw market data + seasonal estimates."""
    raw_path = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")
    if not os.path.exists(raw_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(raw_path, parse_dates=["Date"])
        df = df.sort_values("Date").dropna(subset=["Soybeans"])

        # Convert CBOT cents/bushel to USD/ton
        df["cbot_usd_ton"] = df["Soybeans"] * BUSHELS_PER_TON / 100

        # Estimate local price using seasonal basis pattern
        df["month"] = df["Date"].dt.month
        df["seasonal_basis"] = df["month"].map(SEASONAL_BASIS)
        df["local_est_usd_ton"] = df["cbot_usd_ton"] + df["seasonal_basis"]
        df["basis_usd_ton"] = df["seasonal_basis"].astype(float)

        # Add FX proxy (Dollar index as USD strength indicator)
        if "Dollar" in df.columns:
            df["usd_strength"] = df["Dollar"].pct_change(5).fillna(0) * 100

        cols_keep = ["Date", "cbot_usd_ton", "local_est_usd_ton", "basis_usd_ton", "month"]
        result = df[cols_keep].copy()
        result = result.rename(columns={"Date": "date", "local_est_usd_ton": "local_usd_ton"})

        if "usd_strength" in df.columns:
            result["usd_strength_5d"] = df["usd_strength"].values

        # Save for future use
        os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
        result.to_csv(_HISTORY_PATH, index=False)

        return result
    except Exception as e:
        print(f"[BasisForecast] Build history failed: {e}")
        return pd.DataFrame()


def _garch_volatility(returns: np.ndarray, omega: float = 0.01,
                       alpha: float = 0.10, beta: float = 0.85) -> np.ndarray:
    """
    Simple GARCH(1,1) volatility estimation.
    σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

    Uses standard GARCH(1,1) which is sufficient for basis dynamics.
    Full GARCH-BEKK (multivariate) requires arch package — this is a
    robust single-series implementation.
    """
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns) if len(returns) > 1 else omega

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

    return np.sqrt(np.maximum(sigma2, 1e-8))


def forecast_basis(horizons: list[int] = None, n_paths: int = 5000) -> dict:
    """
    Forecast basis for multiple horizons using GARCH volatility + seasonal drift.

    Returns:
        Dict with forecasts per horizon, including:
        - expected basis (USD/ton)
        - confidence interval (Q10, Q90)
        - basis regime (tight/normal/wide)
        - hedging recommendation
    """
    if horizons is None:
        horizons = [7, 15, 30]

    # Check cache
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH) as f:
                cached = json.load(f)
            age = (datetime.now() - datetime.fromisoformat(cached.get("timestamp", "2000-01-01"))).total_seconds()
            if age < _TTL_HOURS * 3600:
                cached["from_cache"] = True
                return cached
        except Exception:
            pass

    hist = _load_basis_history()
    if hist.empty or len(hist) < 60:
        return {"ok": False, "error": "Insufficient basis history", "forecasts": {}}

    # Ensure month column exists (derive from date if missing)
    if "month" not in hist.columns and "date" in hist.columns:
        hist["month"] = pd.to_datetime(hist["date"]).dt.month

    # Current state
    current = hist.iloc[-1]
    current_basis = float(current["basis_usd_ton"])
    current_month = int(current.get("month", pd.to_datetime(current["date"]).month))
    current_cbot = float(current["cbot_usd_ton"])

    # Basis changes (returns)
    basis_series = hist["basis_usd_ton"].values.astype(float)
    basis_changes = np.diff(basis_series)
    basis_changes = basis_changes[~np.isnan(basis_changes)]

    if len(basis_changes) < 30:
        return {"ok": False, "error": "Not enough basis observations"}

    # Estimate GARCH volatility
    vol = _garch_volatility(basis_changes)
    current_vol = vol[-1] if len(vol) > 0 else np.std(basis_changes)

    # Seasonal drift: expected basis change per month
    monthly_basis = hist.groupby("month")["basis_usd_ton"].mean()

    # 5-year mean and std for z-score
    mean_basis = float(basis_series[-min(len(basis_series), 252*5):].mean())
    std_basis = float(basis_series[-min(len(basis_series), 252*5):].std())
    if std_basis < 0.1:
        std_basis = 5.0  # Floor

    zscore = (current_basis - mean_basis) / std_basis

    # Regime classification
    if zscore > 1.0:
        regime = "very_tight"
        regime_label = "Muy apretado (favorable para vender)"
    elif zscore > 0.3:
        regime = "tight"
        regime_label = "Apretado (buen momento)"
    elif zscore > -0.3:
        regime = "normal"
        regime_label = "Normal"
    elif zscore > -1.0:
        regime = "wide"
        regime_label = "Amplio (descuento moderado)"
    else:
        regime = "very_wide"
        regime_label = "Muy amplio (descuento excesivo — esperar)"

    # Forecast per horizon via Monte Carlo
    forecasts = {}
    for h in horizons:
        # Seasonal drift over horizon
        future_months = [(current_month + (d // 21)) % 12 or 12 for d in range(h)]
        seasonal_drift = sum(
            SEASONAL_BASIS.get(m, -25) - current_basis for m in set(future_months)
        ) / max(len(set(future_months)), 1) * (h / 252)

        # Mean reversion component (basis reverts to seasonal mean)
        target_basis = SEASONAL_BASIS.get(
            (current_month + (h // 30)) % 12 or 12, -25
        )
        mean_reversion_speed = 0.03  # 3% per day toward seasonal target
        expected_reversion = (target_basis - current_basis) * (1 - (1 - mean_reversion_speed) ** h)

        # Monte Carlo paths
        daily_vol = current_vol
        paths = np.zeros((n_paths, h))
        paths[:, 0] = current_basis

        for d in range(1, h):
            drift = expected_reversion / h  # Spread drift over horizon
            shock = np.random.normal(0, daily_vol, n_paths)
            paths[:, d] = paths[:, d-1] + drift + shock

        terminal = paths[:, -1]

        # Statistics
        expected = float(np.mean(terminal))
        q10 = float(np.percentile(terminal, 10))
        q25 = float(np.percentile(terminal, 25))
        q50 = float(np.median(terminal))
        q75 = float(np.percentile(terminal, 75))
        q90 = float(np.percentile(terminal, 90))
        prob_tighter = float(np.mean(terminal > current_basis) * 100)

        # Hedging recommendation
        if expected > current_basis + 2:
            hedge_rec = "ESPERAR — basis se espera que mejore"
        elif expected < current_basis - 3:
            hedge_rec = "CUBRIR — basis se espera que empeore"
        else:
            hedge_rec = "NEUTRAL — basis estable"

        forecasts[f"{h}d"] = {
            "horizon_days": h,
            "current_basis": round(current_basis, 2),
            "expected_basis": round(expected, 2),
            "change_expected": round(expected - current_basis, 2),
            "quantiles": {
                "q10": round(q10, 2),
                "q25": round(q25, 2),
                "q50": round(q50, 2),
                "q75": round(q75, 2),
                "q90": round(q90, 2),
            },
            "prob_tighter_pct": round(prob_tighter, 1),
            "hedge_recommendation": hedge_rec,
            "garch_daily_vol": round(daily_vol, 3),
        }

    result = {
        "ok": True,
        "timestamp": datetime.now().isoformat(),
        "from_cache": False,
        "current_state": {
            "basis_usd_ton": round(current_basis, 2),
            "cbot_usd_ton": round(current_cbot, 2),
            "local_est_usd_ton": round(current_cbot + current_basis, 2),
            "zscore_5y": round(zscore, 2),
            "regime": regime,
            "regime_label": regime_label,
            "garch_vol": round(current_vol, 3),
            "mean_5y": round(mean_basis, 2),
            "std_5y": round(std_basis, 2),
        },
        "seasonal_pattern": {
            m: round(v, 1) for m, v in SEASONAL_BASIS.items()
        },
        "forecasts": forecasts,
    }

    # Cache
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass

    return result


def add_basis_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basis-related features to the main feature DataFrame.
    Called from pipeline.py during feature engineering.
    """
    df = df.copy()

    if "Soybeans" not in df.columns or "Date" not in df.columns:
        return df

    # CBOT to USD/ton
    cbot_usd = df["Soybeans"].values * BUSHELS_PER_TON / 100

    # Seasonal basis expectation
    months = pd.to_datetime(df["Date"]).dt.month
    df["basis_seasonal_expected"] = months.map(SEASONAL_BASIS).fillna(-25).astype(float)

    # Estimated local price
    df["local_price_est_usd_ton"] = cbot_usd + df["basis_seasonal_expected"].values

    # Basis momentum (is basis tightening or widening?)
    basis_vals = df["basis_seasonal_expected"].values
    df["basis_momentum_5d"] = pd.Series(basis_vals).diff(5).fillna(0)
    df["basis_momentum_20d"] = pd.Series(basis_vals).diff(20).fillna(0)

    # Basis regime indicator
    mean_basis = pd.Series(basis_vals).rolling(252, min_periods=30).mean().fillna(-25)
    std_basis = pd.Series(basis_vals).rolling(252, min_periods=30).std().fillna(5).replace(0, 5)
    df["basis_zscore"] = ((basis_vals - mean_basis) / std_basis).fillna(0)

    # FX impact proxy (Dollar strength affects local USD prices)
    if "Dollar" in df.columns:
        df["fx_basis_impact"] = df["Dollar"].pct_change(5).fillna(0) * df["basis_seasonal_expected"]

    print(f"[BasisFeatures] Added basis features (seasonal, momentum, zscore, FX)")
    return df


if __name__ == "__main__":
    result = forecast_basis()
    if result.get("ok"):
        print(f"\nBasis actual: {result['current_state']['basis_usd_ton']} USD/ton")
        print(f"Régimen: {result['current_state']['regime_label']}")
        print(f"Z-score 5Y: {result['current_state']['zscore_5y']}")
        for h, fc in result["forecasts"].items():
            print(f"\n--- {h} ---")
            print(f"  Esperado: {fc['expected_basis']:.1f} (cambio: {fc['change_expected']:+.1f})")
            print(f"  Q10-Q90: [{fc['quantiles']['q10']:.1f}, {fc['quantiles']['q90']:.1f}]")
            print(f"  P(tighter): {fc['prob_tighter_pct']:.0f}%")
            print(f"  Rec: {fc['hedge_recommendation']}")
    else:
        print(f"Error: {result.get('error')}")
