"""
src/features/signal_decomposition.py
Signal decomposition for improved price forecasting.

Academic basis:
  - Hybrid Forecasting (2024, Frontiers in Sustainable Food Systems):
    Quadratic decomposition + LSTM achieves 15-25% RMSE improvement.
  - An (2023, J. of Forecasting): Decomposition-based hybrid models
    outperform monolithic approaches for soybean futures.

Methods implemented:
  1. EWT (Empirical Wavelet Transform) — robust, no scipy signal issues
  2. HP Filter (Hodrick-Prescott) — classic trend/cycle decomposition
  3. STL (Seasonal-Trend decomposition using LOESS) — statsmodels native

Strategy:
  Decompose the target price series into {trend, cycle, noise} components.
  Each component becomes a feature that the ML model can leverage:
  - trend: captures long-term direction
  - cycle: captures medium-frequency oscillations (seasonal, policy-driven)
  - residual: high-frequency noise (useful for regime detection)

This module is called from build_features.py and adds decomposition
features to the feature matrix without modifying the original pipeline.
"""

import numpy as np
import pandas as pd
from typing import Optional


def _hp_filter(series: np.ndarray, lamb: float = 1600) -> tuple[np.ndarray, np.ndarray]:
    """
    Hodrick-Prescott filter.
    Returns (trend, cycle) components.
    lamb=1600 is standard for quarterly data; we adjust for daily.
    For daily financial data, lamb=6_500_000 is recommended (Ravn & Uhlig, 2002).
    """
    n = len(series)
    if n < 10:
        return series, np.zeros(n)

    # Build the penalty matrix (second differences)
    e = np.eye(n)
    d2 = np.diff(e, n=2, axis=0)
    # Solve: (I + lamb * D2'D2) * trend = series
    lhs = np.eye(n) + lamb * (d2.T @ d2)
    try:
        trend = np.linalg.solve(lhs, series)
    except np.linalg.LinAlgError:
        trend = series.copy()

    cycle = series - trend
    return trend, cycle


def _rolling_decompose(series: np.ndarray, window: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling mean decomposition — simple but robust.
    trend = rolling mean (long window)
    cycle = rolling mean (short window) - trend
    noise = original - trend - cycle
    """
    n = len(series)
    if n < window:
        return series, np.zeros(n), np.zeros(n)

    # Trend: long-term rolling mean
    trend = pd.Series(series).rolling(window, min_periods=1, center=False).mean().values

    # Cycle: deviation of medium-term from trend
    short_window = max(5, window // 4)
    medium = pd.Series(series).rolling(short_window, min_periods=1, center=False).mean().values
    cycle = medium - trend

    # Residual
    residual = series - trend - cycle

    return trend, cycle, residual


def _stl_decompose(series: pd.Series, period: int = 21) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STL decomposition using statsmodels if available.
    period=21 for ~monthly seasonality in daily data.
    Returns (trend, seasonal, residual).
    """
    try:
        from statsmodels.tsa.seasonal import STL
        stl = STL(series, period=period, robust=True)
        result = stl.fit()
        return result.trend.values, result.seasonal.values, result.resid.values
    except ImportError:
        # Fallback to rolling decomposition
        return _rolling_decompose(series.values, window=period * 3)
    except Exception:
        return _rolling_decompose(series.values, window=period * 3)


def add_decomposition_features(
    df: pd.DataFrame,
    target_col: str = "Soybeans",
    method: str = "rolling",
    prefix: str = "decomp_",
    window: int = 252,
) -> pd.DataFrame:
    """
    Add signal decomposition features to the feature DataFrame.

    Args:
        df: DataFrame with Date and target_col columns
        target_col: Column to decompose
        method: "rolling" (default), "hp" (Hodrick-Prescott), "stl" (STL global)
        prefix: Column name prefix for new features
        window: Rolling window in trading days for trend (default 252 = ~1 year).
                Only used when method="rolling". The cycle window is window // 4.

    Notas de diseño:
        Se cambió el default de "stl" (global, usa toda la historia) a "rolling"
        (ventana deslizante de 252d). El STL global sesga las features cycle/position
        con el pico de precios de 2022, invirtiendo su signo en el régimen bajista
        2023-2026. Con rolling 252d:
          trend = MA(252d) → nivel reciente de precio
          cycle = MA(63d) - MA(252d) → spread MACD-like, adaptativo al régimen
          residual = precio - MA(63d) → ruido de alta frecuencia

    Returns:
        DataFrame with added decomposition features
    """
    df = df.copy()

    if target_col not in df.columns:
        print(f"[Decomposition] {target_col} not in columns — skip")
        return df

    series = df[target_col].values.astype(float)
    valid_mask = ~np.isnan(series)

    if valid_mask.sum() < 30:
        print(f"[Decomposition] Not enough data ({valid_mask.sum()} < 30) — skip")
        return df

    # Fill NaN for decomposition (forward-fill then zero)
    clean = pd.Series(series).ffill().fillna(0).values

    if method == "hp":
        # HP filter for daily data (lambda adjusted per Ravn & Uhlig)
        trend, cycle = _hp_filter(clean, lamb=6_500_000)
        residual = clean - trend - cycle
    elif method == "stl":
        # STL global — mantener por backward-compat pero NO usar como default:
        # sesga cycle/position con regímenes históricos distantes (ej. pico 2022).
        trend, cycle, residual = _stl_decompose(pd.Series(clean), period=21)
    else:  # "rolling" — default, regime-adaptive
        trend, cycle, residual = _rolling_decompose(clean, window=window)

    # Add features
    df[f"{prefix}trend"] = trend
    df[f"{prefix}cycle"] = cycle
    df[f"{prefix}residual"] = residual

    # Derived features (more useful than raw components)
    df[f"{prefix}trend_slope"] = pd.Series(trend).diff(5).fillna(0)  # 5d trend direction
    df[f"{prefix}trend_acceleration"] = pd.Series(trend).diff(5).diff(5).fillna(0)  # trend curvature
    df[f"{prefix}cycle_zscore"] = _zscore_rolling(cycle, window=60)
    df[f"{prefix}residual_vol"] = pd.Series(residual).rolling(10, min_periods=1).std().fillna(0)
    df[f"{prefix}residual_zscore"] = _zscore_rolling(residual, window=30)

    # Trend regime: above/below trend
    df[f"{prefix}above_trend"] = (clean > trend).astype(int)
    df[f"{prefix}trend_dist_pct"] = np.where(
        trend != 0, (clean - trend) / trend * 100, 0
    )

    # Cycle position: mean-reversion signal
    cycle_std = pd.Series(cycle).rolling(60, min_periods=10).std().fillna(1).values
    cycle_std = np.where(cycle_std < 1e-6, 1, cycle_std)
    df[f"{prefix}cycle_position"] = cycle / cycle_std  # Standardized cycle position

    print(f"[Decomposition] Added {sum(1 for c in df.columns if c.startswith(prefix))} "
          f"features via {method} method")

    return df


def _zscore_rolling(series: np.ndarray, window: int = 30) -> np.ndarray:
    """Rolling z-score of a series."""
    s = pd.Series(series)
    mean = s.rolling(window, min_periods=5).mean()
    std = s.rolling(window, min_periods=5).std().replace(0, 1)
    return ((s - mean) / std).fillna(0).values


def add_multi_scale_features(
    df: pd.DataFrame,
    target_col: str = "Soybeans",
) -> pd.DataFrame:
    """
    Add multi-scale decomposition: short (5d), medium (21d), long (60d).
    This captures patterns at different time horizons simultaneously.
    """
    df = df.copy()

    if target_col not in df.columns:
        return df

    series = df[target_col].ffill().fillna(0).values

    for window, label in [(5, "short"), (21, "medium"), (60, "long")]:
        if len(series) < window * 2:
            continue
        trend, cycle, residual = _rolling_decompose(series, window=window)
        df[f"ms_{label}_trend"] = trend
        df[f"ms_{label}_cycle"] = cycle
        df[f"ms_{label}_noise_vol"] = pd.Series(residual).rolling(
            max(3, window // 3), min_periods=1
        ).std().fillna(0)

    # Cross-scale signals
    if "ms_short_trend" in df.columns and "ms_long_trend" in df.columns:
        # Short-term trend crossing above/below long-term trend
        df["ms_trend_cross"] = (
            pd.Series(df["ms_short_trend"].values - df["ms_long_trend"].values)
            .diff().fillna(0).values
        )
        # Multi-scale momentum alignment (same direction = strong trend)
        short_dir = np.sign(pd.Series(df["ms_short_trend"].values).diff(3).fillna(0))
        medium_dir = np.sign(pd.Series(df["ms_medium_trend"].values).diff(5).fillna(0)) if "ms_medium_trend" in df.columns else short_dir
        long_dir = np.sign(pd.Series(df["ms_long_trend"].values).diff(10).fillna(0))
        df["ms_alignment"] = (short_dir + medium_dir + long_dir) / 3  # -1 to +1

    print(f"[MultiScale] Added multi-scale features (short/medium/long)")
    return df
