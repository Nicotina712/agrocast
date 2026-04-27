"""
src/model/monthly_robust.py
Forecast mensual robusto para 30d/60d/90d.

Reemplaza el problema del Ridge saturado (cap diario ±1% acumulado) usando:
  1) Holt-Winters / ETS si statsmodels está disponible
  2) Naive seasonal mean fallback si no

Salida: artifacts/monthly_forecast.csv con columnas Date, forecast, lower, upper.
"""

import os
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT_PATH     = os.path.join(_PROJECT_ROOT, "artifacts", "monthly_forecast.csv")


def _try_ets(series: pd.Series, horizon: int) -> Optional[pd.DataFrame]:
    """ETS con statsmodels si está disponible."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except Exception:
        return None
    try:
        # Resampleo mensual y ETS
        s = series.copy()
        s.index = pd.to_datetime(s.index)
        monthly = s.resample("ME").last().dropna()
        if len(monthly) < 24:
            return None

        model = ExponentialSmoothing(
            monthly, trend="add", seasonal="add", seasonal_periods=12,
            initialization_method="estimated"
        ).fit(optimized=True)
        n_months = max(1, int(np.ceil(horizon / 30)) + 1)
        fcast    = model.forecast(n_months)

        # Residuals para banda
        resid_std = float(np.std(model.resid)) if model.resid is not None else float(monthly.diff().std())

        # Interpolar a daily
        last_date = series.index.max()
        future_dates = pd.date_range(last_date + timedelta(days=1),
                                      last_date + timedelta(days=horizon), freq="D")
        # Fill mensual → diario por interpolación lineal entre fines de mes
        fcast.index = pd.to_datetime(fcast.index)
        merged = pd.concat([
            pd.Series([float(series.iloc[-1])], index=[last_date]),
            fcast,
        ]).sort_index()
        daily = merged.reindex(
            pd.date_range(merged.index.min(), merged.index.max(), freq="D")
        ).interpolate("linear")
        daily = daily.loc[future_dates.min():future_dates.max()]

        # Banda creciente con sqrt(t)
        days_ahead = np.arange(1, len(daily) + 1)
        sigma = resid_std * np.sqrt(days_ahead / 30.0)
        out = pd.DataFrame({
            "Date":     daily.index,
            "forecast": daily.values,
            "upper":    daily.values + 1.5 * sigma,
            "lower":    daily.values - 1.5 * sigma,
        })
        return out
    except Exception:
        return None


def _seasonal_naive(series: pd.Series, horizon: int) -> pd.DataFrame:
    """Fallback: precio actual + retorno estacional medio del mes correspondiente."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    rets = s.pct_change(21)  # ~ retorno mensual
    df = pd.DataFrame({"price": s, "month": s.index.month, "ret": rets})
    seasonal = df.groupby("month")["ret"].mean().fillna(0.0)

    last_date  = s.index.max()
    last_price = float(s.iloc[-1])
    future_dates = pd.date_range(last_date + timedelta(days=1),
                                  last_date + timedelta(days=horizon), freq="D")

    forecasts = []
    cur = last_price
    for i, d in enumerate(future_dates):
        # acumular retorno estacional proporcionalmente
        m = d.month
        daily_ret = seasonal.get(m, 0.0) / 21.0
        cur = cur * (1 + daily_ret)
        forecasts.append(cur)

    sigma_daily = float(s.pct_change().std()) * last_price
    sigma = sigma_daily * np.sqrt(np.arange(1, horizon + 1))
    out = pd.DataFrame({
        "Date":     future_dates,
        "forecast": forecasts,
        "upper":    np.array(forecasts) + 1.5 * sigma,
        "lower":    np.array(forecasts) - 1.5 * sigma,
    })
    return out


def build_monthly_forecast(horizon_days: int = 90) -> Optional[pd.DataFrame]:
    """Genera forecast robusto y persiste en artifacts/monthly_forecast.csv."""
    raw_path = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")
    if not os.path.exists(raw_path):
        return None
    df = pd.read_csv(raw_path, parse_dates=["Date"]).sort_values("Date")
    if df.empty:
        return None
    s = df.set_index("Date")["Soybeans"].astype(float)

    fc = _try_ets(s, horizon_days)
    method = "ETS"
    if fc is None:
        fc = _seasonal_naive(s, horizon_days)
        method = "seasonal_naive"

    fc["method"] = method
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    fc.to_csv(_OUT_PATH, index=False)
    print(f"[monthly_robust] {method}: {len(fc)} días -> {_OUT_PATH}")
    return fc


if __name__ == "__main__":
    out = build_monthly_forecast()
    if out is not None:
        print(out.head())
        print(out.tail())
