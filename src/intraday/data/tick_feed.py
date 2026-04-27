"""
src/intraday/data/tick_feed.py
Fetcher de barras intradía vía yfinance (Fase 0 — sin costo).

Limitaciones conocidas de yfinance para futuros (ZS=F):
  - interval="1m"  → solo últimos 7 días disponibles
  - interval="5m"  → solo últimos 60 días
  - interval="15m" → solo últimos 60 días
  - interval="60m" → últimos 730 días
  - Datos de futuros pueden tener huecos en sesiones electrónicas (Globex)
  - No incluye order book (solo OHLCV)
  - Volume puede ser cero en barras de baja actividad

Para Fase 1 reemplazar por CME DataMine (histórico) + broker API (live).

Cache:
  data/intraday/zs_5m.parquet
  data/intraday/zs_15m.parquet
  data/intraday/zs_60m.parquet

API:
  fetch_intraday_bars(interval, period, symbol) -> pd.DataFrame
  load_cached_bars(interval, symbol) -> pd.DataFrame | None
  diagnose_coverage(df, interval) -> dict
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, time
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_CACHE_DIR    = os.path.join(_PROJECT_ROOT, "data", "intraday")

# Horarios CME Globex para ZS (todos en CT — Chicago):
#   Domingo 19:00 → Viernes 13:20 con break diario 07:45–08:30
# Sesión "regular" (RTH) = 08:30–13:20 CT (la de mayor liquidez)
_RTH_OPEN_CT  = time(8, 30)
_RTH_CLOSE_CT = time(13, 20)

# Periodos máximos que yfinance acepta por intervalo
_YF_MAX_PERIOD = {
    "1m":  "7d",
    "2m":  "60d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "90m": "60d",
    "1h":  "730d",
}

_DEFAULT_SYMBOL = "ZS=F"


def _cache_path(interval: str, symbol: str) -> str:
    safe_sym = symbol.replace("=", "").replace("^", "").lower()
    return os.path.join(_CACHE_DIR, f"{safe_sym}_{interval}.parquet")


def fetch_intraday_bars(
    interval: str = "5m",
    period: Optional[str] = None,
    symbol: str = _DEFAULT_SYMBOL,
    use_cache: bool = True,
    cache_max_age_min: int = 30,
) -> pd.DataFrame:
    """
    Fetch barras intradía OHLCV vía yfinance.

    Args:
      interval: "1m", "5m", "15m", "60m", etc.
      period:   "7d", "60d", "730d". Si None, usa máximo permitido.
      symbol:   "ZS=F" (soja CBOT), "ZC=F" (maíz), etc.
      use_cache: si True y cache fresco, lo retorna sin llamar a yfinance.
      cache_max_age_min: edad máxima del cache antes de refrescar.

    Returns:
      DataFrame con columnas: ['datetime','open','high','low','close','volume']
      Index: RangeIndex. datetime es tz-aware (UTC normalizado).
    """
    if interval not in _YF_MAX_PERIOD:
        raise ValueError(f"interval inválido: {interval}. Opciones: {list(_YF_MAX_PERIOD)}")
    period = period or _YF_MAX_PERIOD[interval]
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _cache_path(interval, symbol)

    # ── Cache hit ───────────────────────────────────────────────────────
    if use_cache and os.path.exists(path):
        age_min = (datetime.now().timestamp() - os.path.getmtime(path)) / 60
        if age_min < cache_max_age_min:
            try:
                df = pd.read_parquet(path)
                print(f"[tick_feed] cache hit {symbol} {interval} ({len(df)} bars, age={age_min:.0f}min)")
                return df
            except Exception:
                pass

    # ── Fetch yfinance ──────────────────────────────────────────────────
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance no instalado: pip install yfinance")

    print(f"[tick_feed] fetching {symbol} {interval} period={period} ...")
    raw = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        prepost=True,   # incluir Globex extendido para futuros
    )
    if raw is None or raw.empty:
        print(f"[tick_feed] WARNING — yfinance devolvió vacío para {symbol} {interval}")
        return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])

    # yfinance devuelve MultiIndex de columnas si pasamos lista; normalizamos
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index().rename(columns={
        "Datetime": "datetime", "Date": "datetime",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    keep = ["datetime","open","high","low","close","volume"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Normalizar a UTC tz-aware
    if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        else:
            df["datetime"] = df["datetime"].dt.tz_convert("UTC")

    df = df.dropna(subset=["close"]).sort_values("datetime").reset_index(drop=True)

    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        print(f"[tick_feed] no pude escribir cache parquet ({e}); usando csv fallback")
        df.to_csv(path.replace(".parquet", ".csv"), index=False)

    print(f"[tick_feed] OK {symbol} {interval}: {len(df)} bars "
          f"[{df['datetime'].min()} → {df['datetime'].max()}]")
    return df


def load_cached_bars(interval: str = "5m", symbol: str = _DEFAULT_SYMBOL) -> Optional[pd.DataFrame]:
    """Carga cache sin tocar la red. None si no existe."""
    path = _cache_path(interval, symbol)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────
# Diagnósticos de calidad — el corazón del experimento Fase 0
# ──────────────────────────────────────────────────────────────────────────

def _interval_to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    raise ValueError(interval)


def diagnose_coverage(df: pd.DataFrame, interval: str = "5m") -> dict:
    """
    Reporta calidad de los datos intradía:
      - rango temporal cubierto
      - barras esperadas vs observadas (RTH y total)
      - huecos > 1 barra
      - barras con volume==0
      - distribución por hora del día (CT)
      - sesiones con < N% de cobertura

    Returns dict con métricas clave para decidir GO/NO-GO Fase 0.
    """
    if df.empty:
        return {"status": "EMPTY"}

    bar_min = _interval_to_minutes(interval)
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Convertir a CT para análisis de sesiones
    df["dt_ct"]   = df["datetime"].dt.tz_convert("America/Chicago")
    df["date_ct"] = df["dt_ct"].dt.date
    df["time_ct"] = df["dt_ct"].dt.time
    df["hour_ct"] = df["dt_ct"].dt.hour
    df["dow"]     = df["dt_ct"].dt.dayofweek  # 0=Mon

    # Marcar barras dentro de RTH (08:30–13:20 CT)
    in_rth = (df["time_ct"] >= _RTH_OPEN_CT) & (df["time_ct"] < _RTH_CLOSE_CT)
    df["is_rth"] = in_rth
    rth = df[in_rth].copy()

    # Esperado en RTH: 4h50min = 290min → bars/día = 290/bar_min
    bars_per_rth_session = 290 // bar_min

    # Sesiones únicas (días hábiles con datos RTH)
    sessions = rth["date_ct"].unique()
    n_sessions = len(sessions)

    # Cobertura RTH por sesión
    rth_per_session = rth.groupby("date_ct").size()
    coverage_pct = (rth_per_session / bars_per_rth_session * 100).round(1)
    weak_sessions = coverage_pct[coverage_pct < 80]

    # Huecos: diff > bar_min entre observaciones consecutivas
    df_sorted = df.sort_values("datetime")
    deltas = df_sorted["datetime"].diff().dt.total_seconds() / 60
    gaps = deltas[deltas > bar_min * 1.5]
    big_gaps = deltas[deltas > 60]  # >1h: probable break o festivo

    # Volumen cero
    zero_vol = (df["volume"].fillna(0) == 0).sum()

    # Distribución por hora CT
    hour_dist = df.groupby("hour_ct").size().to_dict()

    # NaNs
    nan_close = df["close"].isna().sum()
    nan_vol   = df["volume"].isna().sum()

    # Rango temporal cubierto
    span_days = (df["datetime"].max() - df["datetime"].min()).days

    result = {
        "status":              "OK" if n_sessions >= 5 else "INSUFFICIENT",
        "interval":            interval,
        "n_bars_total":        int(len(df)),
        "n_bars_rth":          int(len(rth)),
        "rth_pct_of_total":    round(len(rth) / len(df) * 100, 1) if len(df) else 0,
        "first_bar":           str(df["datetime"].min()),
        "last_bar":            str(df["datetime"].max()),
        "span_days":           int(span_days),
        "n_sessions_rth":      int(n_sessions),
        "bars_per_session_expected": int(bars_per_rth_session),
        "bars_per_session_mean":     round(float(rth_per_session.mean()), 1) if n_sessions else 0,
        "bars_per_session_min":      int(rth_per_session.min()) if n_sessions else 0,
        "bars_per_session_max":      int(rth_per_session.max()) if n_sessions else 0,
        "weak_sessions_count":       int((coverage_pct < 80).sum()),
        "weak_sessions_pct":         round((coverage_pct < 80).mean() * 100, 1) if n_sessions else 0,
        "n_gaps":               int(len(gaps)),
        "n_big_gaps_gt_1h":     int(len(big_gaps)),
        "max_gap_minutes":      round(float(deltas.max()), 1) if len(deltas) else 0,
        "zero_volume_bars":     int(zero_vol),
        "zero_volume_pct":      round(zero_vol / len(df) * 100, 2),
        "nan_close":            int(nan_close),
        "nan_volume":           int(nan_vol),
        "hour_distribution_ct": {int(k): int(v) for k, v in hour_dist.items()},
    }
    return result


def print_coverage_report(diag: dict) -> None:
    """Imprime el reporte legible."""
    if diag.get("status") == "EMPTY":
        print("⚠️  No hay datos.")
        return

    print("=" * 68)
    print(f"  COBERTURA INTRADÍA — {diag['interval']}")
    print("=" * 68)
    print(f"  Status:               {diag['status']}")
    print(f"  Barras totales:       {diag['n_bars_total']:,}")
    print(f"  Barras RTH:           {diag['n_bars_rth']:,} "
          f"({diag['rth_pct_of_total']}% del total)")
    print(f"  Rango:                {diag['first_bar']}")
    print(f"  →                     {diag['last_bar']}")
    print(f"  Span:                 {diag['span_days']} días")
    print(f"  Sesiones RTH:         {diag['n_sessions_rth']}")
    print(f"  Bars/sesión esperado: {diag['bars_per_session_expected']}")
    print(f"  Bars/sesión obs:      mean={diag['bars_per_session_mean']} "
          f"min={diag['bars_per_session_min']} max={diag['bars_per_session_max']}")
    print(f"  Sesiones débiles:     {diag['weak_sessions_count']} "
          f"({diag['weak_sessions_pct']}% con <80% cobertura)")
    print(f"  Huecos:               {diag['n_gaps']} (>1.5x bar)")
    print(f"  Huecos >1h:           {diag['n_big_gaps_gt_1h']}")
    print(f"  Hueco máx (min):      {diag['max_gap_minutes']}")
    print(f"  Vol cero:             {diag['zero_volume_bars']} "
          f"({diag['zero_volume_pct']}%)")
    print(f"  NaN close:            {diag['nan_close']}")
    print(f"  NaN volume:           {diag['nan_volume']}")
    print()
    print("  Distribución por hora CT (top 10 más activas):")
    hours = sorted(diag["hour_distribution_ct"].items(),
                   key=lambda x: x[1], reverse=True)[:10]
    for h, n in hours:
        bar = "█" * int(n / max(diag["hour_distribution_ct"].values()) * 30)
        print(f"    {h:02d}:00 CT  {n:>6,}  {bar}")
    print("=" * 68)


# ──────────────────────────────────────────────────────────────────────────
# Score de viabilidad Fase 0
# ──────────────────────────────────────────────────────────────────────────

def viability_score(diag: dict) -> dict:
    """
    Devuelve verdict GO / WARN / NO-GO con razones.

    Criterios mínimos para Fase 0:
      - >= 20 sesiones RTH con datos
      - cobertura RTH media >= 85% de barras esperadas
      - sesiones débiles < 15%
      - vol cero < 25% en RTH
      - max gap dentro de RTH < 30min
    """
    if diag.get("status") in ("EMPTY", "INSUFFICIENT"):
        return {"verdict": "NO-GO", "reasons": ["datos insuficientes"]}

    reasons = []
    warnings = []

    if diag["n_sessions_rth"] < 20:
        reasons.append(f"solo {diag['n_sessions_rth']} sesiones RTH (min 20)")
    bars_exp = diag["bars_per_session_expected"]
    coverage = diag["bars_per_session_mean"] / bars_exp if bars_exp else 0
    if coverage < 0.85:
        reasons.append(f"cobertura RTH media {coverage:.0%} (<85%)")
    elif coverage < 0.95:
        warnings.append(f"cobertura RTH {coverage:.0%} aceptable pero subóptima")

    if diag["weak_sessions_pct"] > 15:
        reasons.append(f"{diag['weak_sessions_pct']}% sesiones débiles (>15%)")

    if diag["zero_volume_pct"] > 25:
        reasons.append(f"{diag['zero_volume_pct']}% barras con vol=0 (>25%)")
    elif diag["zero_volume_pct"] > 10:
        warnings.append(f"{diag['zero_volume_pct']}% vol=0 (señal débil)")

    if diag["n_big_gaps_gt_1h"] > diag["n_sessions_rth"] * 1.5:
        warnings.append("muchos gaps >1h (esperables: 1/día por overnight)")

    if reasons:
        return {"verdict": "NO-GO", "reasons": reasons, "warnings": warnings}
    if warnings:
        return {"verdict": "WARN", "reasons": [], "warnings": warnings}
    return {"verdict": "GO", "reasons": [], "warnings": []}


if __name__ == "__main__":
    # Smoke test: ZS=F a 5m y 15m
    for interval in ("5m", "15m", "60m"):
        print(f"\n{'#'*68}\n# ZS=F {interval}\n{'#'*68}")
        df = fetch_intraday_bars(interval=interval, use_cache=True)
        if df.empty:
            continue
        diag = diagnose_coverage(df, interval=interval)
        print_coverage_report(diag)
        verdict = viability_score(diag)
        print(f"\n  VEREDICTO Fase 0: {verdict['verdict']}")
        for r in verdict.get("reasons", []):
            print(f"    ❌ {r}")
        for w in verdict.get("warnings", []):
            print(f"    ⚠️  {w}")
