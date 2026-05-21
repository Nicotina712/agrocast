"""
src/features/build_features.py
Genera features de mercado: lags, cambios, estacionalidad, momentum, spreads.
"""

import os
import numpy as np
import pandas as pd

# Ruta al archivo de rollover flags (relativo a este módulo)
_SCRIPT_DIR   = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_ROLLOVER_PATH = os.path.join(_PROJECT_ROOT, "data", "rollover_flags.csv")


def _load_rollover_dates() -> set:
    """
    Lee data/rollover_flags.csv y retorna un set de fechas flaggeadas.
    Si el archivo no existe, retorna un set vacío.
    """
    if not os.path.exists(_ROLLOVER_PATH):
        return set()
    try:
        df = pd.read_csv(_ROLLOVER_PATH, parse_dates=["Date"])
        flagged = df.loc[df["flagged"] == True, "Date"]
        dates = set(pd.to_datetime(flagged).dt.normalize().tolist())
        if dates:
            print(f"[Rollover] build_features: {len(dates)} dias de rollover cargados desde {_ROLLOVER_PATH}")
        return dates
    except Exception as e:
        print(f"[WARN] No se pudo leer rollover_flags.csv: {e}")
        return set()


def safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods).replace([np.inf, -np.inf], 0)


def compute_rsi(series: pd.Series, periods: int = 14) -> pd.Series:
    """RSI estándar de Wilder."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(periods).mean()
    loss  = (-delta.clip(upper=0)).rolling(periods).mean()
    rs    = gain / loss.replace(0, np.nan).fillna(1)
    return (100 - (100 / (1 + rs))).fillna(50)  # 50 = neutral cuando no hay datos


def make_features(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    exog_cols: list[str],
) -> pd.DataFrame:
    d = df.copy().sort_values(date_col).reset_index(drop=True)

    # ── Escala del target ────────────────────────────────────────
    max_val = d[target_col].max()
    if max_val > 5000:
        d[target_col] = d[target_col] / 100
        print(f"⚠️  Soybeans dividido por 100 (max era {max_val:.0f})")
    elif max_val > 2500:
        d[target_col] = d[target_col] / 10
        print(f"⚠️  Soybeans dividido por 10 (max era {max_val:.0f})")

    d[target_col] = d[target_col].clip(lower=100, upper=2500)

    # ── Log del target ───────────────────────────────────────────
    d["Soybeans_log"] = np.log1p(d[target_col])

    # ── Lags del target ──────────────────────────────────────────
    for lag in (1, 7, 30):
        d[f"{target_col}_lag{lag}"] = d[target_col].shift(lag)

    # ── Variables exógenas ───────────────────────────────────────
    for col in exog_cols:
        if col not in d.columns:
            continue
        d[col] = d[col].ffill()
        d[f"{col}_lag1"] = d[col].shift(1)
        d[f"{col}_lag7"] = d[col].shift(7)
        d[f"{col}_chg1"] = safe_pct_change(d[col], 1)
        d[f"{col}_chg7"] = safe_pct_change(d[col], 7)

    # ── Momentum del precio ──────────────────────────────────────
    # Captura tendencias de corto y mediano plazo
    d["mom_5d"]  = safe_pct_change(d[target_col], 5)
    d["mom_20d"] = safe_pct_change(d[target_col], 20)
    d["mom_60d"] = safe_pct_change(d[target_col], 60)

    # ── RSI (14 períodos) ────────────────────────────────────────
    # >70 = sobrecomprado (posible SELL), <30 = sobrevendido (posible BUY)
    d["rsi_14"] = compute_rsi(d[target_col], 14)

    # ── Volatilidad realizada ────────────────────────────────────
    d["vol_10d"] = d[target_col].pct_change().rolling(10).std().fillna(0)
    d["vol_30d"] = d[target_col].pct_change().rolling(30).std().fillna(0)
    # vol_60d agregada tras experimento H1: el naive 60d le gana al naive 30d
    # por +25 % en MAE de predicción de σ futura (ver experiments_h1_h2_h6).
    # Es el baseline de vol más fuerte y debe estar disponible para los
    # detectores de régimen y bandas.
    d["vol_60d"] = d[target_col].pct_change().rolling(60).std().fillna(0)

    # ── Features de spike y co-driver (post-spike fade research) ──
    # Test empírico (scripts/test_post_spike_fade.py) mostró:
    #   - Spikes >+5.5 %/5d sin co-driver: -1.58 % en 30d siguientes (p=0.013)
    #   - Spikes >+5.5 %/5d con oil co-driver: +1.82 % en 30d (rally fundamental)
    #   - Spikes con RSI<50: -5.19 % en 30d (rebote falso)
    # Estas features explícitan la interacción para que el modelo la capture.
    ret_5d = d[target_col].pct_change(5)
    spike_thr_high = float(ret_5d.quantile(0.95))   # top 5%
    d["is_spike_5d"]      = (ret_5d >= spike_thr_high).astype(float)
    if "Oil" in d.columns:
        oil_5d = d["Oil"].pct_change(5)
        oil_thr = float(oil_5d.quantile(0.75))
        d["spike_with_oil_codriver"] = ((ret_5d >= spike_thr_high) & (oil_5d >= oil_thr)).astype(float)
        d["spike_without_oil"]       = ((ret_5d >= spike_thr_high) & (oil_5d <  oil_thr)).astype(float)
    # Magnitud del spike (no solo binario)
    d["spike_magnitude"]  = ret_5d.where(ret_5d >= spike_thr_high, 0.0).fillna(0.0)
    # Drop simétrico (drops también predicen fade negativo)
    drop_thr_low = float(ret_5d.quantile(0.05))
    d["is_drop_5d"] = (ret_5d <= drop_thr_low).astype(float)
    d["extreme_move_5d"] = ((ret_5d >= spike_thr_high) | (ret_5d <= drop_thr_low)).astype(float)

    # ── Spreads / ratios entre commodities ──────────────────────
    # El ratio soja/maíz es un predictor conocido (arbitraje entre cultivos)
    if "Maize" in d.columns:
        d["soy_corn_ratio"] = (
            d[target_col] / d["Maize"].replace(0, np.nan)
        ).ffill().fillna(2.5)
        # Desviación del ratio respecto a su media móvil (reversión a la media)
        ratio_ma = d["soy_corn_ratio"].rolling(30).mean()
        d["soy_corn_ratio_dev"] = (d["soy_corn_ratio"] - ratio_ma) / ratio_ma.replace(0, 1)

    if "Oil" in d.columns:
        d["soy_oil_ratio"] = (
            d[target_col] / d["Oil"].replace(0, np.nan)
        ).ffill().fillna(30.0)

    # ── Crush spread (margen de procesamiento de soja) ───────────
    # Economía del crushing: 1 bushel soja → ~48 lbs harina + ~11 lbs aceite
    # SoybeanMeal en USD/ton → convertir a cents/bushel: /2204.6 * 100 * 48
    # SoybeanOil en cents/lb → convertir a cents/bushel: * 11
    # crush_spread > 0: procesar soja es rentable → demanda → precio sube
    if "SoybeanMeal" in d.columns and "SoybeanOil" in d.columns:
        meal_cb = (d["SoybeanMeal"] / 2204.6) * 100 * 48
        oil_cb  = d["SoybeanOil"] * 11
        d["crush_spread"]   = (meal_cb + oil_cb - d[target_col]).ffill().fillna(0)
        d["crush_spread_ma"] = d["crush_spread"].rolling(20).mean().fillna(0)
        # Desviación del spread respecto a su media (reversión a la media)
        d["crush_spread_dev"] = (
            d["crush_spread"] - d["crush_spread_ma"]
        ) / d["crush_spread_ma"].abs().replace(0, 1)

    # ── Régimen de mercado (bull vs bear) ───────────────────────
    # price_vs_ma90: ¿el precio está por encima o debajo de su media de 90d?
    # Valor positivo → mercado alcista, negativo → bajista.
    # ma90_slope: pendiente de la MA90 en los últimos 5 días (tendencia).
    # Estas features dan al modelo contexto sobre el régimen actual,
    # evitando que genere señales SELL en tendencias claramente alcistas.
    ma90 = d[target_col].rolling(90, min_periods=30).mean()
    d["price_vs_ma90"]  = ((d[target_col] - ma90) / ma90.replace(0, 1)).fillna(0)
    d["ma90_slope"]     = (ma90.diff(5) / ma90.replace(0, 1)).fillna(0)

    # ma20 para crossover rápido (señal de corto plazo)
    ma20 = d[target_col].rolling(20, min_periods=10).mean()
    d["price_vs_ma20"]  = ((d[target_col] - ma20) / ma20.replace(0, 1)).fillna(0)
    d["ma20_vs_ma90"]   = ((ma20 - ma90) / ma90.replace(0, 1)).fillna(0)  # golden/death cross

    # ── Estacionalidad ───────────────────────────────────────────
    # Soja tiene patrones estacionales fuertes (cosecha mar-may, ago-oct)
    # Encoding circular para que dic y ene estén cerca (no como 12 y 1)
    dates = pd.to_datetime(d[date_col])
    d["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    d["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    d["quarter"]   = dates.dt.quarter.astype(float)

    # ── Retornos forward ─────────────────────────────────────────
    # Antes de calcular retornos, NaN-ear días de rollover en el target
    # para que los artefactos de cambio de contrato no sean aprendidos como señales.
    rollover_dates = _load_rollover_dates()
    if rollover_dates and date_col in d.columns:
        d_dates = pd.to_datetime(d[date_col]).dt.normalize()
        rollover_mask = d_dates.isin(rollover_dates)
        if rollover_mask.any():
            n_masked = rollover_mask.sum()
            print(f"[Rollover] build_features: {n_masked} retorno(s) soja -> NaN por rollover")
            # Crear una copia del precio con NaN en días de rollover para los retornos
            price_clean = d[target_col].copy()
            price_clean.loc[rollover_mask] = np.nan
            d["ret_1d_fwd"]  = price_clean.pct_change(1,  fill_method=None).shift(-1)
            d["ret_7d_fwd"]  = price_clean.pct_change(7,  fill_method=None).shift(-7)
            d["ret_14d_fwd"] = price_clean.pct_change(14, fill_method=None).shift(-14)
            d["ret_30d_fwd"] = price_clean.pct_change(30, fill_method=None).shift(-30)
        else:
            d["ret_1d_fwd"]  = safe_pct_change(d[target_col], 1).shift(-1)
            d["ret_7d_fwd"]  = safe_pct_change(d[target_col], 7).shift(-7)
            d["ret_14d_fwd"] = safe_pct_change(d[target_col], 14).shift(-14)
            d["ret_30d_fwd"] = safe_pct_change(d[target_col], 30).shift(-30)
    else:
        d["ret_1d_fwd"]  = safe_pct_change(d[target_col], 1).shift(-1)
        d["ret_7d_fwd"]  = safe_pct_change(d[target_col], 7).shift(-7)
        d["ret_14d_fwd"] = safe_pct_change(d[target_col], 14).shift(-14)
        d["ret_30d_fwd"] = safe_pct_change(d[target_col], 30).shift(-30)

    # ── Signal Decomposition (CEEMDAN/STL/HP features) ──────────
    try:
        from src.features.signal_decomposition import add_decomposition_features, add_multi_scale_features
        d = add_decomposition_features(d, target_col=target_col, method="stl")
        d = add_multi_scale_features(d, target_col=target_col)
    except Exception as _e:
        print(f"[WARN] signal_decomposition: {_e}")

    # ── Basis features (seasonal, momentum, FX) ──────────────────
    try:
        from src.data.basis_forecast import add_basis_features
        d = add_basis_features(d)
    except Exception as _e:
        print(f"[WARN] basis_features: {_e}")

    # ── Limpieza ─────────────────────────────────────────────────
    d = d.dropna(subset=[target_col])
    d = d.ffill().fillna(0)
    d = d.replace([np.inf, -np.inf], 0)

    return d
