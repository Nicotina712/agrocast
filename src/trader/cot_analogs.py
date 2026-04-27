"""
src/trader/cot_analogs.py
COT Historical Analog Finder.

Cuando el posicionamiento de managed money llega a extremos (top/bottom 10%),
busca las 5 situaciones históricas más similares en los datos COT desde 2006
y reporta qué pasó con el precio en los siguientes 7, 30 y 90 días.

Esto convierte un dato crudo (COT percentil) en inteligencia accionable:
"Las últimas 5 veces que el posicionamiento llegó a este nivel, el precio
cayó en promedio 6.2% en los siguientes 30 días."
"""

import os
from datetime import date

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_COT_PATH     = os.path.join(_PROJECT_ROOT, "data", "cot_soybeans.csv")
_PRICES_PATH  = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")

# Extremo: top/bottom 10% del rango de 52 semanas
EXTREME_THRESHOLD = 10.0   # percentil


def _load_cot_with_prices() -> pd.DataFrame:
    """Combina COT semanal con precio de soja para analizar outcomes."""
    if not os.path.exists(_COT_PATH) or not os.path.exists(_PRICES_PATH):
        return pd.DataFrame()

    cot    = pd.read_csv(_COT_PATH, parse_dates=["Date"])
    prices = pd.read_csv(_PRICES_PATH, parse_dates=["Date"])

    cot    = cot.sort_values("Date").reset_index(drop=True)
    prices = prices.sort_values("Date").reset_index(drop=True)

    # Merge semanal: precio más cercano al reporte COT
    merged = pd.merge_asof(cot, prices[["Date", "Soybeans"]], on="Date",
                           direction="nearest", tolerance=pd.Timedelta("7D"))

    merged = merged.dropna(subset=["Soybeans", "cot_noncomm_net"])
    return merged


def _compute_cot_index(df: pd.DataFrame, window: int = 52) -> pd.Series:
    """Calcula el COT Index (percentil 52 semanas) si no está en el CSV."""
    if "cot_index" in df.columns and df["cot_index"].notna().sum() > 10:
        return df["cot_index"]
    roll_min = df["cot_noncomm_net"].rolling(window, min_periods=20).min()
    roll_max = df["cot_noncomm_net"].rolling(window, min_periods=20).max()
    rng = roll_max - roll_min
    return ((df["cot_noncomm_net"] - roll_min) / rng.replace(0, 1) * 100).clip(0, 100)


def _find_analogs(df: pd.DataFrame, current_idx: float,
                  current_delta: float, n: int = 5) -> pd.DataFrame:
    """
    Encuentra las N filas históricas más similares al posicionamiento actual.
    Similitud basada en: COT index (70%) + delta semanal (30%).
    """
    if df.empty or len(df) < 20:
        return pd.DataFrame()

    df = df.copy()
    df["cot_index_calc"] = _compute_cot_index(df)

    # Normalizar para distancia
    idx_std   = df["cot_index_calc"].std() or 1
    delta_std = df["cot_noncomm_net"].diff().std() or 1

    delta_col = df["cot_noncomm_net"].diff()

    df["distance"] = (
        0.70 * ((df["cot_index_calc"] - current_idx)  / idx_std).abs() +
        0.30 * ((delta_col           - current_delta) / delta_std).abs()
    )

    # Excluir las últimas 52 semanas (evitar self-match)
    cutoff = df["Date"].max() - pd.Timedelta(weeks=52)
    hist   = df[df["Date"] < cutoff].copy()

    if hist.empty:
        return pd.DataFrame()

    return hist.nsmallest(n, "distance").reset_index(drop=True)


def _compute_forward_returns(df: pd.DataFrame, analogs: pd.DataFrame,
                             horizons: list = [7, 30, 90]) -> dict:
    """Calcula retornos futuros para cada analog y agrega estadísticas."""
    results = {}
    for h in horizons:
        returns = []
        for _, row in analogs.iterrows():
            future_date = row["Date"] + pd.Timedelta(days=h)
            future_row  = df[df["Date"] >= future_date]
            if future_row.empty:
                continue
            future_price = future_row["Soybeans"].iloc[0]
            entry_price  = row["Soybeans"]
            if entry_price and entry_price > 0:
                ret = (future_price - entry_price) / entry_price * 100
                returns.append(round(float(ret), 2))

        if returns:
            results[f"ret_{h}d"] = {
                "mean":    round(float(np.mean(returns)), 2),
                "median":  round(float(np.median(returns)), 2),
                "std":     round(float(np.std(returns)), 2),
                "min":     round(min(returns), 2),
                "max":     round(max(returns), 2),
                "n":       len(returns),
                "pct_positive": round(sum(r > 0 for r in returns) / len(returns) * 100, 0),
                "raw":     returns,
            }
        else:
            results[f"ret_{h}d"] = None

    return results


def get_cot_analogs() -> dict:
    """
    Busca situaciones históricas similares al posicionamiento COT actual.

    Retorna dict con:
      is_extreme     : bool — si el posicionamiento está en extremo
      current_index  : COT index actual (0-100)
      current_net    : posición neta actual (contratos)
      extreme_type   : LARGO_EXTREMO / CORTO_EXTREMO / NEUTRO
      analogs        : lista de N situaciones similares históricas
      outcomes       : retornos medios/medianos a 7d, 30d, 90d
      interpretation : texto
      n_analogs_found: int
    """
    df = _load_cot_with_prices()

    if df.empty:
        return {
            "is_extreme": False, "current_index": None,
            "current_net": None, "extreme_type": "NEUTRO",
            "analogs": [], "outcomes": {}, "interpretation": "Sin datos COT disponibles.",
            "n_analogs_found": 0,
        }

    df["cot_index_calc"] = _compute_cot_index(df)

    last = df.iloc[-1]
    current_idx   = float(last.get("cot_index", last["cot_index_calc"]))
    current_net   = float(last.get("cot_noncomm_net", 0))
    current_delta = float(df["cot_noncomm_net"].diff().iloc[-1]) if len(df) > 1 else 0.0

    is_extreme = current_idx <= EXTREME_THRESHOLD or current_idx >= (100 - EXTREME_THRESHOLD)
    if current_idx >= 90:
        extreme_type = "LARGO_EXTREMO"
    elif current_idx <= 10:
        extreme_type = "CORTO_EXTREMO"
    else:
        extreme_type = "NEUTRO"

    analogs  = _find_analogs(df, current_idx, current_delta, n=5)
    outcomes = _compute_forward_returns(df, analogs) if not analogs.empty else {}

    # Armar lista de analogs para el frontend
    analog_list = []
    for _, row in analogs.iterrows():
        analog_list.append({
            "date":        str(row["Date"])[:10],
            "cot_index":   round(float(row["cot_index_calc"]), 1),
            "net":         int(row["cot_noncomm_net"]),
            "price":       round(float(row["Soybeans"]), 2),
            "distance":    round(float(row.get("distance", 0)), 3),
        })

    # Interpretación
    ret30 = outcomes.get("ret_30d")
    if ret30 and ret30.get("n", 0) >= 3:
        mean30 = ret30["mean"]
        pct_pos = ret30["pct_positive"]
        if extreme_type == "LARGO_EXTREMO":
            interp = (
                f"Posicionamiento especulativo en máximos históricos (percentil {current_idx:.0f}). "
                f"En {ret30['n']} situaciones similares, el precio cayó/subió {mean30:+.1f}% a 30 días "
                f"({pct_pos:.0f}% de casos positivos). "
                f"{'Señal de reversión bajista.' if mean30 < -2 else 'Momentum alcista aún dominante.'}"
            )
        elif extreme_type == "CORTO_EXTREMO":
            interp = (
                f"Posicionamiento especulativo en mínimos históricos (percentil {current_idx:.0f}). "
                f"En {ret30['n']} situaciones similares, el precio subió/cayó {mean30:+.1f}% a 30 días "
                f"({pct_pos:.0f}% de casos positivos). "
                f"{'Posible reversión alcista — manos débiles ya salieron.' if mean30 > 2 else 'Cautela — momentum bajista persistente.'}"
            )
        else:
            interp = f"Posicionamiento COT en zona neutral (percentil {current_idx:.0f}). Sin señal extrema."
    elif extreme_type != "NEUTRO":
        interp = f"Posicionamiento en extremo ({extreme_type}, percentil {current_idx:.0f}) — pocos análogos históricos disponibles."
    else:
        interp = f"Posicionamiento COT neutral (percentil {current_idx:.0f}) — sin señal de reversión."

    return {
        "is_extreme":      is_extreme,
        "current_index":   round(current_idx, 1),
        "current_net":     int(current_net),
        "current_delta":   int(current_delta),
        "extreme_type":    extreme_type,
        "analogs":         analog_list,
        "outcomes":        outcomes,
        "interpretation":  interp,
        "n_analogs_found": len(analog_list),
        "as_of":           str(date.today()),
    }
