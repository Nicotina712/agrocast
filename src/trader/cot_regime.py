"""
src/trader/cot_regime.py
Convierte cot_index continuo en flags de régimen discreto.

Régimen contrarian clásico:
  cot_index >= 90  → extreme_long  (especuladores muy largos → señal BAJISTA)
  cot_index >= 70  → long
  30 <= idx < 70   → neutral
  cot_index < 30   → short
  cot_index < 10   → extreme_short (especuladores muy cortos → señal ALCISTA)

Genera columnas one-hot + un score numérico (-2 a +2) interpretable.
"""

import pandas as pd


def add_cot_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas cot_regime_* y cot_contrarian_score al df."""
    if "cot_index" not in df.columns:
        df["cot_regime_extreme_long"]  = 0
        df["cot_regime_long"]          = 0
        df["cot_regime_neutral"]       = 1
        df["cot_regime_short"]         = 0
        df["cot_regime_extreme_short"] = 0
        df["cot_contrarian_score"]     = 0.0
        return df

    idx = df["cot_index"].fillna(50.0)

    df["cot_regime_extreme_long"]  = (idx >= 90).astype(int)
    df["cot_regime_long"]          = ((idx >= 70) & (idx < 90)).astype(int)
    df["cot_regime_neutral"]       = ((idx >= 30) & (idx < 70)).astype(int)
    df["cot_regime_short"]         = ((idx >= 10) & (idx < 30)).astype(int)
    df["cot_regime_extreme_short"] = (idx < 10).astype(int)

    # Score contrarian: extreme_long → -2 (bajista), extreme_short → +2 (alcista)
    score = pd.Series(0.0, index=df.index)
    score = score.where(~(idx >= 90), -2.0)
    score = score.where(~((idx >= 70) & (idx < 90)), -1.0)
    score = score.where(~((idx >= 10) & (idx < 30)), 1.0)
    score = score.where(~(idx < 10), 2.0)
    df["cot_contrarian_score"] = score

    return df
