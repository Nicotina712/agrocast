"""
src/trader/ml_quality.py
Métricas honestas del clasificador ML 7d.

Compara señales históricas (artifacts/signals.csv) contra retornos reales
(data/raw_market.csv) y calcula:
  - hit_rate          : aciertos direccionales BUY/SELL (excluye HOLD)
  - hit_rate_high_conf: hit rate restringido a confianza >= 0.6
  - roc_auc           : AUC de expected_return vs (ret_7d > 0)
  - n_signals         : total no-HOLD evaluadas
  - quality_label     : "robusto" | "aceptable" | "señal débil"

Cache: data/ml_quality.json (TTL 6h)
"""

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "ml_quality.json")
_TTL_HOURS    = 6


def _cache_valid() -> bool:
    if not os.path.exists(_CACHE_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(_CACHE_PATH))
    return age < timedelta(hours=_TTL_HOURS)


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC sin sklearn: rank-based Mann-Whitney."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order  = np.argsort(np.concatenate([pos, neg]))
    ranks  = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    sum_pos = ranks[:len(pos)].sum()
    return float((sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _label_for(hit: float, auc: float, n: int) -> str:
    if n < 50 or np.isnan(hit):
        return "datos insuficientes"
    if hit >= 58 and auc >= 0.58:
        return "robusto"
    if hit >= 53 and auc >= 0.54:
        return "aceptable"
    return "señal débil"


def compute_ml_quality(force: bool = False) -> dict:
    if not force and _cache_valid():
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    sig_path = os.path.join(_PROJECT_ROOT, "artifacts", "signals.csv")
    raw_path = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")
    if not (os.path.exists(sig_path) and os.path.exists(raw_path)):
        return {"ok": False, "reason": "missing input files"}

    sig = pd.read_csv(sig_path, parse_dates=["Date"]).sort_values("Date")
    raw = pd.read_csv(raw_path, parse_dates=["Date"]).sort_values("Date")[["Date", "Soybeans"]]

    df = sig.merge(raw, on="Date", how="inner")
    df["price_7d"] = df["Soybeans"].shift(-7)
    df["ret_7d"]   = (df["price_7d"] / df["Soybeans"] - 1.0)
    df = df.dropna(subset=["ret_7d"])

    def _metrics_block(sub: pd.DataFrame) -> dict:
        nonhold = sub[sub["signal"] != "HOLD"].copy()
        nonhold["pred_up"]   = (nonhold["signal"] == "BUY").astype(int)
        nonhold["actual_up"] = (nonhold["ret_7d"] > 0).astype(int)
        n  = int(len(nonhold))
        hr = float((nonhold["pred_up"] == nonhold["actual_up"]).mean() * 100) if n else float("nan")
        hc = nonhold[nonhold["confidence"] >= 0.6]
        n_hc  = int(len(hc))
        hr_hc = float((hc["pred_up"] == hc["actual_up"]).mean() * 100) if n_hc else float("nan")
        auc_df = sub.dropna(subset=["expected_return", "ret_7d"])
        auc = _roc_auc(auc_df["expected_return"].values,
                       (auc_df["ret_7d"] > 0).astype(int).values) if len(auc_df) else float("nan")
        mae = float((nonhold["ret_7d"].abs().mean()) * 100) if n else float("nan")
        return {
            "n_signals":          n,
            "n_signals_high_conf": n_hc,
            "hit_rate":           round(hr, 1) if not np.isnan(hr) else None,
            "hit_rate_high_conf": round(hr_hc, 1) if not np.isnan(hr_hc) else None,
            "roc_auc":            round(auc, 3) if not np.isnan(auc) else None,
            "avg_abs_return_7d_pct": round(mae, 2) if not np.isnan(mae) else None,
            "quality_label":      _label_for(hr, auc, n),
            "period_start":       sub["Date"].min().date().isoformat() if len(sub) else None,
            "period_end":         sub["Date"].max().date().isoformat() if len(sub) else None,
        }

    cutoff = df["Date"].max() - pd.Timedelta(days=365)
    full_block   = _metrics_block(df)
    recent_block = _metrics_block(df[df["Date"] >= cutoff])

    result = {
        "ok":         True,
        "as_of":      datetime.now().isoformat(),
        "full":       full_block,
        "recent_12m": recent_block,
        "note":       "full = histórico completo (probable in-sample, inflado). recent_12m = últimos 12 meses (más cercano a out-of-sample).",
    }

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    print(json.dumps(compute_ml_quality(force=True), indent=2, ensure_ascii=False))
