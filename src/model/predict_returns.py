"""
src/model/predict_returns.py
Genera señales BUY / SELL / HOLD usando el clasificador de dirección.

El modelo devuelve P(precio sube en 7d):
  P > buy_thresh  → BUY
  P < sell_thresh → SELL
  intermedio      → HOLD

Confianza: distancia de P respecto a 0.5, escalada a [0, 1].
  P=0.58 → conf≈0.16, P=0.75 → conf≈0.50, P=1.0 → conf≈1.0

Compatibilidad: si el modelo guardado es un regresor legacy,
  se usa la lógica de percentiles anterior (backward compat).
"""

import os

import joblib
import numpy as np
import pandas as pd

NON_FEATURE_COLS = {"Date", "Soybeans",
                    "ret_1d_fwd", "ret_7d_fwd", "ret_14d_fwd", "ret_30d_fwd",
                    "ret_7d_net", "ret_14d_net", "direction"}

_DEFAULT_ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "artifacts",
)


def generate_signal(ret: float, low_threshold: float, high_threshold: float) -> str:
    """Mantiene compatibilidad con backtest.py legacy."""
    if ret > high_threshold:
        return "BUY"
    elif ret < low_threshold:
        return "SELL"
    return "HOLD"


def predict_returns(features: pd.DataFrame, artifacts_dir: str | None = None) -> pd.DataFrame:
    if artifacts_dir is None:
        artifacts_dir = _DEFAULT_ARTIFACTS

    model_path = os.path.join(artifacts_dir, "returns_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    saved        = joblib.load(model_path)
    model        = saved["model"]
    vol_model    = saved.get("vol_model")
    feature_cols = saved.get("feature_cols")
    model_type   = saved.get("model_type", "regressor")
    buy_thresh   = saved.get("buy_thresh",  0.58)
    sell_thresh  = saved.get("sell_thresh", 0.42)

    df = features.copy()

    # ── Seleccionar features ───────────────────────────────────────
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"⚠️  Columnas faltantes en predict: {missing[:5]}{'…' if len(missing)>5 else ''} → relleno 0")
        X = df.reindex(columns=feature_cols, fill_value=0)
    else:
        X = df[[c for c in df.columns if c not in NON_FEATURE_COLS]]

    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    # ── Predicción ────────────────────────────────────────────────
    if model_type == "classifier":
        # P(precio sube) en [0, 1]
        probs = model.predict_proba(X)[:, 1].astype(float)
        df["expected_return"] = probs - 0.5   # centrado en 0 para display

        df["signal"] = "HOLD"
        df.loc[probs > buy_thresh,  "signal"] = "BUY"
        df.loc[probs < sell_thresh, "signal"] = "SELL"

        # Confianza: qué tan lejos está P de 0.5 (zona de indecisión)
        # 0.5 → conf=0, 1.0 → conf=1.0, 0.0 → conf=1.0
        df["confidence"] = (np.abs(probs - 0.5) * 2).clip(0, 1)

        dist = df["signal"].value_counts().to_dict()
        print(f"📊 Clasificador P(↑): mean={probs.mean():.3f} std={probs.std():.3f}")
        print(f"   Señales → BUY:{dist.get('BUY',0)} SELL:{dist.get('SELL',0)} HOLD:{dist.get('HOLD',0)}")

    else:
        # ── Legado: regresor con ranking forzado ──────────────────
        preds       = model.predict(X).astype(float)
        df["expected_return"] = preds
        low_thresh  = float(pd.Series(preds).quantile(0.33))
        high_thresh = float(pd.Series(preds).quantile(0.67))

        if abs(high_thresh - low_thresh) < 1e-6:
            print("⚠️  Predicciones degeneradas → distribución forzada por ranking.")
            n     = len(df)
            ranks = pd.Series(preds).rank(method="first")
            df["signal"] = "HOLD"
            df.loc[ranks <= n // 3,      "signal"] = "SELL"
            df.loc[ranks > n - (n // 3), "signal"] = "BUY"
        else:
            df["signal"] = df["expected_return"].apply(
                lambda r: generate_signal(r, low_thresh, high_thresh)
            )

        pred_range = pd.Series(preds).max() - pd.Series(preds).min()
        df["confidence"] = (
            df["expected_return"].apply(
                lambda r: min(abs(r - high_thresh) / (pred_range / 2),
                              abs(r - low_thresh)  / (pred_range / 2), 1.0)
                if r > high_thresh or r < low_thresh else 0.0
            ) if pred_range > 1e-6 else pd.Series(0.5, index=df.index)
        )

    df["confidence"] = df["confidence"].clip(0, 1)

    # ── Head de volatilidad esperada (si fue entrenada) ───────────
    if vol_model is not None:
        try:
            df["expected_vol"] = vol_model.predict(X).astype(float).clip(0, 2)
        except Exception as _e:
            print(f"   [WARN] vol_model.predict falló: {_e}")
            df["expected_vol"] = 0.0
    else:
        df["expected_vol"] = 0.0

    return df[["Date", "expected_return", "signal", "confidence", "expected_vol"]]
