"""
src/model/predict_multihorizon.py
Forecast multi-horizonte para AgroCast PRO.

Horizontes:
  7d  — señal XGB + volatilidad histórica
  30d — modelo Ridge (pipeline principal)
  90d — tendencia MA90 + estacionalidad histórica (OOS correcto)

Corrección de data leakage en 90d:
  La estimación estacional usa únicamente datos hasta HOLDOUT_YEARS antes
  del período de evaluación, garantizando que el modelo nunca ve el futuro.
  Los metrics de accuracy son calculados out-of-sample sobre el 20% final.
"""

import os
import numpy as np
import pandas as pd
import joblib

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ARTIFACTS    = os.path.join(_PROJECT_ROOT, "artifacts")

HOLDOUT_YEARS = 2   # años excluidos del entrenamiento estacional para OOS


def _load_features() -> pd.DataFrame | None:
    path = os.path.join(_PROJECT_ROOT, "data", "features.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        return None


def _load_forecast_30d() -> list:
    path = os.path.join(_ARTIFACTS, "forecast.csv")
    if not os.path.exists(path):
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except Exception:
        return []


def _compute_oos_metrics_7d(features: pd.DataFrame) -> dict:
    """
    Métricas OOS para el horizonte de 7 días.
    Usa val_acc/val_auc almacenados en el artefacto del modelo (20% holdout
    aplicado correctamente durante el entrenamiento).
    """
    try:
        model_path = os.path.join(_PROJECT_ROOT, "src", "model", "artifacts",
                                  "returns_model.joblib")
        if not os.path.exists(model_path):
            return {}
        artifact = joblib.load(model_path)
        clf      = artifact["model"]
        feat_cols = artifact.get("feature_cols", [])

        val_acc = artifact.get("val_acc")
        val_auc = artifact.get("val_auc")

        # Sharpe sobre el 20% holdout
        sharpe = None
        if "ret_7d_fwd" in features.columns and feat_cols:
            df    = features.dropna(subset=["ret_7d_fwd"]).copy()
            split = int(len(df) * 0.80)
            oos   = df.iloc[split:]
            if len(oos) > 10:
                valid_cols = [c for c in feat_cols if c in oos.columns]
                X_oos  = oos[valid_cols].fillna(0)
                proba  = clf.predict_proba(X_oos)[:, 1]
                # Retorno simulado: long cuando proba > buy_thresh, short cuando < sell_thresh
                buy_t  = artifact.get("buy_thresh", 0.58)
                sell_t = artifact.get("sell_thresh", 0.42)
                pos    = np.where(proba > buy_t, 1, np.where(proba < sell_t, -1, 0))
                strat_rets = pos * oos["ret_7d_fwd"].values
                sharpe = float(strat_rets.mean() / (strat_rets.std() + 1e-8) * np.sqrt(52))

        return {
            "accuracy_pct":      round(val_acc * 100, 1) if val_acc else None,
            "auc":               round(val_auc, 3) if val_auc else None,
            "sharpe_annualized": round(sharpe, 2) if sharpe else None,
            "buy_thresh":        artifact.get("buy_thresh"),
            "sell_thresh":       artifact.get("sell_thresh"),
        }
    except Exception:
        return {}


def _compute_oos_metrics_30d(features: pd.DataFrame) -> dict:
    """
    Calcula métricas OOS para el horizonte de 30 días.
    Usa el modelo Ridge ya entrenado + features del 20% holdout.
    """
    try:
        model_path = os.path.join(_ARTIFACTS, "model.joblib")
        if not os.path.exists(model_path):
            return {}
        artifact  = joblib.load(model_path)
        model     = artifact["model"] if isinstance(artifact, dict) else artifact
        feat_cols = artifact.get("features") if isinstance(artifact, dict) else None

        exclude = {"Date", "Soybeans", "ret_1d_fwd", "ret_7d_fwd", "ret_30d_fwd"}
        if feat_cols is None:
            feat_cols = [c for c in features.columns if c not in exclude]

        df    = features.dropna(subset=[c for c in feat_cols if c in features.columns]).copy()
        split = int(len(df) * 0.80)
        oos   = df.iloc[split:].copy()
        if len(oos) < 10:
            return {}

        valid_cols = [c for c in feat_cols if c in oos.columns]
        X_oos    = oos[valid_cols].fillna(0)
        y_actual = oos["Soybeans"].values

        # Model trained on log1p(price) — convert predictions back to price
        y_pred_log = model.predict(X_oos)
        y_pred     = np.expm1(y_pred_log)
        mae    = float(np.mean(np.abs(y_pred - y_actual)))
        mape   = float(np.mean(np.abs((y_pred - y_actual) / (y_actual + 1e-8)))) * 100

        direction_correct = None
        if "ret_30d_fwd" in oos.columns:
            direction_correct = float(
                (np.sign(y_pred - y_actual) == np.sign(oos["ret_30d_fwd"].values)).mean()
            ) * 100

        return {
            "mae_usd_bu":        round(mae, 3),
            "mape_pct":          round(mape, 2),
            "direction_acc_pct": round(direction_correct, 1) if direction_correct else None,
            "n_oos_periods":     len(oos),
        }
    except Exception:
        return {}


def _compute_oos_metrics_90d(features: pd.DataFrame) -> dict:
    """
    Calcula métricas OOS para el modelo estacional de 90 días.
    Reserva el 20% final como holdout, usa solo el 80% para calibrar estacionalidad.
    """
    try:
        df = features.sort_values("Date").copy()
        df["ret_90d"] = df["Soybeans"].pct_change(63)
        df["month"]   = df["Date"].dt.month
        df = df.dropna(subset=["ret_90d"])

        split    = int(len(df) * 0.80)
        train_df = df.iloc[:split]
        oos_df   = df.iloc[split:]

        if len(oos_df) < 10:
            return {}

        # Curva estacional calibrada SOLO en train
        seasonal = train_df.groupby("month")["ret_90d"].mean()

        # Predicción en OOS: retorno estacional del mes correspondiente
        oos_pred = oos_df["month"].map(seasonal).values
        oos_true = oos_df["ret_90d"].values

        valid = ~(np.isnan(oos_pred) | np.isnan(oos_true))
        if valid.sum() < 5:
            return {}

        mae_ret  = float(np.mean(np.abs(oos_pred[valid] - oos_true[valid]))) * 100
        dir_acc  = float(np.mean(np.sign(oos_pred[valid]) == np.sign(oos_true[valid]))) * 100

        return {
            "mae_return_pct":    round(mae_ret, 2),
            "direction_acc_pct": round(dir_acc, 1),
            "n_oos_periods":     int(valid.sum()),
            "train_years":       round((train_df["Date"].max() - train_df["Date"].min()).days / 365, 1),
        }
    except Exception:
        return {}


def _forecast_7d(features: pd.DataFrame) -> dict:
    try:
        signals_path = os.path.join(_ARTIFACTS, "signals.csv")
        mkt_path     = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")

        sig_df = pd.read_csv(signals_path) if os.path.exists(signals_path) else pd.DataFrame()
        mkt_df = pd.read_csv(mkt_path, parse_dates=["Date"]) if os.path.exists(mkt_path) else pd.DataFrame()

        if mkt_df.empty:
            return {}

        mkt_df        = mkt_df.sort_values("Date")
        current_price = float(mkt_df["Soybeans"].iloc[-1])

        expected_return = 0.0
        confidence = 0.5
        signal = "HOLD"
        if not sig_df.empty:
            last_sig        = sig_df.iloc[-1]
            expected_return = float(last_sig.get("expected_return", 0))
            confidence      = float(last_sig.get("confidence", 0.5))
            signal          = str(last_sig.get("signal", "HOLD"))

        vol_7d = mkt_df["Soybeans"].pct_change().rolling(20).std().iloc[-1]
        if np.isnan(vol_7d):
            vol_7d = 0.015

        price_return_est = 2 * expected_return * vol_7d * np.sqrt(7)
        price_7d  = current_price * (1 + price_return_est)
        vol_7d_abs = current_price * vol_7d * np.sqrt(7)
        upper = price_7d + 1.5 * vol_7d_abs
        lower = price_7d - 1.5 * vol_7d_abs
        ret_pct = (price_7d - current_price) / current_price * 100

        oos = _compute_oos_metrics_7d(features) if features is not None else {}

        return {
            "horizon":       "7d",
            "label":         "Corto plazo (7 días)",
            "description":   "Señal XGBoost + volatilidad histórica",
            "current_price": round(current_price, 2),
            "forecast":      round(price_7d, 2),
            "upper":         round(upper, 2),
            "lower":         round(lower, 2),
            "return_pct":    round(ret_pct, 2),
            "signal":        signal,
            "confidence":    round(confidence, 3),
            "target_use":    "Traders de futuros",
            "oos_metrics":   oos,
        }
    except Exception as e:
        print(f"[WARN] forecast_7d: {e}")
        return {}


def _ensemble_30d_adjustment(features: pd.DataFrame, ridge_price: float,
                              current_price: float) -> tuple[float, str, float]:
    """
    Ajuste ensemble: usa la señal XGBoost para sesgar el precio Ridge.

    Lógica:
      - XGBoost predice dirección (BUY/SELL/HOLD) con una probabilidad
      - Si la señal es consistente con el forecast Ridge, aumentamos la
        convicción un 30% (blend 70% Ridge + 30% XGBoost-adjusted)
      - Si es inconsistente, reducimos el forecast hacia current_price (20%)

    Retorna: (precio_ajustado, descripcion_ensemble, confianza_xgb)
    """
    try:
        model_path = os.path.join(_PROJECT_ROOT, "src", "model", "artifacts",
                                  "returns_model.joblib")
        if not os.path.exists(model_path) or features is None:
            return ridge_price, "Ridge solo", 0.5

        artifact  = joblib.load(model_path)
        clf       = artifact["model"]
        feat_cols = artifact.get("feature_cols", [])
        buy_t     = artifact.get("buy_thresh", 0.58)
        sell_t    = artifact.get("sell_thresh", 0.42)

        valid_cols = [c for c in feat_cols if c in features.columns]
        if not valid_cols:
            return ridge_price, "Ridge solo (sin features XGB)", 0.5

        X_last = features[valid_cols].fillna(0).iloc[[-1]]
        proba  = float(clf.predict_proba(X_last)[0, 1])

        ridge_ret = (ridge_price - current_price) / (current_price + 1e-8)

        if proba > buy_t:
            xgb_dir = 1
            xgb_desc = f"XGB BUY ({proba:.2f})"
        elif proba < sell_t:
            xgb_dir = -1
            xgb_desc = f"XGB SELL ({proba:.2f})"
        else:
            # XGB neutral — mantener Ridge sin cambio
            return ridge_price, f"Ridge + XGB HOLD ({proba:.2f})", proba

        ridge_dir = 1 if ridge_ret > 0 else -1

        if xgb_dir == ridge_dir:
            # Señales alineadas: reforzar 30%
            xgb_price = current_price * (1 + ridge_ret * 1.30)
            adj_price = 0.70 * ridge_price + 0.30 * xgb_price
            desc = f"Ensemble Ridge+XGB alineados — {xgb_desc}"
        else:
            # Señales contradictorias: mover 20% hacia precio actual
            adj_price = 0.80 * ridge_price + 0.20 * current_price
            desc = f"Ensemble Ridge+XGB divergentes — {xgb_desc} (ajuste conservador)"

        return round(adj_price, 2), desc, proba
    except Exception as e:
        print(f"[WARN] ensemble_30d: {e}")
        return ridge_price, "Ridge solo (error ensemble)", 0.5


def _forecast_30d_summary(forecast_30d: list, current_price: float,
                           features: pd.DataFrame) -> dict:
    if not forecast_30d or len(forecast_30d) < 20:
        return {}
    try:
        last       = forecast_30d[-1]
        ridge_price = float(last["Soybeans"])
        upper      = float(last.get("upper", ridge_price * 1.05))
        lower      = float(last.get("lower", ridge_price * 0.95))

        # Ensemble: ajuste XGBoost sobre precio Ridge
        adj_price, ensemble_desc, xgb_proba = _ensemble_30d_adjustment(
            features, ridge_price, current_price
        )
        # Escalar bandas proporcionalmente
        band_half = (upper - lower) / 2
        adj_upper = adj_price + band_half
        adj_lower = adj_price - band_half
        ret_pct   = (adj_price - current_price) / current_price * 100

        oos = _compute_oos_metrics_30d(features) if features is not None else {}

        return {
            "horizon":         "30d",
            "label":           "Mediano plazo (30 días)",
            "description":     ensemble_desc,
            "current_price":   round(current_price, 2),
            "forecast":        adj_price,
            "forecast_ridge":  round(ridge_price, 2),
            "upper":           round(adj_upper, 2),
            "lower":           round(adj_lower, 2),
            "return_pct":      round(ret_pct, 2),
            "xgb_proba":       round(xgb_proba, 3),
            "target_use":      "Decisiones de cobertura / hedging",
            "oos_metrics":     oos,
        }
    except Exception as e:
        print(f"[WARN] forecast_30d_summary: {e}")
        return {}


def _forecast_90d(features: pd.DataFrame, current_price: float) -> dict:
    """
    Horizonte largo (90 días): tendencia MA90 + estacionalidad OOS-correcta.

    FIX data leakage: la curva estacional se calibra ÚNICAMENTE con datos
    anteriores al período de holdout (últimos HOLDOUT_YEARS años).
    """
    try:
        if features is None or features.empty:
            return {}

        feats = features.sort_values("Date").copy()

        # ── Tendencia: pendiente de MA90 ────────────────────────────────
        ma90_slope = float(feats["ma90_slope"].iloc[-1]) \
            if "ma90_slope" in feats.columns else 0.0

        # ── Estacionalidad: calibrada solo con datos de entrenamiento ───
        today      = pd.Timestamp.today()
        cutoff     = today - pd.DateOffset(years=HOLDOUT_YEARS)

        train_df   = feats[feats["Date"] < cutoff].copy()
        train_df["ret_3m"] = train_df["Soybeans"].pct_change(63)
        train_df["month"]  = train_df["Date"].dt.month

        target_month = ((today.month + 2) % 12) + 1  # ~3 meses adelante
        month_rets   = train_df.groupby("month")["ret_3m"].mean()
        seasonal_ret = float(month_rets.get(target_month, 0.0))
        if np.isnan(seasonal_ret):
            seasonal_ret = 0.0

        # ── Combinar tendencia + estacionalidad ─────────────────────────
        trend_ret    = ma90_slope * 90 / (current_price + 1e-8)
        combined_ret = np.clip(0.60 * trend_ret + 0.40 * seasonal_ret, -0.20, 0.20)

        price_90d = current_price * (1 + combined_ret)
        upper     = price_90d * 1.10
        lower     = price_90d * 0.90
        ret_pct   = combined_ret * 100

        oos = _compute_oos_metrics_90d(feats)

        return {
            "horizon":           "90d",
            "label":             "Largo plazo (90 días)",
            "description":       f"Tendencia MA90 + estacionalidad histórica (calibrada en {train_df['Date'].max().date()} hacia atrás)",
            "current_price":     round(current_price, 2),
            "forecast":          round(price_90d, 2),
            "upper":             round(upper, 2),
            "lower":             round(lower, 2),
            "return_pct":        round(ret_pct, 2),
            "seasonal_month":    target_month,
            "seasonal_ret_pct":  round(seasonal_ret * 100, 2),
            "target_use":        "Productores — contratos forward / decisión de venta",
            "oos_metrics":       oos,
        }
    except Exception as e:
        print(f"[WARN] forecast_90d: {e}")
        return {}


def get_multihorizon_forecast() -> dict:
    """Punto de entrada principal. Retorna forecasts OOS-correctos a 7d, 30d y 90d."""
    features    = _load_features()
    forecast_30 = _load_forecast_30d()

    current_price = 0.0
    try:
        mkt = pd.read_csv(os.path.join(_PROJECT_ROOT, "data", "raw_market.csv"),
                          parse_dates=["Date"])
        current_price = float(mkt.sort_values("Date")["Soybeans"].iloc[-1])
    except Exception:
        if features is not None and not features.empty:
            current_price = float(features["Soybeans"].iloc[-1])

    horizons = []
    for h in [
        _forecast_7d(features),
        _forecast_30d_summary(forecast_30, current_price, features),
        _forecast_90d(features, current_price) if features is not None else {},
    ]:
        if h:
            horizons.append(h)

    return {
        "ok":            True,
        "current_price": round(current_price, 2),
        "horizons":      horizons,
        "generated_at":  str(pd.Timestamp.now())[:19],
    }
