"""
src/model/predict_horizons.py
Predicción multi-horizonte para AgroCast — reemplazo planificado de predict.py.

Diseño:
  - Carga modelos entrenados por horizonte (artifacts/horizons/model_h{H}d.joblib)
  - Para cada horizonte H disponible, predice retorno → precio_tH = p_t · (1+ret)
  - Aplica blend con Random Walk: precio_final = α·modelo + (1-α)·p_t
  - Bandas: Q10/Q90 modelo + δ conformal → cobertura ~80%
  - Curva diaria: interpolación lineal entre anchors (p_t, p_7d, p_30d)

API principal:
    forecast_curve(df, target="Soybeans", artifacts_dir=...) -> pd.DataFrame
        columnas: Date, Soybeans, upper, lower (mismo formato que el actual)

    forecast_anchors(df, target="Soybeans", artifacts_dir=...) -> dict
        retorna {7: {price, q10, q90}, 30: {price, q10, q90}, ...}
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import joblib

ANCHOR_HORIZONS = [7, 30]   # consistente con train_horizons.HORIZONS


# ─────────────────────────────────────────────────────────────────
# Carga de bundles por horizonte
# ─────────────────────────────────────────────────────────────────
def _load_bundle(artifacts_dir: str, horizon: int) -> dict | None:
    path = os.path.join(artifacts_dir, f"model_h{horizon}d.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────
# Predicción puntual por horizonte
# ─────────────────────────────────────────────────────────────────
def _load_alpha_adjustment(artifacts_root: str | None = None) -> float:
    """Lee artifacts/regime_switching.json y devuelve alpha_adjustment.
    Cuando hay alta probabilidad de high_vol, el adjustment es negativo
    (más peso al RW). Default 0.0 si no hay archivo."""
    if artifacts_root is None:
        project_root  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        artifacts_root = os.path.join(project_root, "artifacts")
    path = os.path.join(artifacts_root, "regime_switching.json")
    if not os.path.exists(path):
        return 0.0
    try:
        import json as _json
        with open(path, "r", encoding="utf-8") as _f:
            ms = _json.load(_f)
        if not ms.get("ok"):
            return 0.0
        return float(ms.get("alpha_adjustment", 0.0))
    except Exception:
        return 0.0


def predict_horizon(df_last_row: pd.DataFrame, bundle: dict, current_price: float,
                    alpha_adjustment: float = 0.0) -> dict:
    """Aplica el bundle a una sola fila de features y devuelve el ancla
    {price, q10, q90, return_pct, model_return_pct, alpha, alpha_effective, delta}.

    alpha_adjustment: ajuste aditivo al alpha calibrado, viene de Markov-Switching
    (H4). Negativo en regímenes de high_vol → reduce peso del modelo, sube peso
    del RW. Clipped a [0, 1].
    """
    feats = bundle["features"]
    X = df_last_row.reindex(columns=feats, fill_value=0).fillna(0).replace([np.inf, -np.inf], 0)

    r_central = float(bundle["model"].predict(X)[0])
    r_q10     = float(bundle["q10"].predict(X)[0])
    r_q90     = float(bundle["q90"].predict(X)[0])

    # Precio puntual del modelo
    model_price = current_price * (1 + r_central)
    # Blend con RW: α calibrado + ajuste por régimen (H4)
    alpha_calibrated = float(bundle["alpha"])
    alpha_effective  = float(np.clip(alpha_calibrated + alpha_adjustment, 0.0, 1.0))
    blended = alpha_effective * model_price + (1 - alpha_effective) * current_price

    # Bandas: convertir retornos a precios y agregar δ conformal
    delta  = float(bundle["delta"])
    q10_price = current_price * (1 + min(r_q10, r_central)) - delta
    q90_price = current_price * (1 + max(r_q90, r_central)) + delta
    # Las bandas siguen al blended center, no al modelo puro:
    # mantenemos el ancho original respecto al model_price.
    band_lower_offset = model_price - q10_price
    band_upper_offset = q90_price   - model_price
    q10_blended = blended - band_lower_offset
    q90_blended = blended + band_upper_offset

    return {
        "price":             round(blended, 2),
        "q10":               round(max(q10_blended, 0), 2),
        "q90":               round(q90_blended, 2),
        "return_pct":        round((blended - current_price) / current_price * 100, 3),
        "model_price":       round(model_price, 2),
        "model_return_pct":  round(r_central * 100, 3),
        "alpha":             alpha_effective,
        "alpha_calibrated":  alpha_calibrated,
        "alpha_adjustment":  round(alpha_adjustment, 3),
        "delta":             round(delta, 2),
    }


# ─────────────────────────────────────────────────────────────────
# Anchors completos {7: {...}, 30: {...}}
# ─────────────────────────────────────────────────────────────────
def forecast_anchors(df: pd.DataFrame, target: str = "Soybeans",
                     artifacts_dir: str | None = None) -> dict:
    """Devuelve los anchors por horizonte para la última fila de `df`."""
    if artifacts_dir is None:
        project_root  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        artifacts_dir = os.path.join(project_root, "artifacts", "horizons")

    last_row = df.iloc[[-1]].copy()
    current_price = float(last_row[target].iloc[0])

    # Ajuste de alpha desde Markov-Switching (H4) — el archivo vive en
    # artifacts/regime_switching.json (un nivel arriba de horizons/).
    artifacts_root = os.path.dirname(artifacts_dir) if artifacts_dir else None
    alpha_adjustment = _load_alpha_adjustment(artifacts_root)

    out = {"current_price": round(current_price, 2),
           "alpha_adjustment_regime": round(alpha_adjustment, 3),
           "horizons": {}}
    for h in ANCHOR_HORIZONS:
        bundle = _load_bundle(artifacts_dir, h)
        if bundle is None:
            continue
        out["horizons"][h] = predict_horizon(last_row, bundle, current_price,
                                              alpha_adjustment=alpha_adjustment)
    return out


# ─────────────────────────────────────────────────────────────────
# Curva diaria por interpolación lineal entre anchors
# ─────────────────────────────────────────────────────────────────
def forecast_curve(df: pd.DataFrame, target: str = "Soybeans", date_col: str = "Date",
                   artifacts_dir: str | None = None, real_last_date=None,
                   steps: int = 30) -> pd.DataFrame:
    """Genera la curva diaria de `steps` días con bandas Q10/Q90.
    Mantiene el contrato de salida del predict.py original:
        DataFrame con columnas Date, Soybeans, upper, lower
    """
    anchors = forecast_anchors(df, target=target, artifacts_dir=artifacts_dir)
    current_price = anchors["current_price"]
    h_keys = sorted(anchors["horizons"].keys())
    if not h_keys:
        raise FileNotFoundError(
            "No se encontraron modelos por horizonte. Ejecutar train_horizons primero.")

    # Anchors para interpolar (incluye día 0 = current)
    xs = [0] + list(h_keys)
    ys_price = [current_price] + [anchors["horizons"][h]["price"] for h in h_keys]
    ys_q10   = [current_price] + [anchors["horizons"][h]["q10"]   for h in h_keys]
    ys_q90   = [current_price] + [anchors["horizons"][h]["q90"]   for h in h_keys]

    # Si el horizonte máximo solicitado supera el último anchor → extrapolar plano
    max_h = h_keys[-1]
    days  = np.arange(1, steps + 1)
    days_clamped = np.minimum(days, max_h)

    price = np.interp(days_clamped, xs, ys_price)
    q10   = np.interp(days_clamped, xs, ys_q10)
    q90   = np.interp(days_clamped, xs, ys_q90)

    # Última fecha
    if real_last_date is not None:
        last_date = pd.to_datetime(real_last_date)
    else:
        last_date = pd.to_datetime(df[date_col]).sort_values().iloc[-1]

    today = pd.Timestamp.today().normalize()
    if last_date < today - pd.Timedelta(days=30):
        last_date = today

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=steps, freq="D")

    return pd.DataFrame({
        "Date":     future_dates,
        "Soybeans": np.round(price, 2),
        "upper":    np.round(q90, 2),
        "lower":    np.round(q10, 2),
    })


def save_forecast_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"💾 Forecast (horizons) guardado en {path}")


# ─────────────────────────────────────────────────────────────────
# Monte Carlo paths para densidad probabilistica
# ─────────────────────────────────────────────────────────────────
def _gaussian_anchor_sample(price_t: float, anchor: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Muestrea n precios al horizonte a partir del ancla.
    Aproximación normal con μ=blended, σ derivada del ancho de banda Q10/Q90.
    Fundamento: si la banda Q10/Q90 se calibra para cubrir 80%, entonces
    Q90-Q10 ≈ 2·1.282·σ → σ = (Q90-Q10)/2.564"""
    mu    = float(anchor["price"])
    width = max(float(anchor["q90"]) - float(anchor["q10"]), 1e-6)
    sigma = width / 2.564
    return rng.normal(mu, sigma, size=n)


def forecast_paths(df: pd.DataFrame, target: str = "Soybeans",
                   artifacts_dir: str | None = None, n_paths: int = 1000,
                   horizon_days: int = 30, seed: int = 42) -> dict:
    """Genera `n_paths` trayectorias diarias por interpolación lineal entre
    anclas muestreadas (p_t, p_7d ~ N, p_30d ~ N) y devuelve métricas
    probabilisticas útiles para decisión del productor.

    Returns dict con:
        paths            : matriz (n_paths × horizon_days) de precios simulados
        terminal_density : tuple (precios, percentiles) para histograma
        prob_above       : función price → P(price_terminal > price)
        useful           : ya pre-calculados Q05, Q25, Q50, Q75, Q95
    """
    anchors = forecast_anchors(df, target=target, artifacts_dir=artifacts_dir)
    p_t = anchors["current_price"]
    h_keys = sorted(anchors["horizons"].keys())
    if not h_keys:
        return {"ok": False, "error": "no horizons available"}

    rng = np.random.default_rng(seed)
    # Muestrear precio en cada ancla
    samples_by_h = {0: np.full(n_paths, p_t)}
    for h in h_keys:
        samples_by_h[h] = _gaussian_anchor_sample(p_t, anchors["horizons"][h], n_paths, rng)
    # Construir paths interpolando linealmente entre anclas para cada draw
    days  = np.arange(1, horizon_days + 1)
    max_h = h_keys[-1]
    days_clamped = np.minimum(days, max_h)
    paths = np.zeros((n_paths, horizon_days))
    xs = [0] + h_keys
    for i in range(n_paths):
        ys = [samples_by_h[xx][i] for xx in xs]
        paths[i, :] = np.interp(days_clamped, xs, ys)

    terminal = paths[:, -1]
    quantiles = {
        "q05": float(np.quantile(terminal, 0.05)),
        "q25": float(np.quantile(terminal, 0.25)),
        "q50": float(np.quantile(terminal, 0.50)),
        "q75": float(np.quantile(terminal, 0.75)),
        "q95": float(np.quantile(terminal, 0.95)),
    }
    return {
        "ok":            True,
        "current_price": p_t,
        "horizon_days":  horizon_days,
        "n_paths":       n_paths,
        "terminal_quantiles": quantiles,
        "terminal_mean":      float(terminal.mean()),
        "terminal_std":       float(terminal.std()),
        "prob_up_pct":        float((terminal > p_t).mean() * 100),
        "anchors":            anchors,
        "paths":              paths,   # (n_paths, horizon_days) matriz cents/bu
    }


def prob_above(df: pd.DataFrame, threshold: float, artifacts_dir: str | None = None,
               n_paths: int = 5000, horizon_days: int = 30) -> float:
    """P(precio_t+horizon > threshold). Útil para "P(precio > mi costo total)"."""
    paths_out = forecast_paths(df, artifacts_dir=artifacts_dir,
                                n_paths=n_paths, horizon_days=horizon_days)
    if not paths_out.get("ok"):
        return float("nan")
    # Re-muestreo rápido del terminal
    rng = np.random.default_rng(42)
    p_t = paths_out["current_price"]
    h_keys = sorted(paths_out["anchors"]["horizons"].keys())
    h = min(horizon_days, h_keys[-1])
    # Encontrar ancla más cercana al horizon solicitado
    nearest = min(h_keys, key=lambda x: abs(x - h))
    sample = _gaussian_anchor_sample(p_t, paths_out["anchors"]["horizons"][nearest], n_paths, rng)
    return float((sample > threshold).mean() * 100)
