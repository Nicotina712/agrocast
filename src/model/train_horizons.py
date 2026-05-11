"""
src/model/train_horizons.py
Entrena modelos paralelos por horizonte para el forecast del Productor.

Diseño (validado en scripts/forecast_*_eval*.py):
  - Target = retorno forward directo (ret_Hd_fwd), no precio en log
  - Walk-forward retrain cada REFIT_DAYS días (en producción se llama una vez
    por corrida del pipeline → entrena con los últimos TRAIN_YEARS años)
  - XGBoost regresor + dos quantile regressors (Q10, Q90)
  - α óptimo de blending con Random Walk calibrado sobre los últimos VAL_DAYS
  - δ conformal calibrada sobre los mismos VAL_DAYS para alcanzar 80% cobertura
  - Horizontes activos: 7d y 30d (14d descartado por lift negativo en backtest)

Uso:
    python -m src.model.train_horizons
"""
from __future__ import annotations

import os, json, time
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

# ── Constantes (derivadas del backtest) ───────────────────────────
HORIZONS       = [7, 30]                   # 14d descartado
TRAIN_YEARS    = int(os.getenv("HORIZONS_TRAIN_YEARS", "5"))
VAL_DAYS       = int(os.getenv("HORIZONS_VAL_DAYS",   "180"))   # ventana de calibración (α, δ)
MAX_FEATURES   = 25
TARGET_COVERAGE = 0.80                     # nivel nominal de las bandas
N_SEEDS        = int(os.getenv("HORIZONS_N_SEEDS",    "5"))     # multi-seed ensemble
SEEDS          = list(range(42, 42 + N_SEEDS))

EXCLUDE = {
    "Date", "Soybeans", "Soybeans_log", "target_log",
    "ret_1d_fwd", "ret_7d_fwd", "ret_14d_fwd", "ret_30d_fwd",
    "Soybeans_High", "Soybeans_Low", "Soybeans_Open",
    "Maize_High", "Maize_Low", "Maize_Open",
    "SoybeanMeal_High", "SoybeanMeal_Low", "SoybeanMeal_Open",
    "SoybeanOil_High", "SoybeanOil_Low", "SoybeanOil_Open",
    "released_at", "as_of_date", "wasde_date", "week_ending",
}


USE_GRANGER_PRUNING = os.getenv("USE_GRANGER_PRUNING", "1") == "1"
GRANGER_P_THRESHOLD = float(os.getenv("GRANGER_P_THRESHOLD", "0.10"))
GRANGER_MAX_LAG     = int(os.getenv("GRANGER_MAX_LAG", "5"))


def _granger_filter(df: pd.DataFrame, target_col: str, candidates: list[str]) -> list[str]:
    """Filtra `candidates` quedándose con las que tienen Granger causality
    p < GRANGER_P_THRESHOLD sobre el target. Resultado validado en
    scripts/experiments_h1_h2_h6.py: pruning baja MAE −5 % vs full set y reduce
    73 % la dimensionalidad."""
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return candidates

    import warnings
    warnings.filterwarnings("ignore")

    target = df[target_col].dropna()
    if len(target) < 200:
        return candidates

    causal = []
    for f in candidates:
        if f not in df.columns:
            continue
        s = df[[f]].iloc[target.index].copy()
        s["target"] = target.values
        s = s.dropna()
        if len(s) < 100 or s[f].std() < 1e-8:
            continue
        try:
            res = grangercausalitytests(s[["target", f]],
                                         maxlag=GRANGER_MAX_LAG, verbose=False)
            p_min = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, GRANGER_MAX_LAG + 1))
            if p_min < GRANGER_P_THRESHOLD:
                causal.append((f, p_min))
        except Exception:
            continue
    causal.sort(key=lambda x: x[1])
    return [f for f, _ in causal]


def _select_features(df: pd.DataFrame, target_col: str, max_n: int = MAX_FEATURES) -> list[str]:
    """Selecciona features para el modelo. Por default (USE_GRANGER_PRUNING=1):
       1. Filtra con Granger causality (p < 0.10, lags 1-5).
       2. Ordena el subset causal por correlación absoluta con el target.
       3. Toma top-N.

    Si Granger falla o devuelve <5 features, cae a selección clásica por
    correlación.

    Validación: backtest en scripts/experiments_h1_h2_h6.py mostró que el
    pruning Granger reduce MAE 5 % y mantiene estabilidad entre runs."""
    num = df.select_dtypes(include=[np.number]).copy()
    num[target_col] = df[target_col]
    corr = num.corr()[target_col].abs().sort_values(ascending=False)
    candidates = [c for c in corr.index if c not in EXCLUDE and c != target_col]

    if USE_GRANGER_PRUNING:
        try:
            causal = _granger_filter(df, target_col, candidates)
            if len(causal) >= 5:
                # Re-ordenar por correlación absoluta para mantener priorización por relevancia
                causal_corr = corr.reindex(causal).abs().sort_values(ascending=False)
                return list(causal_corr.index)[:max_n]
        except Exception as _e:
            pass  # fallback a clásico

    return candidates[:max_n]


def _fit_xgb(X: pd.DataFrame, y: pd.Series, seed: int = 42, **kw) -> XGBRegressor:
    m = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        eval_metric="mae", **kw,
    )
    m.fit(X, y, verbose=False)
    return m


def _fit_xgb_quantile(X: pd.DataFrame, y: pd.Series, alpha: float, seed: int = 42) -> XGBRegressor:
    m = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        objective="reg:quantileerror", quantile_alpha=alpha,
    )
    m.fit(X, y, verbose=False)
    return m


class SeedEnsemble:
    """Promedia predicciones de N modelos entrenados con seeds distintos.
    Reduce ~25-35% la varianza de la predicción central frente a single-seed,
    sin coste de datos. Compatible con joblib.dump."""
    def __init__(self, models: list):
        self.models = list(models)

    def predict(self, X) -> np.ndarray:
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return preds.mean(axis=0)

    def predict_std(self, X) -> np.ndarray:
        """Desviación entre seeds — proxy de incertidumbre epistémica."""
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return preds.std(axis=0)


def _conformal_delta(y_val: np.ndarray, q10_val: np.ndarray, q90_val: np.ndarray,
                     target: float = TARGET_COVERAGE) -> float:
    """δ tal que [Q10-δ, Q90+δ] cubra `target` fracción de las observaciones de val."""
    score = np.maximum(np.maximum(y_val - q90_val, q10_val - y_val), 0)
    return float(np.quantile(score, target)) if len(score) else 0.0


def _best_alpha(preds_val: np.ndarray, actuals_val: np.ndarray, pt_val: np.ndarray,
                grid: np.ndarray | None = None) -> float:
    """α que minimiza MAE de blend = α·model + (1-α)·RW sobre val."""
    if grid is None:
        grid = np.linspace(0, 1, 21)
    best_a, best_mae = 0.0, float("inf")
    for a in grid:
        blend = a * preds_val + (1 - a) * pt_val
        mae   = float(np.mean(np.abs(blend - actuals_val)))
        if mae < best_mae:
            best_a, best_mae = float(a), mae
    return best_a


def train_horizon(df: pd.DataFrame, horizon: int, target_col: str) -> dict:
    """Entrena un horizonte completo: modelo central, Q10, Q90, α, δ.
    Retorna dict listo para joblib.dump.
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()
    last_date   = df["Date"].max()
    train_start = last_date - pd.DateOffset(years=TRAIN_YEARS)
    val_start   = last_date - pd.Timedelta(days=VAL_DAYS)

    train = df[(df["Date"] >= train_start) & (df["Date"] < val_start)].dropna(subset=[target_col])
    val   = df[(df["Date"] >= val_start)   & (df["Date"] <= last_date)].dropna(subset=[target_col])

    feats = _select_features(train, target_col)
    Xtr = train[feats].fillna(0).replace([np.inf, -np.inf], 0)
    ytr = train[target_col]
    # Multi-seed ensemble: N modelos entrenados con seeds distintos.
    # El predictor central queda robusto a variaciones del split bootstrap.
    model = SeedEnsemble([_fit_xgb(Xtr, ytr, seed=s) for s in SEEDS])
    # Quantiles también con multi-seed → más estabilidad en las bandas
    q10   = SeedEnsemble([_fit_xgb_quantile(Xtr, ytr, 0.10, seed=s) for s in SEEDS])
    q90   = SeedEnsemble([_fit_xgb_quantile(Xtr, ytr, 0.90, seed=s) for s in SEEDS])

    # Calibración α y δ sobre val
    Xv = val[feats].fillna(0).replace([np.inf, -np.inf], 0)
    yv = val[target_col].values
    rp_v  = model.predict(Xv)
    r10_v = q10.predict(Xv)
    r90_v = q90.predict(Xv)

    pt_val      = val["Soybeans"].values
    actuals_val = pt_val * (1 + yv)
    preds_val   = pt_val * (1 + rp_v)
    q10_val     = pt_val * (1 + np.minimum(r10_v, rp_v))
    q90_val     = pt_val * (1 + np.maximum(r90_v, rp_v))

    alpha = _best_alpha(preds_val, actuals_val, pt_val)
    delta = _conformal_delta(actuals_val, q10_val, q90_val, TARGET_COVERAGE)

    # Métricas de validación (informativas, no se usan para decidir)
    rw_mae  = float(np.mean(np.abs(pt_val - actuals_val)))
    ens_val = alpha * preds_val + (1 - alpha) * pt_val
    ens_mae = float(np.mean(np.abs(ens_val - actuals_val)))
    cov     = float(np.mean(((q10_val - delta) <= actuals_val) &
                            (actuals_val <= (q90_val + delta)))) * 100

    return {
        "horizon":       horizon,
        "target":        target_col,
        "model":         model,
        "q10":           q10,
        "q90":           q90,
        "features":      feats,
        "alpha":         alpha,
        "delta":         delta,
        "train_years":   TRAIN_YEARS,
        "val_days":      VAL_DAYS,
        "n_train":       int(len(train)),
        "n_val":         int(len(val)),
        "val_metrics": {
            "mae_rw":      rw_mae,
            "mae_ens":     ens_mae,
            "lift_pct":    (rw_mae - ens_mae) / rw_mae * 100 if rw_mae else 0,
            "coverage_pct": cov,
        },
        "trained_at":    pd.Timestamp.now().isoformat(timespec="seconds"),
    }


def train_all(df: pd.DataFrame, artifacts_dir: str) -> dict:
    """Entrena todos los horizontes activos y guarda en `artifacts_dir`."""
    os.makedirs(artifacts_dir, exist_ok=True)
    summary = {"horizons": {}, "trained_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    for h in HORIZONS:
        target = f"ret_{h}d_fwd"
        if target not in df.columns:
            print(f"   [SKIP] {target} no está en features.csv")
            continue
        t0 = time.time()
        bundle = train_horizon(df, h, target)
        out = os.path.join(artifacts_dir, f"model_h{h}d.joblib")
        joblib.dump(bundle, out)
        elapsed = time.time() - t0
        m = bundle["val_metrics"]
        print(f"   ✅ h={h}d  α={bundle['alpha']:.2f}  δ=${bundle['delta']:.1f}  "
              f"MAE_ens=${m['mae_ens']:.2f}  lift={m['lift_pct']:+.1f}%  cov={m['coverage_pct']:.0f}%  "
              f"({elapsed:.1f}s)")
        summary["horizons"][f"{h}d"] = {
            "alpha":      bundle["alpha"],
            "delta":      bundle["delta"],
            "n_train":    bundle["n_train"],
            "n_val":      bundle["n_val"],
            "val_metrics": m,
            "path":       out,
        }
    with open(os.path.join(artifacts_dir, "horizons_meta.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    feats_path   = os.path.join(project_root, "data", "features.csv")
    artifacts    = os.path.join(project_root, "artifacts", "horizons")
    df = pd.read_csv(feats_path, parse_dates=["Date"])
    print(f"\n🤖 Entrenando modelos por horizonte (TY={TRAIN_YEARS}a, val={VAL_DAYS}d)\n")
    train_all(df, artifacts)
    print(f"\n💾 Artefactos en {artifacts}")


if __name__ == "__main__":
    main()
