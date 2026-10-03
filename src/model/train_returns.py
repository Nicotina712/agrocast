"""
src/model/train_returns.py
Entrena un XGBClassifier para predecir la DIRECCIÓN del precio a 7 días.

Cambio clave vs versión anterior (XGBRegressor):
  - Predecir el valor exacto del retorno es casi imposible (r²≈0 en finanzas).
  - Predecir la DIRECCIÓN (sube/baja) es más factible y directamente útil.
  - El clasificador devuelve una probabilidad P(sube) en [0,1]:
      P > 0.58 → BUY, P < 0.42 → SELL, 0.42–0.58 → HOLD
  - Esto elimina el problema de predicciones degeneradas (todo≈0) de raíz.
"""

import os

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

NON_FEATURE_COLS = {"Date", "Soybeans",
                     "ret_1d_fwd", "ret_7d_fwd", "ret_14d_fwd", "ret_30d_fwd",
                     "ret_7d_net", "ret_14d_net", "direction",
                     "realized_vol_14d",
                     # OHLC del mismo día = look-ahead implícito
                     "Soybeans_High", "Soybeans_Low", "Soybeans_Open",
                     "Maize_High", "Maize_Low", "Maize_Open",
                     "SoybeanMeal_High", "SoybeanMeal_Low", "SoybeanMeal_Open",
                     "SoybeanOil_High",  "SoybeanOil_Low",  "SoybeanOil_Open",
                     # auxiliares point-in-time
                     "released_at", "as_of_date", "wasde_date", "week_ending"}
TARGET_REG = "ret_14d_fwd"  # horizonte 14d (más señal, menos ruido que 7d)
VOL_TARGET = "realized_vol_14d"  # head adicional: vol esperada (sizing/conviction)
EMBARGO_DAYS = 18                # = horizonte (14) + buffer evento (4)
MAX_RETURN_FEATURES = 40         # Tope de features tras selección por consistencia
MIN_REAL_SHARE = 0.5             # % mínimo de filas con dato real (≠0, ≠NaN) en el train

# ── Auditoría 2026-10-03 (scripts/research_model_audit.py / _variants.py) ──
# Walk-forward trimestral 2019-2026, rolling 5y, embargo 18d, AUC por período
# (19-20 / 21-22 / 23-24H1 / 24H2-25 / 25Q4-26):
#   199 variables → top-40 (anterior) : .45 .53 .48 .59 .61  (media .53)
#   cobertura ≥50% → top-40 (ACTUAL)  : .41-.44 .56-.58 .44-.46 .57-.60 .62 (media .52-.53)
#   solo cross-asset+insp+wasde (34)  : .54 .56 .59 .53 .43  (media .53) → NO adoptado:
#       gana 3 de 5 períodos pero colapsa en el último año; mismo promedio = ruido.
# Conclusiones robustas: ningún set tiene edge estable; las 79 variables sin
# historia (satélite, CME, opciones, curva, noticias, crop progress) no aportan
# (Δ AUC ≈ 0 en ablation) y se filtran por cobertura; el SELL pierde en todas
# las variantes.
#
# El SELL no tuvo edge en NINGUNA variante fuera de muestra (−0.25% a −0.79%
# por señal a 14d; vende contra la tendencia). Se reporta como HOLD.
ALLOW_SELL = False


def production_feature_cols(columns) -> list[str]:
    """Variables candidatas del clasificador (compartido con audit_lookahead).
    El filtro de cobertura (MIN_REAL_SHARE) se aplica después, sobre el train."""
    return [c for c in columns if c not in NON_FEATURE_COLS | {"direction"}]


def coverage_filter(X: pd.DataFrame, min_share: float = MIN_REAL_SHARE) -> list[str]:
    """Columnas con al menos `min_share` de filas con dato real (≠0, ≠NaN)."""
    real_share = (X.notna() & (X != 0)).mean()
    return real_share[real_share >= min_share].index.tolist()


def _select_return_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dates_train: pd.Series | None = None,
    max_features: int = MAX_RETURN_FEATURES,
) -> list[str]:
    """
    Selecciona las top `max_features` features con señal consistente.

    Lógica:
      - Computa correlación con el target en TODO el training set (r_full).
      - Computa correlación en el 40% más reciente del training (r_recent).
      - Mantiene solo features con signo consistente: r_full × r_recent > 0.
        Esto descarta features cuya señal se invirtió en el régimen actual
        (ej. decomp_cycle con STL global era +0.042 histórico / -0.247 reciente).
      - Ordena por score = (|r_full| + |r_recent|) / 2.
      - Fallback a top-N por |r_full| si quedan < 10 candidatas.

    Sin data leakage: usa solo X_train / y_train, nunca val.
    """
    # Cobertura mínima: variables con <50% de datos reales en el train se
    # rellenaban con 0 y el árbol aprendía "0 = sin dato" (satélite, CME OI,
    # IV de opciones, noticias, curva: existen recién desde 2026).
    # Walk-forward 2019-2026: sacarlas no cambia el AUC (0.505 → 0.508).
    # Entran solas cuando acumulen historia.
    X_train = X_train[coverage_filter(X_train)]

    y_float = y_train.astype(float)
    full_corr = X_train.corrwith(y_float)

    # Mitad reciente del training (40% más nuevo por fecha o por posición)
    if dates_train is not None and len(dates_train) > 60:
        cutoff = pd.to_datetime(dates_train).quantile(0.60)
        recent_mask = pd.to_datetime(dates_train) >= cutoff
    else:
        recent_mask = pd.Series(
            [False] * int(len(X_train) * 0.6) + [True] * int(len(X_train) * 0.4),
            index=X_train.index,
        )

    if recent_mask.sum() > 30:
        X_rec = X_train[recent_mask.values]
        y_rec = y_train[recent_mask.values]
        recent_corr = X_rec.corrwith(y_rec.astype(float))
    else:
        recent_corr = full_corr

    candidates = []
    for col in X_train.columns:
        r_full   = full_corr.get(col, float("nan"))
        r_recent = recent_corr.get(col, float("nan"))
        if pd.isna(r_full) or pd.isna(r_recent):
            continue
        if abs(r_full) > 0.02 and r_full * r_recent > 0:
            score = (abs(r_full) + abs(r_recent)) / 2
            candidates.append((col, score))

    candidates.sort(key=lambda x: -x[1])
    selected = [c[0] for c in candidates[:max_features]]

    # Fallback: si quedan muy pocas, usar top por |r_full|
    if len(selected) < 10:
        selected = (
            full_corr.abs()
            .sort_values(ascending=False)
            .head(max_features)
            .index.tolist()
        )
        print(f"   [FeatureSel] Fallback a top-{max_features} por |r_full| "
              f"(solo {len(selected)} candidatas consistentes)")

    return selected


def train_returns_model(
    features_path: str,
    artifacts_dir: str,
    rolling_window_years: int | None = None,
) -> XGBClassifier:
    """
    Entrena XGBClassifier para predecir si el precio sube en 7 días.

    Parámetros
    ----------
    features_path        : path completo al CSV de features
    artifacts_dir        : directorio donde guardar returns_model.joblib
    rolling_window_years : si se pasa, limita el entrenamiento a los últimos N años
    """
    print(f"📂 Cargando features desde: {features_path}")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"No se encontró features CSV en: {features_path}")

    df = pd.read_csv(features_path)
    df = df.dropna(subset=[TARGET_REG])
    df = df.replace([float("inf"), float("-inf")], 0)

    # ── Ventana deslizante (rolling retrain) ──────────────────────
    if rolling_window_years is not None and "Date" in df.columns:
        cutoff = pd.Timestamp.today() - pd.DateOffset(years=rolling_window_years)
        df_full_len = len(df)
        df = df[pd.to_datetime(df["Date"]) >= cutoff]
        print(f"   📅 Rolling window {rolling_window_years}a: {df_full_len} → {len(df)} filas")

    # ── Target NETO de costos ──────────────────────────────────────
    # Restamos el costo round-trip estimado del retorno antes de binarizar.
    # Si el día tiene event_cost_mult > 1, el threshold sube en consecuencia.
    try:
        from src.trader.costs import estimate_round_trip_cost_pct
        if "event_cost_mult" in df.columns:
            cost_pct = df["event_cost_mult"].apply(
                lambda m: estimate_round_trip_cost_pct(float(m))
            )
        else:
            base_cost = estimate_round_trip_cost_pct(1.0)
            cost_pct = pd.Series(base_cost, index=df.index)
    except Exception:
        cost_pct = pd.Series(0.0005, index=df.index)  # 5 bps fallback

    df["ret_14d_net"] = df[TARGET_REG] - cost_pct
    df["direction"]   = (df["ret_14d_net"] > 0).astype(int)

    # ── Volatilidad realizada forward-looking (target del head de vol) ──
    # Std de retornos diarios en la ventana D+1..D+14, anualizada.
    if VOL_TARGET not in df.columns and "Soybeans" in df.columns:
        daily_ret = df["Soybeans"].pct_change()
        df[VOL_TARGET] = (
            daily_ret.shift(-1).rolling(14, min_periods=7).std() * np.sqrt(252)
        ).bfill().fillna(0)
    target_dist = df["direction"].value_counts()
    print(f"📊 Target: {target_dist[1]} subidas ({target_dist[1]/len(df)*100:.1f}%) "
          f"| {target_dist[0]} bajadas ({target_dist[0]/len(df)*100:.1f}%)")

    feature_cols = production_feature_cols(df.columns)
    print(f"📊 Features candidatas: {len(feature_cols)} (filtro de cobertura + top-{MAX_RETURN_FEATURES} en el train)")

    X     = df[feature_cols].fillna(0)
    y     = df["direction"]
    y_vol = df[VOL_TARGET].fillna(0) if VOL_TARGET in df.columns else None

    # ── Split temporal 80/20 con embargo ──────────────────────────
    # Embargo entre train y val para evitar contaminación por la ventana
    # forward del target (ret_14d_fwd usa D+1..D+14) y eventos macro programados.
    split = int(len(df) * 0.8)
    if "Date" in df.columns:
        df_dates    = pd.to_datetime(df["Date"])
        cut_date    = df_dates.iloc[split - 1]
        embargo_end = cut_date + pd.Timedelta(days=EMBARGO_DAYS)
        val_mask    = df_dates > embargo_end
        train_mask  = df_dates <= cut_date
        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        if y_vol is not None:
            y_vol_train = y_vol[train_mask]
            y_vol_val   = y_vol[val_mask]
        print(f"   [Embargo] {EMBARGO_DAYS}d entre {cut_date.date()} y {embargo_end.date()} "
              f"({split - len(X_train) + (len(df) - split - len(X_val))} filas dropeadas)")
    else:
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]
        if y_vol is not None:
            y_vol_train = y_vol.iloc[:split]
            y_vol_val   = y_vol.iloc[split:]

    # ── Selección de features (solo sobre training, sin data leakage) ──
    # Reduce de ~195 a top-40 features con señal consistente entre la
    # ventana completa de training y la mitad más reciente.
    # Descarta features cuyo signo de correlación se invirtió (régimen flip).
    if len(feature_cols) > MAX_RETURN_FEATURES:
        dates_train_ser = df_dates[train_mask].reset_index(drop=True) if "Date" in df.columns else None
        selected_feats  = _select_return_features(X_train, y_train, dates_train_ser)
    else:
        selected_feats  = feature_cols
    n_before        = len(feature_cols)
    feature_cols    = selected_feats
    X_train         = X_train[selected_feats]
    X_val           = X_val[selected_feats]
    if y_vol is not None:
        pass  # y_vol_train / y_vol_val no cambian
    print(f"   [FeatureSel] {len(selected_feats)}/{n_before} features "
          f"seleccionadas (consistentes en 5y y reciente)")

    # scale_pos_weight balancea clases si hay desbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = float(n_neg / n_pos) if n_pos > 0 else 1.0

    # ── Clasificador ──────────────────────────────────────────────
    # Parámetros = los validados en walk-forward (scripts/research_model_*.py).
    # Sin early stopping ni calibración isotónica: ambos se ajustaban sobre el
    # mismo tramo que después se reportaba (accuracy 72.5% inflado) y, medida
    # honestamente, la isotónica con ~250 filas EMPEORABA el Brier (0.253→0.315)
    # y producía saltos de P(suba) de 51%→97% entre días. Val queda como
    # holdout limpio, solo para reportar.
    xgb_params = dict(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, gamma=0.05,
        eval_metric="logloss", random_state=42,
    )
    model = XGBClassifier(scale_pos_weight=spw, **xgb_params)
    model.fit(X_train, y_train, verbose=False)

    # ── Métricas (holdout limpio) ─────────────────────────────────
    probs  = model.predict_proba(X_val)[:, 1]
    preds  = (probs > 0.5).astype(int)
    acc    = accuracy_score(y_val, preds)
    try:
        auc = roc_auc_score(y_val, probs)
    except Exception:
        auc = float("nan")

    print(f"📊 Clasificador (holdout) — Accuracy: {acc*100:.1f}% | ROC-AUC: {auc:.3f}")
    print(f"   P(sube) — mean: {probs.mean():.3f} | std: {probs.std():.3f} "
          f"| min: {probs.min():.3f} | max: {probs.max():.3f}")
    calibrator = None
    brier_raw  = float(np.mean((probs - y_val.values) ** 2))
    brier_cal  = brier_raw

    # BUY/SELL/HOLD distribution con thresholds por defecto
    buy_pct  = (probs > 0.58).mean() * 100
    sell_pct = (probs < 0.42).mean() * 100
    hold_pct = 100 - buy_pct - sell_pct
    print(f"   Señales (val) — BUY: {buy_pct:.0f}% | SELL: {sell_pct:.0f}% | HOLD: {hold_pct:.0f}%")

    # ── Head de volatilidad esperada (multi-output) ───────────────
    # Predice realized_vol_14d para sizing/conviction. No reemplaza al
    # clasificador — se entrena en paralelo con las mismas features.
    vol_model = None
    vol_mae   = None
    if y_vol is not None and y_vol_train.std() > 1e-6:
        try:
            vol_model = XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, gamma=0.05,
                random_state=42, eval_metric="mae",
            )
            vol_model.fit(X_train, y_vol_train,
                          eval_set=[(X_val, y_vol_val)], verbose=False)
            vol_pred = vol_model.predict(X_val)
            vol_mae  = float(mean_absolute_error(y_vol_val, vol_pred))
            print(f"📊 Vol head — MAE={vol_mae:.4f} | "
                  f"pred mean={vol_pred.mean():.3f} target mean={y_vol_val.mean():.3f}")
        except Exception as _e:
            print(f"   [WARN] vol head falló: {_e}")
            vol_model = None

    # ── SHAP explainability ───────────────────────────────────────
    # Calcula importancia SHAP media absoluta en el val set y persiste
    # en shap_explanation.json para el endpoint /api/shap_explanation.
    shap_top = []
    try:
        import shap as _shap
        _explainer = _shap.TreeExplainer(model)
        _shap_vals = _explainer.shap_values(X_val)
        _shap_mean = np.abs(_shap_vals).mean(axis=0)
        _shap_df = sorted(
            zip(feature_cols, _shap_mean.tolist()),
            key=lambda x: -x[1],
        )
        shap_top = [{"feature": f, "shap_mean_abs": round(v, 6)} for f, v in _shap_df[:20]]
        print(f"   🔍 SHAP top-3: " +
              ", ".join(f"{r['feature']}({r['shap_mean_abs']:.4f})" for r in shap_top[:3]))
    except ImportError:
        print("   [INFO] shap no instalado — pip install shap para SHAP explainability")
    except Exception as _se:
        print(f"   [WARN] SHAP falló: {_se}")

    # ── Modelo final: re-fit sobre train+val (toda la ventana) ─────
    # Igual que en el walk-forward: el modelo que predice hoy se entrena con
    # todo lo disponible hasta el último target conocido.
    _X_all = pd.concat([X_train, X_val]); _y_all = pd.concat([y_train, y_val])
    _spw_all = float((_y_all == 0).sum() / max((_y_all == 1).sum(), 1))
    model = XGBClassifier(scale_pos_weight=_spw_all, **xgb_params)
    model.fit(_X_all, _y_all, verbose=False)

    # ── Guardar ───────────────────────────────────────────────────
    os.makedirs(artifacts_dir, exist_ok=True)
    out_path = os.path.join(artifacts_dir, "returns_model.joblib")

    joblib.dump({
        "model":        model,
        "vol_model":    vol_model,
        "calibrator":   calibrator,         # IsotonicRegression o None
        "feature_cols": feature_cols,
        "model_type":   "classifier",
        "buy_thresh":   0.58,
        "sell_thresh":  0.42,
        "allow_sell":   ALLOW_SELL,
        "val_acc":      round(acc, 4),
        "val_auc":      round(auc, 4) if not np.isnan(auc) else None,
        "val_brier_raw": round(brier_raw, 4),
        "val_brier_cal": round(brier_cal, 4),
        "vol_mae":      round(vol_mae, 4) if vol_mae is not None else None,
        "embargo_days": EMBARGO_DAYS,
        "shap_top":     shap_top,
    }, out_path)

    # Persistir SHAP por separado como JSON legible para el endpoint
    if shap_top:
        import json as _json
        shap_path = os.path.join(artifacts_dir, "shap_explanation.json")
        _payload = {
            "generated_at":  pd.Timestamp.now().isoformat(),
            "model":         "XGBClassifier_direction_14d",
            "val_n":         len(X_val),
            "val_acc":       round(acc, 4),
            "val_auc":       round(auc, 4) if not np.isnan(auc) else None,
            "top_features":  shap_top,
        }
        with open(shap_path, "w", encoding="utf-8") as _f:
            _json.dump(_payload, _f, indent=2)
        print(f"💾 SHAP guardado en {shap_path}")

    print(f"✅ Clasificador guardado en {out_path}")
    return model
