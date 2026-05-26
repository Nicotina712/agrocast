"""
src/pipeline.py
Orquestador principal de AgroCast PRO.

Orden de ejecución:
  1. Descarga datos de mercado (con caché)
  2. Construye features de mercado
  3. Merge con features de noticias (si las hay)
  4. Calcula retornos forward
  5. Entrena modelo de precios (Ridge)
  6. Entrena modelo de retornos (XGBoost) — solo si no existe
  7. Genera forecast de 30 días
  8. Genera señales BUY/SELL/HOLD

✅ FIX: retornos forward se calculan en build_features (ya no se
   duplican aquí), eliminando el bloque redundante.
✅ FIX: paths de artifacts pasados explícitamente a cada función.
✅ FIX: carga de .env para NEWS_API_KEY.
"""

import os
import sys

import pandas as pd

# ── Fix encoding Windows (cmd no soporta UTF-8 por defecto) ──────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass  # Python < 3.7 o entorno sin reconfigure

# ── Resolver rutas ────────────────────────────────────────────────
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.insert(0, PROJECT_ROOT)

# ── Cargar variables de entorno ───────────────────────────────────
try:
    from dotenv import load_dotenv
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print("[OK] .env cargado desde " + env_path)
except ImportError:
    print("[WARN] python-dotenv no instalado; .env no se cargara automaticamente")

# ── Imports del proyecto ──────────────────────────────────────────
from src.data.load_data          import load_all_data
from src.data.load_external      import load_all_external
from src.features.build_features import make_features
from src.features.news_features  import build_news_features
from src.model.train             import train_model
from src.model.predict           import forecast_30d, save_forecast_csv
from src.model.train_returns     import train_returns_model
from src.model.predict_returns   import predict_returns

# ── Directorios ───────────────────────────────────────────────────
ARTIFACTS_DIR        = os.path.join(PROJECT_ROOT, "artifacts")
DATA_DIR             = os.path.join(PROJECT_ROOT, "data")
MODEL_ARTIFACTS_DIR  = os.path.join(CURRENT_DIR, "model", "artifacts")

os.makedirs(ARTIFACTS_DIR,       exist_ok=True)
os.makedirs(DATA_DIR,            exist_ok=True)
os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)


def run_pipeline() -> None:
    print("⚙️  Ejecutando pipeline AgroCast…")

    # ── 1. Datos de mercado ───────────────────────────────────────
    df             = load_all_data()
    last_real_date = df["Date"].max()

    # ── 2. Features de mercado ────────────────────────────────────
    # SoybeanMeal y SoybeanOil NO van como exog (generarían lags crudos muy
    # correlacionados con el target en nivel). En cambio, build_features
    # los convierte en crush_spread — el indicador económico correcto.
    exog_cols = ["Maize", "Oil", "Dollar"]
    exog_cols = [c for c in exog_cols if c in df.columns]

    features = make_features(
        df,
        date_col="Date",
        target_col="Soybeans",
        exog_cols=exog_cols,
    )

    # ── 3. Fuentes externas (CSVs históricos + FRED si disponible) ─
    print("📦 Cargando datos externos...")
    ext_df = load_all_external(start_date=df["Date"].min().date())
    if not ext_df.empty:
        ext_df["Date"] = pd.to_datetime(ext_df["Date"])
        features = features.merge(ext_df, on="Date", how="left")
        # Forward-fill para días sin datos externos, luego fill con 0
        ext_cols = [c for c in ext_df.columns if c != "Date"]
        for col in ext_cols:
            if col in features.columns:
                features[col] = features[col].ffill().fillna(0)
        print(f"   ✅ {len(ext_cols)} columnas externas añadidas")

    # ── 4. Historial de sentimiento de noticias (RSS/NewsAPI) ────────
    # news_sentiment_history.csv se acumula cada vez que news_server
    # actualiza noticias. Se mergea con features para que el modelo
    # vea el sentimiento real (antes era siempre 0).
    print("🧠 Cargando historial de sentimiento de noticias…")
    try:
        from src.features.news_history import load_news_sentiment_history
        news_hist = load_news_sentiment_history()
        if not news_hist.empty:
            news_hist["Date"] = pd.to_datetime(news_hist["Date"])
            # Normalizar volumen a escala 0-1 para que no domine por magnitud
            max_vol = news_hist["news_volume"].replace(0, 1).max()
            news_hist["news_volume_norm"] = news_hist["news_volume"] / max_vol
            features = features.merge(
                news_hist[["Date", "news_sentiment", "news_volume",
                           "news_volume_norm", "china_score", "weather_score", "macro_score"]],
                on="Date", how="left",
            )
            # Forward-fill: el sentimiento del día persiste hasta el siguiente registro
            for col in ("news_sentiment", "news_volume", "news_volume_norm",
                        "china_score", "weather_score", "macro_score"):
                features[col] = features[col].ffill().fillna(0.0)
            print(f"   ✅ Sentimiento RSS: {news_hist.shape[0]} días | "
                  f"rango {news_hist['Date'].min().date()} → {news_hist['Date'].max().date()}")
        else:
            print("   ℹ️  Sin historial RSS aún — las columnas de sentimiento quedan en 0")
            for col in ("news_sentiment", "news_volume", "news_volume_norm",
                        "china_score", "weather_score", "macro_score"):
                features[col] = 0.0
    except Exception as _e:
        print(f"   [WARN] news_history: {_e}")
        for col in ("news_sentiment", "news_volume", "news_volume_norm",
                    "china_score", "weather_score", "macro_score"):
            if col not in features.columns:
                features[col] = 0.0

    # news_impact como promedio ponderado de scores por topic
    features["news_impact"] = (
        features["china_score"]   * 0.35 +
        features["weather_score"] * 0.40 +
        features["macro_score"]   * 0.25
    ).fillna(0.0)

    # news_shock: señal no lineal (amplifica eventos extremos)
    features["news_shock"] = features["news_sentiment"] * features["news_volume_norm"]
    features["news_shock"] = features["news_shock"].apply(lambda x: x * abs(x))

    # news_velocity_7d: ratio artículos hoy vs media móvil 7d.
    # Picos de cobertura preceden volatilidad de precio (paper §4, informe IA).
    vol_rolling = features["news_volume"].rolling(7, min_periods=1).mean().replace(0, 1)
    features["news_velocity_7d"] = (features["news_volume"] / vol_rolling).clip(0, 10).fillna(1.0)
    print("   ✅ news_velocity_7d añadida")

    # ── 4b. Intel LLM (señales desagregadas por driver) ──────────────
    try:
        intel_path = os.path.join(PROJECT_ROOT, "data", "news_intel_history.csv")
        if os.path.exists(intel_path):
            intel_hist = pd.read_csv(intel_path, parse_dates=["Date"])
            features["Date"] = pd.to_datetime(features["Date"])
            features = features.merge(intel_hist, on="Date", how="left")
            intel_cols = [c for c in intel_hist.columns if c != "Date"]
            for col in intel_cols:
                features[col] = features[col].ffill().fillna(0.0)
            print(f"   ✅ Intel LLM: {len(intel_cols)} señales por driver | "
                  f"{intel_hist.shape[0]} días")
        else:
            print("   ℹ️  Sin historial de intel LLM aún")
    except Exception as _e:
        print(f"   [WARN] intel history: {_e}")

    # ── 5. Features WASDE (surprise proxy + opcional actuals) ────────
    try:
        from src.data.wasde_surprise import build_wasde_features
        wasde_feats = build_wasde_features(features[["Date", "Soybeans"]])
        if not wasde_feats.empty and len(wasde_feats.columns) > 1:
            wasde_feats["Date"] = pd.to_datetime(wasde_feats["Date"])
            features["Date"]    = pd.to_datetime(features["Date"])
            features = features.merge(wasde_feats, on="Date", how="left")
            wasde_cols = [c for c in wasde_feats.columns if c != "Date"]
            for col in wasde_cols:
                if col in features.columns:
                    features[col] = features[col].ffill().fillna(0)
            print(f"   ✅ WASDE features: {wasde_cols}")
    except Exception as _e:
        print(f"   [WARN] wasde_surprise: {_e}")

    # ── 5b. Curve features (front-next spread, roll yield, contango/backw) ──
    try:
        from src.trader.curve_features import snapshot_curve, load_curve_features
        snapshot_curve()  # añade row del día a curve_history.csv
        features = load_curve_features(features)
        print("   ✅ Curve features añadidas (spread/roll yield/structure)")
    except Exception as _e:
        print(f"   [WARN] curve_features: {_e}")

    # ── 5c. Calendario de eventos (WASDE/Acreage/Stocks) ─────────────
    try:
        from src.data.event_calendar import build_event_features
        features = build_event_features(features)
        print("   ✅ Event calendar features añadidas")
    except Exception as _e:
        print(f"   [WARN] event_calendar: {_e}")

    # ── 5d. COT regime discreto ──────────────────────────────────────
    try:
        from src.trader.cot_regime import add_cot_regime_features
        features = add_cot_regime_features(features)
        print("   ✅ COT regime discreto añadido")
    except Exception as _e:
        print(f"   [WARN] cot_regime: {_e}")

    # Helper: ejecutar fetch con hard-timeout para que ningún endpoint
    # externo lento bloquee el pipeline completo.
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FT
    def _bounded(fn, timeout_s, label, *a, **k):
        with ThreadPoolExecutor(max_workers=1) as _ex:
            try:
                return _ex.submit(fn, *a, **k).result(timeout=timeout_s)
            except _FT:
                print(f"   [WARN] {label}: timeout {timeout_s}s — skip")
            except Exception as _e2:
                print(f"   [WARN] {label}: {_e2}")
        return None

    # ── 5e. CME Group: snapshot OI/volumen + features ────────────────
    try:
        from src.data.fetch_cme import run_cme_snapshot, load_cme_features
        _bounded(run_cme_snapshot, 30, "CME snapshot")
        features = load_cme_features(features)
        print("   ✅ CME Group features añadidas (OI, volumen, spread)")
    except Exception as _e:
        print(f"   [WARN] fetch_cme: {_e}")

    # ── 5e.bis CVOL proxy via IV ATM Yahoo (Fix #6) ─────────────────
    try:
        from src.data.fetch_cvol import run_cvol_snapshot, load_cvol_features
        _bounded(run_cvol_snapshot, 30, "CVOL/IV snapshot")
        features = load_cvol_features(features)
        print("   ✅ CVOL features añadidas (IV ATM, skew, z-score)")
    except Exception as _e:
        print(f"   [WARN] fetch_cvol: {_e}")

    # ── 5f. NASA POWER: clima/condición regiones sojeras ─────────────
    try:
        from src.data.fetch_satellite import fetch_satellite_window, load_satellite_features
        sat_path = os.path.join(DATA_DIR, "satellite_history.csv")
        needs_sat = True
        days_back = 720
        if os.path.exists(sat_path):
            try:
                _last = pd.read_csv(sat_path, parse_dates=["Date"])["Date"].max()
                needs_sat = (pd.Timestamp.today() - _last).days >= 5
                days_back = 120
            except Exception:
                pass
        if needs_sat:
            _bounded(fetch_satellite_window, 90, "NASA POWER fetch", days_back)
        features = load_satellite_features(features)
        print("   ✅ NASA POWER (clima/stress) añadido")
    except Exception as _e:
        print(f"   [WARN] fetch_satellite: {_e}")

    # ── 5f.bis ENSO + drought macro climate (Fix #10) ──────────────
    try:
        from src.data.fetch_climate_macro import load_climate_macro_features
        features = load_climate_macro_features(features)
        print("   ✅ Climate macro features añadidas (ENSO + drought)")
    except Exception as _e:
        print(f"   [WARN] fetch_climate_macro: {_e}")

    # ── 5f.ter Forecast climático FORWARD (H5) ────────────────────
    # Diferencia con 5f.bis: 5f.bis es clima REALIZADO; éste es FORWARD
    # (outlooks 3-6 meses adelante de ENSO). La deep research apuntó a forecasts
    # forward como predictor causal de cosecha futura → precio futuro.
    try:
        from src.data.fetch_climate_forecast import get_climate_forecast, load_climate_forecast_features
        cf = _bounded(get_climate_forecast, 30, "Climate forecast fetch")
        features = load_climate_forecast_features(features)
        if cf and cf.get("ok"):
            print(f"   ✅ Climate forecast: ENSO 3m={cf.get('enso_value_3m', '?')} "
                  f"phase={cf.get('enso_phase_3m', '?')} (fallback={cf.get('fallback')})")
        else:
            print(f"   [INFO] Climate forecast: usando heurística sobre histórico")
    except Exception as _e:
        print(f"   [WARN] climate_forecast: {_e}")

    # ── 5f.ter Basis Uruguay (prima local que cobra el productor) ────
    # El basis es lo que efectivamente recibe el productor uruguayo vs CBOT.
    # Útil como feature: cuando el basis se aprieta (sube), exporta más;
    # cuando se ensancha (baja), demanda local débil. Predice flujos.
    try:
        basis_path = os.path.join(DATA_DIR, "basis_history.csv")
        if os.path.exists(basis_path):
            bh = pd.read_csv(basis_path)
            bh["Date"] = pd.to_datetime(bh["date"]).dt.normalize()
            bh = bh.rename(columns={"basis_usd_ton": "basis_uy_usd_ton"})[["Date", "basis_uy_usd_ton"]]
            features["Date"] = pd.to_datetime(features["Date"])
            features = features.merge(bh, on="Date", how="left")
            features["basis_uy_usd_ton"] = features["basis_uy_usd_ton"].ffill().fillna(0)
            # Cambio del basis (compresión/expansión)
            features["basis_uy_chg7"] = features["basis_uy_usd_ton"].diff(7).fillna(0)
            features["basis_uy_chg30"] = features["basis_uy_usd_ton"].diff(30).fillna(0)
            print(f"   ✅ Basis Uruguay: {bh.shape[0]} días | rango {bh['Date'].min().date()} → {bh['Date'].max().date()}")
        else:
            print(f"   [INFO] basis_history.csv no existe — se omite basis Uruguay")
    except Exception as _e:
        print(f"   [WARN] basis Uruguay: {_e}")

    # ── 5g. USDA Crop Progress weekly ────────────────────────────────
    try:
        from src.data.fetch_crop_progress import fetch_crop_progress, load_crop_progress_features
        cp_path = os.path.join(DATA_DIR, "crop_progress.csv")
        years_back = 3 if not os.path.exists(cp_path) else 1
        needs_cp = True
        if os.path.exists(cp_path):
            try:
                _last = pd.read_csv(cp_path, parse_dates=["Date"])["Date"].max()
                needs_cp = (pd.Timestamp.today() - _last).days >= 5
            except Exception:
                pass
        if needs_cp:
            _bounded(fetch_crop_progress, 60, "USDA Crop Progress fetch", years_back)
        features = load_crop_progress_features(features)
        print("   ✅ USDA Crop Progress añadido")
    except Exception as _e:
        print(f"   [WARN] fetch_crop_progress: {_e}")

    # ── 6. Limpieza final con validación de completitud ─────────
    # Forward-fill first, then fill remaining NaN with 0.
    # Track imputation rate so downstream consumers know data quality.
    _pre_na = features.isna().sum().sum()
    _total_cells = features.shape[0] * features.shape[1]
    features = features.ffill().fillna(0)   # removed bfill (prevents future leak)
    _impute_rate = round(_pre_na / _total_cells * 100, 2) if _total_cells > 0 else 0

    # Health check: if >10% of feature cells were imputed, warn loudly
    _IMPUTE_WARN_PCT = 10.0
    if _impute_rate > _IMPUTE_WARN_PCT:
        print(f"   ⚠️  ALERTA: {_impute_rate}% de celdas de features fueron imputadas "
              f"({_pre_na}/{_total_cells}). Posible degradación de señal.")
    else:
        print(f"   ✅ Feature matrix health: {_impute_rate}% imputado ({_pre_na} celdas)")

    # Save imputation metadata for the freshness indicator
    _health_path = os.path.join(DATA_DIR, "pipeline_health.json")
    try:
        import json as _json
        _health = {
            "impute_rate_pct": _impute_rate,
            "total_features": features.shape[1],
            "total_rows": features.shape[0],
            "imputed_cells": int(_pre_na),
            "status": "ok" if _impute_rate <= _IMPUTE_WARN_PCT else "degraded",
            "timestamp": datetime.now().isoformat(),
        }
        with open(_health_path, "w") as _hf:
            _json.dump(_health, _hf, indent=2)
    except Exception:
        pass

    # ── 6b. Guardar features ──────────────────────────────────────
    features_path = os.path.join(DATA_DIR, "features.csv")
    features.to_csv(features_path, index=False)
    print(f"💾 Features guardadas en {features_path}")

    # ── 7. Modelo de precios ──────────────────────────────────────
    # Si ROLLING_WINDOW_YEARS está seteado (ej. por el job mensual del scheduler),
    # se entrena con ventana deslizante en lugar de datos acumulativos.
    _rolling_years = int(os.getenv("ROLLING_WINDOW_YEARS", "5")) or None
    if _rolling_years:
        print(f"   📅 Rolling retrain activado: ventana {_rolling_years} años")
    train_model(features, target="Soybeans", artifacts_dir=ARTIFACTS_DIR,
                rolling_window_years=_rolling_years)

    # ── 8. Modelo de retornos ─────────────────────────────────────
    # El modelo de retornos siempre se reentrena: es rápido (~5s) y así
    # los cambios de hiperparámetros y features se aplican en cada run.
    returns_model_path = os.path.join(MODEL_ARTIFACTS_DIR, "returns_model.joblib")
    needs_train = True

    if needs_train:
        train_returns_model(features_path, MODEL_ARTIFACTS_DIR,
                            rolling_window_years=_rolling_years)
        print("[OK] Modelo de retornos entrenado")

    # ── 9. Forecast 30 días ───────────────────────────────────────
    forecast = forecast_30d(
        features,
        target="Soybeans",
        date_col="Date",
        artifacts_dir=ARTIFACTS_DIR,
        real_last_date=last_real_date,
    )

    forecast_path = os.path.join(ARTIFACTS_DIR, "forecast.csv")
    save_forecast_csv(forecast, forecast_path)

    # ── 9.bis Forecast variante "horizons" (A/B con flag) ────────
    # FORECAST_VARIANT controla qué se publica:
    #   legacy   → solo forecast.csv (sin cambios)
    #   horizons → forecast.csv se sobreescribe con horizons + forecast_horizons.csv
    #   both     → forecast.csv intacto + forecast_horizons.csv adicional (default)
    #
    # HORIZONS_REFIT_DAYS controla la frecuencia de RE-ENTRENAMIENTO de los
    # modelos por horizonte (default 7 = semanal). El forecast diario siempre
    # se regenera con los pesos vigentes; reentrenar más seguido sólo agrega
    # ruido sin más datos. Usar 0 para forzar reentrenamiento en cada run.
    _variant     = os.getenv("FORECAST_VARIANT", "both").lower()
    _refit_days  = int(os.getenv("HORIZONS_REFIT_DAYS", "7"))
    if _variant in ("horizons", "both"):
        try:
            from src.model.train_horizons   import train_all as train_horizons_all
            from src.model.predict_horizons import forecast_curve as horizons_curve
            horizons_artifacts = os.path.join(ARTIFACTS_DIR, "horizons")
            os.makedirs(horizons_artifacts, exist_ok=True)
            meta_path = os.path.join(horizons_artifacts, "horizons_meta.json")

            # Decide si reentrenar
            needs_retrain = True
            if _refit_days > 0 and os.path.exists(meta_path):
                age_days = (pd.Timestamp.now().timestamp() - os.path.getmtime(meta_path)) / 86400
                if age_days < _refit_days:
                    needs_retrain = False
                    print(f"   ⏩ Modelos horizons frescos ({age_days:.1f}d < {_refit_days}d) — skip retrain")

            print(f"   🔄 Forecast variante 'horizons' (variant={_variant}, refit_days={_refit_days})")
            if needs_retrain:
                train_horizons_all(features, horizons_artifacts)
            new_fc = horizons_curve(
                features, target="Soybeans", date_col="Date",
                artifacts_dir=horizons_artifacts, real_last_date=last_real_date, steps=30,
            )
            new_fc_path = os.path.join(ARTIFACTS_DIR, "forecast_horizons.csv")
            new_fc.to_csv(new_fc_path, index=False)
            print(f"   💾 Forecast horizons → {new_fc_path}")
            if _variant == "horizons":
                # Cutover total: horizons reemplaza al legacy
                new_fc.to_csv(forecast_path, index=False)
                print(f"   🎯 Cutover activo: forecast.csv = horizons")
        except Exception as _e:
            print(f"   [WARN] forecast horizons falló (se mantiene legacy): {_e}")

    # ── 9.ter Detector de régimen + score de confianza ───────────
    # Output → artifacts/regime.json. La UI lo usa para semáforo de confianza
    # y framing del precio puntual ("predicción" vs "rango con probabilidad").
    try:
        from src.model.regime import save_regime
        regime_out = save_regime(features, ARTIFACTS_DIR)
        print(f"   🌡️  Régimen: {regime_out.get('regime', '?')} — {regime_out.get('explanation', '')[:80]}")
    except Exception as _e:
        print(f"   [WARN] regime detector falló: {_e}")

    # ── 9.sexies Decision Classifier multi-horizonte + multi-profile (Fase 1) ─────
    # Cost-aware binary classifier sobre "¿esperar pagó el costo de carrying?"
    # Multi-horizon (7d, 15d, 30d) × multi-profile (default, low_cost, high_cost,
    # liquidity_need, quality_aware) + análogos explicativos.
    # Validado empíricamente como la mejor aproximación predictiva (PnL -1.36%/año
    # vs always-sell). Panel INFORMATIVO, NO decisor.
    # Output → artifacts/decision_classifier.json (default) + artifacts/decision_classifier/{profile}.json
    try:
        from src.model.decision_classifier import save_all_profiles
        dc_results = save_all_profiles(features, ARTIFACTS_DIR)
        ok_profiles = [p for p, r in dc_results.items() if r.get("ok")]
        # Resumen por profile
        for prof in ok_profiles:
            r = dc_results[prof]
            best_h = r.get("best_horizon", "?")
            h_data = r.get("horizons", {}).get(best_h or "", {})
            p = h_data.get("prob_wait_pays_calibrated") if isinstance(h_data, dict) else None
            print(f"   🎯 Decision[{prof:<14}] best_h={best_h or '?':<3}  P(WAIT)={p}")
        failed = [p for p in dc_results if not dc_results[p].get("ok")]
        if failed:
            print(f"   [WARN] decision_classifier failed profiles: {failed}")
    except Exception as _e:
        print(f"   [WARN] decision_classifier: {_e}")

    # ── 9.septies Backtest comparativo decision classifier (Fase 2.3) ──
    # OOS backtest 6 estrategias × 5 profiles × 3 horizontes.
    # Se ejecuta solo si el módulo está disponible (no bloquea el pipeline).
    # Output → artifacts/backtest_decision/{profile}.json
    try:
        from src.model.backtest_decision import save_all_backtest_profiles
        bt_results = save_all_backtest_profiles(features, ARTIFACTS_DIR)
        ok_bt = [p for p, r in bt_results.items() if r.get("ok")]
        fail_bt = [p for p in bt_results if not bt_results[p].get("ok")]
        if ok_bt:
            # Resumen: model_partial mean delta en horizonte 15d para profile default
            dr = bt_results.get("default", {})
            h15 = dr.get("horizons", {}).get("15d", {})
            mp = h15.get("strategies", {}).get("model_partial", {})
            mean_d = mp.get("mean_pnl_usd_ton")
            print(f"   📊 Backtest[default 15d] model_partial Δmedio={mean_d} USD/ton"
                  f" · perfiles OK: {ok_bt}")
        if fail_bt:
            print(f"   [WARN] backtest_decision failed profiles: {fail_bt}")
    except Exception as _e:
        print(f"   [WARN] backtest_decision: {_e}")

    # ── 9.octies Event Memory + Hybrid Backtest (Market Intelligence V1) ──
    # Construye event memory retroactivo (eventos narrativos + outcomes) y
    # ejecuta backtest comparativo ML vs Narrativa vs Hibrido.
    # Output → artifacts/event_memory.csv + .json, artifacts/hybrid_backtest/{profile}.json
    try:
        from src.intel.event_intelligence import save_event_memory
        em_summary = save_event_memory(features, ARTIFACTS_DIR)
        n_ev = em_summary.get("n_events", 0)
        fade = em_summary.get("fade_rate_7d_pct", "?")
        print(f"   🔍 Event memory: {n_ev} eventos, fade rate 7d={fade}%")
    except Exception as _e:
        print(f"   [WARN] event_memory: {_e}")

    try:
        from src.intel.hybrid_model import save_hybrid_backtest
        hbt = save_hybrid_backtest(features, ARTIFACTS_DIR, profile_name="default")
        h15 = hbt.get("horizons", {}).get("15d", {})
        if h15.get("ok"):
            hy = h15.get("strategies", {}).get("hybrid", {})
            ml = h15.get("strategies", {}).get("ml_only", {})
            print(f"   Hybrid backtest[default 15d]: hybrid={hy.get('mean_pnl_usd_ton','?')}"
                  f" ml_only={ml.get('mean_pnl_usd_ton','?')} USD/ton")
    except Exception as _e:
        print(f"   [WARN] hybrid_backtest: {_e}")

    # ── 9.novies Narrative Forecast (rango diario + multi-horizonte) ──
    # Genera forecast narrativo con rangos esperados (1d, 7d, 15d, 30d)
    # basado en analogos del event_memory.
    # Output → artifacts/narrative_forecast/latest.json
    try:
        from src.intel.narrative_forecast import save_narrative_forecast
        nf = save_narrative_forecast(features, ARTIFACTS_DIR)
        fc = nf.get("forecast", {})
        if fc.get("ok"):
            d1 = fc.get("forecasts", {}).get("1d", {})
            if d1.get("available"):
                rng = d1.get("range_pct", {})
                print(f"   Narrative forecast 1d: [{rng.get('q10','?')}%, {rng.get('q90','?')}%] "
                      f"P(up)={d1.get('p_up','?')}")
    except Exception as _e:
        print(f"   [WARN] narrative_forecast: {_e}")

    # ── 9.quinquies Shock Engine (catalog + analog) ───────────────
    # Detecta si hay shock activo HOY, busca análogos en histórico, calcula
    # estadísticas agregadas y emite recomendación condicional al productor.
    # Output → artifacts/active_shock.json + shock_catalog.csv
    try:
        from src.model.shock_engine import save_active_shock
        sh = save_active_shock(features, ARTIFACTS_DIR)
        cur = sh.get("current", {})
        if cur.get("is_shock"):
            rec = sh.get("recommendation", {})
            print(f"   ⚡ SHOCK ACTIVO: {cur.get('shock_type')} ({cur.get('shock_direction')}) "
                  f"ret_5d={cur.get('ret_5d_pct')}% — análogos={sh.get('analogs_found')} "
                  f"→ {rec.get('action')}")
        else:
            print(f"   🟢 Sin shock activo — modelo regular gobierna")
    except Exception as _e:
        print(f"   [WARN] shock_engine: {_e}")

    # ── 9.quater Markov-Switching Regression (H4) ────────────────
    # Modelo probabilistico de régimen sobre log-returns con K=2 estados.
    # Output → artifacts/regime_switching.json — usado por la API y como
    # ajuste sugerido al α del modelo horizons.
    try:
        from src.model.regime_switching import save_regime_switching
        ms_out = save_regime_switching(features, ARTIFACTS_DIR)
        if ms_out.get("ok"):
            print(f"   📊 Markov-Switching: {ms_out.get('current_state')} "
                  f"(p={ms_out.get('current_state_prob')}) — "
                  f"α_adj={ms_out.get('alpha_adjustment')}")
        else:
            print(f"   [WARN] Markov-Switching: {ms_out.get('error')}")
    except Exception as _e:
        print(f"   [WARN] Markov-Switching falló: {_e}")

    # ── 9a. Forecast mensual robusto (ETS/Holt-Winters) ───────────
    try:
        from src.model.monthly_robust import build_monthly_forecast
        build_monthly_forecast(horizon_days=90)
    except Exception as _e:
        print(f"   [INFO] monthly_robust: {_e}")

    # ── 10. Señales ───────────────────────────────────────────────
    signals = predict_returns(features, artifacts_dir=MODEL_ARTIFACTS_DIR)

    signals_path = os.path.join(ARTIFACTS_DIR, "signals.csv")
    signals.to_csv(signals_path, index=False)
    print(f"💾 Señales guardadas en {signals_path}")

    # ── 9b. Snapshot de accountability ───────────────────────────
    # IMPORTANTE: se ejecuta DESPUÉS de predict_returns para que
    # get_signal_breakdown() lea signals.csv fresco (no el del run anterior).
    try:
        from src.trader.accountability import save_forecast_snapshot
        from src.trader.signal_breakdown import get_signal_breakdown
        breakdown = get_signal_breakdown()
        save_forecast_snapshot(
            price_now=float(features["Soybeans"].iloc[-1]),
            signal=breakdown.get("composite_signal", "HOLD"),
            forecast_df=forecast,
            score=breakdown.get("composite_score"),
        )
    except Exception as _e:
        print(f"   [INFO] Accountability snapshot: {_e}")

    # ── 10.bis ETL manifest (Fix #11 Bronze/Silver/Gold) ──────────
    try:
        from src.infra.etl_manifest import write_manifest, validate_manifest
        write_manifest()
        for _w in validate_manifest():
            print(f"   {_w}")
    except Exception as _e:
        print(f"   [WARN] etl_manifest: {_e}")

    # ── 10b. Bayesian ensemble (modelo + LLM) ────────────────────
    try:
        from src.trader.ensemble import ensemble_signal
        import json as _json
        # p_model = P(↑) última fila; expected_return está centrado en 0
        last = signals.iloc[-1]
        p_model_last = float(last["expected_return"]) + 0.5

        llm_stance = None; llm_conviction = None
        syn_path = os.path.join(DATA_DIR, "market_synthesis.json")
        if os.path.exists(syn_path):
            try:
                syn = _json.loads(open(syn_path, "r", encoding="utf-8").read())
                llm_stance     = syn.get("stance") or syn.get("market_stance")
                llm_conviction = syn.get("conviction") or syn.get("confidence")
            except Exception:
                pass

        ens = ensemble_signal(p_model_last, llm_stance, llm_conviction)
        ens_path = os.path.join(ARTIFACTS_DIR, "ensemble_signal.json")
        with open(ens_path, "w", encoding="utf-8") as _f:
            _json.dump(ens, _f, indent=2)
        print(f"   ✅ Ensemble: {ens['signal_ensemble']} "
              f"(p_model={ens['p_model']:.2f} p_llm={ens['p_llm']} "
              f"p_ens={ens['p_ensemble']:.2f} disag={ens['disagreement']:.2f})")
    except Exception as _e:
        print(f"   [INFO] Ensemble: {_e}")

    # ── 11. Paper Trading — registrar señal y cerrar trades ──────
    try:
        from src.trader.paper_trading import run_paper_trading_cycle
        pt_capital  = float(os.getenv("PAPER_TRADING_CAPITAL",  "10000"))
        pt_risk_pct = float(os.getenv("PAPER_TRADING_RISK_PCT", "1.0"))
        pt_result   = run_paper_trading_cycle(pt_capital, pt_risk_pct)
        if pt_result["new_trade_opened"]:
            print(f"   [PaperTrading] Nuevo trade: {pt_result['signal']}")
        if pt_result["closed_this_cycle"] > 0:
            print(f"   [PaperTrading] Cerrados: {pt_result['closed_this_cycle']} trades")
    except Exception as _e:
        print(f"   [INFO] Paper trading: {_e}")

    # ── 12. Alertas Telegram (si bot configurado) ─────────────────
    try:
        from src.alerts.telegram_bot import check_and_alert_signal, check_and_alert_price_target
        check_and_alert_signal()
        check_and_alert_price_target()
    except Exception as _e:
        print(f"   [INFO] Telegram no configurado: {_e}")

    # ── 13. Alertas WhatsApp (si Twilio configurado) ──────────────
    try:
        from src.alerts.whatsapp_bot import (
            check_and_alert_signal as wa_signal,
            check_and_alert_price_target as wa_price,
        )
        wa_signal()
        wa_price()
    except Exception as _e:
        print(f"   [INFO] WhatsApp no configurado: {_e}")

    # ── 13b. Harvest Plan — check triggers & send alerts ──────────
    try:
        from src.producer.harvest_plan import check_triggers, send_plan_alerts
        from src.data.basis_forecast import BUSHELS_PER_TON, SEASONAL_BASIS
        # Get current price in USD/ton from pipeline data
        _price_usc_bu = float(features["Soybeans"].iloc[-1])
        _price_usd_ton = _price_usc_bu * BUSHELS_PER_TON / 100
        _month = pd.to_datetime(features["Date"].iloc[-1]).month
        _basis = SEASONAL_BASIS.get(_month, -25)
        _local_price_usd_ton = _price_usd_ton + _basis
        # Get signal from ensemble if available
        _sell_signal = "ESPERAR"
        try:
            _ens_signal = ens.get("signal_ensemble", "HOLD") if 'ens' in dir() else "HOLD"
            _sell_signal = {"SELL": "VENDER", "BUY": "ESPERAR", "HOLD": "ESPERAR"}.get(
                _ens_signal, "ESPERAR")
        except Exception:
            pass
        hp_result = check_triggers(
            current_price_usd_ton=_local_price_usd_ton,
            sell_signal=_sell_signal,
        )
        if hp_result.get("alerts"):
            sent = send_plan_alerts(hp_result["alerts"])
            print(f"   [HarvestPlan] {len(hp_result['alerts'])} triggers fired, "
                  f"{sent} alerts sent")
        else:
            print(f"   [HarvestPlan] No triggers fired")
    except Exception as _e:
        print(f"   [INFO] Harvest plan: {_e}")

    # ── 14. Señal Argentina ampliada (cepo + retenciones + CIARA) ──
    try:
        from src.data.fetch_argentina import get_argentina_supply_signal
        arg = get_argentina_supply_signal()
        print(f"   [AR] Score: {arg.get('supply_score', '?')} | "
              f"Cepo: {'activo' if arg.get('cepo_activo') else 'inactivo'} | "
              f"Impacto: {arg.get('impacto_precio', '?')}")
    except Exception as _e:
        print(f"   [INFO] Argentina signal: {_e}")

    # ── 15. Basis Uruguay (caché 24h) + Basis Forecast (GARCH) ────
    try:
        from src.data.fetch_basis_uruguay import get_basis_uruguay
        get_basis_uruguay()
    except Exception as _e:
        print(f"   [INFO] Basis Uruguay: {_e}")

    try:
        from src.data.basis_forecast import forecast_basis
        bf = forecast_basis()
        if bf.get("ok"):
            cs = bf["current_state"]
            print(f"   [BasisForecast] Basis={cs['basis_usd_ton']} USD/ton | "
                  f"Régimen={cs['regime']} | Z={cs['zscore_5y']:.1f}")
    except Exception as _e:
        print(f"   [INFO] Basis forecast: {_e}")

    # ── 16. WASDE API oficial (caché 6h) ──────────────────────────
    try:
        from src.data.wasde_api import get_wasde_official
        wasde_off = get_wasde_official()
        print(f"   [WASDE API] Signal: {wasde_off.get('signal', '?')} | "
              f"Stocks: {wasde_off.get('world', {}).get('ending_stocks_mmt', '?')} MMT")
    except Exception as _e:
        print(f"   [INFO] WASDE API: {_e}")

    # ── 17. Brazil Export Pace (caché 24h) ────────────────────────
    try:
        from src.data.fetch_brazil_exports import get_brazil_export_pace
        get_brazil_export_pace()
    except Exception as _e:
        print(f"   [INFO] Brazil exports: {_e}")

    # ── 18. China Demand Module (caché 24h) ───────────────────────
    try:
        from src.data.fetch_china_demand import get_china_demand
        get_china_demand()
    except Exception as _e:
        print(f"   [INFO] China demand: {_e}")

    # ── 19. Brief semanal (solo lunes) ────────────────────────────
    try:
        from src.alerts.weekly_brief import generate_weekly_brief
        generate_weekly_brief()
    except Exception as _e:
        print(f"   [INFO] Weekly brief: {_e}")

    # ── 20. Intelligence Engine — debate multi-agente (1x/día) ───
    # Corre siempre que haya ANTHROPIC_API_KEY. force=True bypasea el cache
    # interno (TTL 1h) y el gate de market_hours — apropiado porque el
    # pipeline ya solo corre 1x/día vía GitHub Actions.
    # El flag IE_SKIP=1 permite omitirlo en ejecuciones locales de test.
    try:
        if os.environ.get("IE_SKIP") == "1":
            print("   [IE] Skipped — IE_SKIP=1")
        elif os.environ.get("ANTHROPIC_API_KEY"):
            from src.intel.intelligence_engine import run_intelligence_engine
            ie_result = run_intelligence_engine(force=True)
            verdict = ie_result.get("verdict", {})
            from_cache = ie_result.get("from_cache", False)
            print(f"   [IE] Verdict: {verdict.get('verdict', '?')} "
                  f"(conf={verdict.get('confidence', '?')}) "
                  f"{'[desde cache] ' if from_cache else ''}"
                  f"in {ie_result.get('execution_time_seconds', '?')}s")
        else:
            print("   [IE] Skipped — ANTHROPIC_API_KEY no configurado "
                  "(agregar como secret en GitHub Actions → Settings → Secrets)")
    except Exception as _e:
        print(f"   [WARN] Intelligence Engine: {_e}")

    # ── 21. IE Accountability — verificar veredictos maduros ─────
    try:
        from src.intel.ie_accountability import evaluate_verdicts, get_verdict_history
        n_verified = evaluate_verdicts()
        if n_verified:
            print(f"   [IE-Accountability] Verified {n_verified} past verdicts")
        hist = get_verdict_history()
        if hist.get("direction_accuracy_7d") is not None:
            print(f"   [IE-Accountability] Direction accuracy 7d: {hist['direction_accuracy_7d']}% "
                  f"({hist['verified_7d']} verified)")
    except Exception as _e:
        print(f"   [INFO] IE Accountability: {_e}")

    print("✅ Pipeline completo")


if __name__ == "__main__":
    run_pipeline()
