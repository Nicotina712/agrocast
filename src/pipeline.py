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

    # ── 6. Limpieza final ─────────────────────────────────────────
    features = features.ffill().bfill().fillna(0)

    # ── 6. Guardar features ───────────────────────────────────────
    features_path = os.path.join(DATA_DIR, "features.csv")
    features.to_csv(features_path, index=False)
    print(f"💾 Features guardadas en {features_path}")

    # ── 7. Modelo de precios ──────────────────────────────────────
    train_model(features, target="Soybeans", artifacts_dir=ARTIFACTS_DIR)

    # ── 8. Modelo de retornos ─────────────────────────────────────
    # El modelo de retornos siempre se reentrena: es rápido (~5s) y así
    # los cambios de hiperparámetros y features se aplican en cada run.
    returns_model_path = os.path.join(MODEL_ARTIFACTS_DIR, "returns_model.joblib")
    needs_train = True

    if needs_train:
        train_returns_model(features_path, MODEL_ARTIFACTS_DIR)
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

    # ── 9a. Forecast mensual robusto (ETS/Holt-Winters) ───────────
    try:
        from src.model.monthly_robust import build_monthly_forecast
        build_monthly_forecast(horizon_days=90)
    except Exception as _e:
        print(f"   [INFO] monthly_robust: {_e}")

    # ── 9b. Snapshot de accountability ───────────────────────────
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

    # ── 10. Señales ───────────────────────────────────────────────
    signals = predict_returns(features, artifacts_dir=MODEL_ARTIFACTS_DIR)

    signals_path = os.path.join(ARTIFACTS_DIR, "signals.csv")
    signals.to_csv(signals_path, index=False)
    print(f"💾 Señales guardadas en {signals_path}")

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

    # ── 14. Señal Argentina ampliada (cepo + retenciones + CIARA) ──
    try:
        from src.data.fetch_argentina import get_argentina_supply_signal
        arg = get_argentina_supply_signal()
        print(f"   [AR] Score: {arg.get('supply_score', '?')} | "
              f"Cepo: {'activo' if arg.get('cepo_activo') else 'inactivo'} | "
              f"Impacto: {arg.get('impacto_precio', '?')}")
    except Exception as _e:
        print(f"   [INFO] Argentina signal: {_e}")

    # ── 15. Basis Uruguay (caché 24h) ─────────────────────────────
    try:
        from src.data.fetch_basis_uruguay import get_basis_uruguay
        get_basis_uruguay()
    except Exception as _e:
        print(f"   [INFO] Basis Uruguay: {_e}")

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

    print("✅ Pipeline completo")


if __name__ == "__main__":
    run_pipeline()
