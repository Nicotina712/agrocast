"""
news_server.py - AgroCast PRO
"""

import json
import os
import sys
import subprocess
import threading
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import hashlib
import secrets

import pandas as pd
from flask import Flask, jsonify, request

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        print("[OK] .env cargado")
except ImportError:
    pass

from multi_news_fetcher import MultiNewsFetcher

app     = Flask(__name__)
fetcher = MultiNewsFetcher()

# ── JSON provider seguro contra NaN/Inf ───────────────────────────────
# Python permite NaN/Inf en json.dumps por default, pero JSON.parse de JS
# los rechaza (Unexpected token 'N'). Limpiamos a None en todo el árbol
# de respuesta antes de serializar.
import math as _math
from flask.json.provider import DefaultJSONProvider as _DefaultJSONProvider

def _clean_json_nans(o):
    if isinstance(o, float):
        return None if (_math.isnan(o) or _math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _clean_json_nans(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json_nans(v) for v in o]
    return o

class _NaNSafeJSONProvider(_DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        return super().dumps(_clean_json_nans(obj), **kwargs)

app.json = _NaNSafeJSONProvider(app)

# ── Favicon stub para silenciar 404 en logs ──────────────────────────
@app.route("/favicon.ico")
def _favicon():
    # Devolvemos 204 No Content en lugar de un 404. Así el navegador deja
    # de pedirlo cada navegación y los logs quedan limpios.
    return ("", 204)

# ── API Key auth ──────────────────────────────────────────────────────
# Configura AGROCAST_API_KEY en .env; si no está, se genera una al arrancar.
_API_KEY_ENV = os.getenv("AGROCAST_API_KEY", "").strip()
if not _API_KEY_ENV:
    _API_KEY_ENV = secrets.token_urlsafe(32)
    print(f"[API] AGROCAST_API_KEY no configurada — usando clave temporal: {_API_KEY_ENV}")

def _check_api_key() -> bool:
    """Verifica Bearer token o query param ?api_key=..."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = request.args.get("api_key", "")
    return secrets.compare_digest(token, _API_KEY_ENV)

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT      = os.path.dirname(BASE_DIR)
ARTIFACTS_DIR     = os.path.join(PROJECT_ROOT, "artifacts")
PIPELINE_PATH     = os.path.join(PROJECT_ROOT, "src", "pipeline.py")
ALERT_HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "alert_history.json")
MODEL_ARTIFACTS   = os.path.join(PROJECT_ROOT, "src", "model", "artifacts")

_cache_lock       = threading.Lock()
_last_update      = 0.0
_cache            = {}
CACHE_TTL         = 300
PIPELINE_TIMEOUT  = 600

_pipeline_lock    = threading.Lock()
_pipeline_running = False

_backtest_lock    = threading.Lock()
_backtest_cache   = {}
_backtest_ts      = 0.0
BACKTEST_TTL      = 3600  # 1 hora

_contract_lock    = threading.Lock()
_contract_cache   = {}
_contract_ts      = 0.0
CONTRACT_TTL      = 3600  # 1 hora
CONTRACT_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "current_contract.json")

_ni_lock          = threading.Lock()
_ni_running       = False
NEWS_IMPACT_DIR   = os.path.join(PROJECT_ROOT, "artifacts", "news_impact")
NI_PIPELINE_PATH  = os.path.join(PROJECT_ROOT, "src", "pipeline_news_impact.py")


# ── Pipeline ──────────────────────────────────────────────────────

def _run_pipeline_process():
    try:
        print(f"[>>] Iniciando pipeline: {PIPELINE_PATH}")
        result = subprocess.run(
            [sys.executable, PIPELINE_PATH],
            text=True,
            timeout=PIPELINE_TIMEOUT,
            encoding="utf-8",
            errors="replace",
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            print(f"[WARN] Pipeline termino con codigo {result.returncode}")
            return False
        print("[OK] Pipeline finalizado correctamente")
        return True
    except subprocess.TimeoutExpired:
        print(f"[WARN] Pipeline timeout ({PIPELINE_TIMEOUT}s)")
        return False
    except Exception as e:
        print(f"[ERROR] Pipeline: {e}")
        return False


def run_pipeline_blocking():
    global _pipeline_running
    with _pipeline_lock:
        if _pipeline_running:
            return
        _pipeline_running = True
    try:
        ok = _run_pipeline_process()
        print("[OK] Pipeline completo" if ok else "[WARN] Pipeline con errores")
    finally:
        with _pipeline_lock:
            _pipeline_running = False


def run_pipeline_background():
    global _pipeline_running
    with _pipeline_lock:
        if _pipeline_running:
            print("[>>] Pipeline ya en ejecucion, se omite")
            return
        _pipeline_running = True

    def _worker():
        global _pipeline_running
        try:
            ok = _run_pipeline_process()
            if ok:
                _check_signal_change()
            else:
                print("[WARN] Pipeline background termino con errores")
        finally:
            with _pipeline_lock:
                _pipeline_running = False

    threading.Thread(target=_worker, daemon=True).start()


# ── Alertas automáticas ───────────────────────────────────────────

def _load_alert_history() -> list:
    if not os.path.exists(ALERT_HISTORY_PATH):
        return []
    try:
        with open(ALERT_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_alert_history(history: list) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(ALERT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history[-50:], f, ensure_ascii=False)


def _check_signal_change() -> None:
    """Detecta cambio de señal del modelo y registra alerta persistente."""
    current = load_signals()
    if not current:
        return

    history = _load_alert_history()
    prev_signal = history[-1]["signal"] if history else None

    if current["signal"] != prev_signal:
        alert = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M", time.localtime()),
            "signal":    current["signal"],
            "prev":      prev_signal or "—",
            "return":    round(current["expected_return"] * 100, 2),
            "confidence": round(current["confidence"] * 100, 1),
        }
        history.append(alert)
        _save_alert_history(history)
        print(f"[ALERTA] Señal cambió: {prev_signal} → {current['signal']}")


# ── Loaders ───────────────────────────────────────────────────────

def load_seasonality() -> dict:
    """
    Calcula el patrón estacional histórico de soja:
    - Retorno promedio y mediana por mes calendario
    - Fracción de meses positivos (win rate estacional)
    Usa todo el historial disponible en raw_market.csv.
    """
    path = os.path.join(PROJECT_ROOT, "data", "raw_market.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").dropna(subset=["Soybeans"])

        # Retorno mensual
        df["month"]        = df["Date"].dt.month
        df["year"]         = df["Date"].dt.year
        df["ret_1m"]       = df["Soybeans"].pct_change(21)  # ~21 días hábiles

        month_names = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        months = []
        for m in range(1, 13):
            subset = df[df["month"] == m]["ret_1m"].dropna()
            if subset.empty:
                months.append({"month": m, "name": month_names[m-1], "avg_pct": 0, "median_pct": 0, "win_rate": 50, "std_pct": 0, "ci_upper": 0, "ci_lower": 0, "n": 0})
                continue
            avg    = round(float(subset.mean()) * 100, 2)
            median = round(float(subset.median()) * 100, 2)
            wr     = round(float((subset > 0).mean()) * 100, 1)
            std    = round(float(subset.std()) * 100, 2) if len(subset) > 1 else 0
            months.append({
                "month": m, "name": month_names[m-1],
                "avg_pct": avg, "median_pct": median,
                "win_rate": wr, "std_pct": std,
                "ci_upper": round(avg + std, 2),
                "ci_lower": round(avg - std, 2),
                "n": int(len(subset)),
            })

        # Mejor y peor mes
        best  = max(months, key=lambda x: x["avg_pct"])
        worst = min(months, key=lambda x: x["avg_pct"])

        # Período cubierto
        years_covered = int(df["year"].max() - df["year"].min() + 1)

        return {
            "months":        months,
            "best_month":    best,
            "worst_month":   worst,
            "years_covered": years_covered,
            "data_from":     str(df["Date"].min())[:7],
            "data_to":       str(df["Date"].max())[:7],
        }
    except Exception as e:
        print(f"[WARN] seasonality: {e}")
        return {}


def load_history(days: int = 90) -> list:
    path = os.path.join(PROJECT_ROOT, "data", "raw_market.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").tail(days)
        return [
            {"Date": str(row["Date"])[:10], "Soybeans": round(float(row["Soybeans"]), 2)}
            for _, row in df.iterrows()
            if pd.notna(row.get("Soybeans"))
        ]
    except Exception as e:
        print(f"[WARN] history: {e}")
        return []


def load_forecast() -> list:
    path = os.path.join(ARTIFACTS_DIR, "forecast.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        result = []
        has_bands = "upper" in df.columns and "lower" in df.columns
        for _, row in df.iterrows():
            try:
                entry = {"Date": str(row["Date"])[:10], "Soybeans": float(row["Soybeans"])}
                if has_bands:
                    entry["upper"] = float(row["upper"])
                    entry["lower"] = float(row["lower"])
                result.append(entry)
            except (ValueError, KeyError):
                continue
        return result
    except Exception as e:
        print(f"[WARN] forecast: {e}")
        return []


def load_signals() -> dict | None:
    path = os.path.join(ARTIFACTS_DIR, "signals.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        last = df.iloc[-1]
        # expected_vol: vol anualizada esperada en los próximos 14d (head opcional)
        exp_vol = float(last["expected_vol"]) if "expected_vol" in df.columns and pd.notna(last.get("expected_vol")) else None
        return {
            "date":            str(last.get("Date", ""))[:10],
            "expected_return": float(last["expected_return"]),
            "signal":          str(last["signal"]),
            "confidence":      float(last["confidence"]),
            "expected_vol":    exp_vol,
            "model_type":      "classifier",
        }
    except Exception as e:
        print(f"[WARN] signals: {e}")
        return None


def _live_front_price() -> float | None:
    """Lee el precio live del front-month CME (current_contract.json) si existe.
    Más fiable que el cierre histórico cuando hubo rollover entre el último cierre y hoy."""
    try:
        if os.path.exists(CONTRACT_CACHE_PATH):
            with open(CONTRACT_CACHE_PATH, "r") as f:
                ct = json.load(f)
            if ct.get("ok") and ct.get("price"):
                return float(ct["price"])
    except Exception:
        pass
    return None


def load_14d_forecast() -> dict | None:
    """Retorna el precio objetivo a 14 días del modelo de forecast.
    El %return se calcula contra el precio LIVE del front-month CME cuando está disponible
    (evita comparación apples-to-oranges entre forecast en serie continua y precio actual,
    sobre todo en días de rollover donde la serie spliced cambia de contrato)."""
    path = os.path.join(ARTIFACTS_DIR, "forecast.csv")
    if not os.path.exists(path):
        return None
    try:
        df  = pd.read_csv(path)
        if len(df) < 14:
            return None
        row = df.iloc[13]
        live = _live_front_price()
        hist = load_history(1)
        hist_close = hist[-1]["Soybeans"] if hist else None
        current = live if live is not None else hist_close
        price_14d = round(float(row["Soybeans"]), 2)
        result = {"date": str(row["Date"])[:10], "price": price_14d}
        if current:
            result["return_pct"] = round((price_14d - current) / current * 100, 2)
            result["base_price"] = round(float(current), 2)
            result["base_source"] = "live_cme" if live is not None else "hist_close"
        if "upper" in df.columns:
            result["upper"] = round(float(row["upper"]), 2)
            result["lower"] = round(float(row["lower"]), 2)
        return result
    except Exception as e:
        print(f"[WARN] 14d forecast: {e}")
        return None


def compute_signal_accuracy() -> dict | None:
    """
    Calcula la precisión histórica de señales vs retornos reales.

    Metodología: igual que el backtest — muestrea cada 14 días (sin solapamiento)
    para contar operaciones reales, no días consecutivos con la misma señal.
    Ventana: últimos 90 días de historia con ret_14d_fwd confirmado.
    """
    signals_path  = os.path.join(ARTIFACTS_DIR, "signals.csv")
    features_path = os.path.join(PROJECT_ROOT, "data", "features.csv")
    if not os.path.exists(signals_path) or not os.path.exists(features_path):
        return None
    try:
        sig_df  = pd.read_csv(signals_path,  parse_dates=["Date"])
        feat_df = pd.read_csv(features_path, parse_dates=["Date"])
        ret_col = "ret_14d_fwd" if "ret_14d_fwd" in feat_df.columns else "ret_7d_fwd"
        merged  = sig_df.merge(feat_df[["Date", ret_col]], on="Date", how="left")

        # Excluir últimos 14 días: ret_14d_fwd no está confirmado aún
        confirmed = merged.dropna(subset=[ret_col])

        # Tomar los últimos 90 días con retorno confirmado
        window = confirmed.tail(90).reset_index(drop=True)

        # Muestrear cada 14 días (sin solapamiento) — igual que backtest
        sampled = window.iloc[::14].copy()

        active = sampled[sampled["signal"] != "HOLD"].copy()
        if active.empty:
            return {
                "accuracy": None, "n_signals": 0, "n_correct": 0,
                "n_days": len(sampled), "buy_signals": 0, "sell_signals": 0,
                "note": "Sin señales activas en el período",
            }

        active["correct"] = (
            ((active["signal"] == "BUY")  & (active[ret_col] > 0)) |
            ((active["signal"] == "SELL") & (active[ret_col] < 0))
        )
        n_correct = int(active["correct"].sum())
        n_signals = len(active)
        p = n_correct / n_signals
        # Wilson confidence interval 95% (más robusto que normal para n pequeño)
        import math
        z = 1.96
        denom  = 1 + z**2 / n_signals
        center = (p + z**2 / (2 * n_signals)) / denom
        margin = z * math.sqrt(p * (1 - p) / n_signals + z**2 / (4 * n_signals**2)) / denom
        ci_low  = round(max(0, center - margin) * 100, 1)
        ci_high = round(min(1, center + margin) * 100, 1)
        return {
            "accuracy":     round(p * 100, 1),
            "ci_low":       ci_low,
            "ci_high":      ci_high,
            "n_correct":    n_correct,
            "n_signals":    n_signals,
            "n_days":       len(sampled),
            "buy_signals":  int((active["signal"] == "BUY").sum()),
            "sell_signals": int((active["signal"] == "SELL").sum()),
            "note":         "Muestreo cada 14 días sin solapamiento (como el backtest)",
        }
    except Exception as e:
        print(f"[WARN] accuracy: {e}")
        return None


def load_wasde_upcoming() -> list:
    """Calcula las próximas 4 fechas WASDE (segundo martes de cada mes)."""
    from datetime import date, timedelta

    def second_tuesday(year: int, month: int) -> date:
        d = date(year, month, 1)
        while d.weekday() != 1:
            d += timedelta(days=1)
        return d + timedelta(weeks=1)

    today  = date.today()
    year, month = today.year, today.month
    dates  = []
    for _ in range(8):
        dt = second_tuesday(year, month)
        if dt >= today:
            days_to = (dt - today).days
            dates.append({"date": str(dt), "days_to": days_to})
            if len(dates) == 4:
                break
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return dates


def generate_executive_summary(data: dict) -> list:
    """Genera 3 bullets en lenguaje llano explicando la predicción actual."""
    bullets = []
    model_sig    = data.get("model_signal")
    market       = data.get("market", {})
    price_target = data.get("price_target_14d")

    # Bullet 1: señal del clasificador — expresar como probabilidad, no retorno
    if model_sig:
        sig        = model_sig["signal"]
        # expected_return = P(up) - 0.5  →  P(up) = 0.5 + expected_return
        prob_up    = round((0.5 + model_sig["expected_return"]) * 100)
        prob_down  = 100 - prob_up
        conf       = round(model_sig["confidence"] * 100)

        if sig == "BUY":
            bullets.append(
                f"El clasificador estima una probabilidad de subida del {prob_up}% "
                f"en 7 días (confianza {conf}%). Los indicadores de momentum, "
                f"régimen de mercado y correlación de commodities muestran presión alcista."
            )
        elif sig == "SELL":
            bullets.append(
                f"El clasificador estima una probabilidad de subida de solo el {prob_up}% "
                f"en 7 días (confianza {conf}%). La presión bajista domina — "
                f"momentum, RSI y régimen de mercado apuntan hacia abajo."
            )
        else:
            bullets.append(
                f"El clasificador no detecta dirección clara (P(sube)={prob_up}%, "
                f"confianza {conf}%). Indicadores mixtos — se recomienda "
                f"mantener posición y esperar confirmación."
            )

    # Bullet 2: sentimiento de noticias
    sent = market.get("sentiment", 0)
    vol  = market.get("volume", 0)
    if vol > 0:
        if sent > 0.05:
            bullets.append(
                f"El flujo noticioso es alcista (score {sent:.2f}, {vol} artículos). "
                f"La cobertura mediática refuerza las expectativas positivas."
            )
        elif sent < -0.05:
            bullets.append(
                f"El flujo noticioso es bajista (score {sent:.2f}, {vol} artículos). "
                f"Los medios reportan factores de presión sobre el precio."
            )
        else:
            bullets.append(
                f"El sentimiento noticioso es neutro (score {sent:.2f}, {vol} artículos). "
                f"No hay señal de mercado clara desde la prensa especializada."
            )

    # Bullet 3: precio objetivo — con nota si diverge del clasificador
    if price_target:
        ret_7d  = price_target.get("return_pct", 0)
        price   = price_target.get("price", 0)
        upper   = price_target.get("upper")
        lower   = price_target.get("lower")
        sign    = "+" if ret_7d >= 0 else ""
        sig_dir = model_sig["signal"] if model_sig else "HOLD"

        # Divergencia real: solo cuando el modelo de precios espera un movimiento
        # fuerte (>2%) en dirección opuesta al clasificador. Un umbral bajo
        # genera falsos positivos porque ambos modelos tienen objetivos distintos.
        diverge = (ret_7d > 2.0 and sig_dir == "SELL") or (ret_7d < -2.0 and sig_dir == "BUY")

        if upper and lower:
            base = (
                f"El modelo de precios proyecta USD {price:.2f}/bu "
                f"({sign}{ret_7d:.1f}% en 7 días), rango USD {lower:.2f}–{upper:.2f}."
            )
            if diverge:
                base += (
                    f" Nota: el modelo de precios (tendencia) y el clasificador "
                    f"(dirección corta) divergen — señal mixta."
                )
            bullets.append(base)
        else:
            bullets.append(
                f"El modelo de precios proyecta USD {price:.2f}/bu a 7 días "
                f"({sign}{ret_7d:.1f}%)."
            )
    elif forecast:
        hist = load_history(1)
        if hist and forecast:
            curr   = hist[-1]["Soybeans"]
            fc_30  = forecast[-1]["Soybeans"]
            ret_30 = (fc_30 - curr) / curr * 100
            sign   = "+" if ret_30 >= 0 else ""
            bullets.append(
                f"A 30 días el modelo proyecta USD {fc_30:.2f}/bu "
                f"({sign}{ret_30:.1f}% vs precio actual)."
            )

    return bullets


def get_current_contract_data() -> dict:
    """
    Descarga los precios del contrato front-month y del siguiente contrato de soja
    para calcular el spread de rollover.

    Tickers CBOT:
      ZSK26.CBT — Soja Mayo 2026
      ZSN26.CBT — Soja Julio 2026

    Si no puede descargarlos, retorna lo que pueda con los campos disponibles.
    Guarda el resultado en data/current_contract.json.
    """
    # Mapa de mes → código de contrato
    _MONTH_CODES = {
        1: "F", 2: "G", 3: "H", 4: "J", 5: "K",
        6: "M", 7: "N", 8: "Q", 9: "U", 10: "V",
        11: "X", 12: "Z",
    }
    _MONTH_NAMES = {
        "F": "ENE", "G": "FEB", "H": "MAR", "J": "ABR", "K": "MAY",
        "M": "JUN", "N": "JUL", "Q": "AGO", "U": "SEP", "V": "OCT",
        "X": "NOV", "Z": "DIC",
    }
    # Contratos activos de soja CBOT (front-month más líquidos):
    # ENE(F), MAR(H), MAY(K), JUL(N), AGO(Q), SEP(U), NOV(X)
    _ACTIVE_MONTHS = [1, 3, 5, 7, 8, 9, 11]

    from datetime import date
    import json as _json

    try:
        import yfinance as _yf
    except ImportError:
        return {"ok": False, "error": "yfinance no disponible"}

    today = date.today()
    year  = today.year
    month = today.month

    # Encontrar front-month y siguiente contrato activo
    def next_active(y, m):
        """Retorna (year, month) del próximo mes activo de soja desde (y, m) inclusive."""
        for _ in range(24):
            if m in _ACTIVE_MONTHS:
                return y, m
            m += 1
            if m > 12:
                m, y = 1, y + 1
        return y, m

    def business_days_until(target: date) -> int:
        """Días hábiles (lun-vie) desde hoy hasta target."""
        from datetime import timedelta
        delta, count = (target - today).days, 0
        for i in range(delta):
            if (today + timedelta(days=i + 1)).weekday() < 5:
                count += 1
        return count

    def first_notice_day(contract_year: int, contract_month: int) -> date:
        """FND de futuros CBOT = último día hábil del mes anterior al contrato."""
        import calendar
        from datetime import timedelta
        if contract_month == 1:
            fnd_year, fnd_month = contract_year - 1, 12
        else:
            fnd_year, fnd_month = contract_year, contract_month - 1
        last_day = calendar.monthrange(fnd_year, fnd_month)[1]
        d = date(fnd_year, fnd_month, last_day)
        # Retroceder al último día hábil si cae en finde
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d

    ROLL_DAYS_BEFORE_FND = 5  # Rotar cuando quedan ≤5 días hábiles para FND

    front_y, front_m = next_active(year, month)
    fnd = first_notice_day(front_y, front_m)
    # Si estamos demasiado cerca del FND, ya operamos el contrato siguiente
    if business_days_until(fnd) <= ROLL_DAYS_BEFORE_FND:
        next_start_m = front_m + 1
        next_start_y = front_y
        if next_start_m > 12:
            next_start_m, next_start_y = 1, front_y + 1
        front_y, front_m = next_active(next_start_y, next_start_m)

    # Siguiente contrato después del front-month
    next_start_m = front_m + 1
    next_start_y = front_y
    if next_start_m > 12:
        next_start_m, next_start_y = 1, front_y + 1
    next_y, next_m = next_active(next_start_y, next_start_m)

    def make_ticker(y, m):
        code = _MONTH_CODES[m]
        short_y = str(y)[-2:]
        return f"ZS{code}{short_y}.CBT"

    def contract_label(y, m):
        code = _MONTH_CODES[m]
        return f"{_MONTH_NAMES[code]}{y}"

    front_ticker = make_ticker(front_y, front_m)
    next_ticker  = make_ticker(next_y,  next_m)
    front_label  = contract_label(front_y, front_m)
    next_label   = contract_label(next_y,  next_m)

    def fetch_last_price(ticker: str):
        try:
            raw = _yf.download(ticker, period="5d", interval="1d", progress=False)
            if raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            return float(raw["Close"].dropna().iloc[-1])
        except Exception as e:
            print(f"[WARN] current_contract: no se pudo descargar {ticker}: {e}")
            return None

    front_price = fetch_last_price(front_ticker)
    next_price  = fetch_last_price(next_ticker)

    spread = None
    if front_price is not None and next_price is not None:
        spread = round(next_price - front_price, 2)

    front_fnd = first_notice_day(front_y, front_m)
    days_to_fnd = business_days_until(front_fnd)

    result = {
        "ok":             True,
        "ticker":         "ZS=F",
        "front_contract": front_label,
        "front_ticker":   front_ticker,
        "price":          round(front_price, 2) if front_price is not None else None,
        "next_contract":  next_label,
        "next_ticker":    next_ticker,
        "next_price":     round(next_price, 2) if next_price is not None else None,
        "spread_usc":     spread,
        "days_to_fnd":    days_to_fnd,
        "fnd_date":       front_fnd.isoformat(),
        "generated_at":   pd.Timestamp.now().isoformat(timespec="seconds"),
    }

    # Guardar en disco
    try:
        os.makedirs(os.path.dirname(CONTRACT_CACHE_PATH), exist_ok=True)
        with open(CONTRACT_CACHE_PATH, "w", encoding="utf-8") as f:
            _json.dump(result, f, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] No se pudo guardar current_contract.json: {e}")

    return result


# ── Endpoints ─────────────────────────────────────────────────────

@app.route("/api/news")
def get_news():
    global _last_update, _cache
    now = time.time()

    with _cache_lock:
        if now - _last_update <= CACHE_TTL and _cache:
            return jsonify(_cache)

        print("[>>] Actualizando cache...")
        run_pipeline_background()

        try:
            news_data = fetcher.update_news()
        except Exception as e:
            print(f"[ERROR] Noticias: {e}")
            news_data = {"articles": [], "market": {}, "alerts": [], "forecast": []}

        articles   = news_data.get("articles", [])
        sentiments = [a.get("sentiment", 0) for a in articles if a.get("sentiment") is not None]

        if sentiments:
            top       = sorted(sentiments, key=abs, reverse=True)[:5]
            sentiment = sum(s * abs(s) for s in top) / len(top)
        else:
            sentiment = 0.0

        # Volumen real = batch fresco de hoy (no la memoria acumulada de 50)
        fresh_count = news_data.get("fresh_count", len(articles))
        raw_count   = news_data.get("raw_count", 0)
        market = {
            "sentiment":     round(sentiment, 3),
            "volume":        fresh_count,
            "raw_volume":    raw_count,
            "memory_size":   len(articles),
        }

        # Persistir sentimiento diario para el módulo news-impact
        try:
            sys.path.insert(0, PROJECT_ROOT)
            from src.features.news_history import save_daily_sentiment
            china_s   = float(sum(a.get("sentiment", 0) for a in articles if a.get("topic") == "china")   / max(1, sum(1 for a in articles if a.get("topic") == "china")))
            weather_s = float(sum(a.get("sentiment", 0) for a in articles if a.get("topic") == "weather") / max(1, sum(1 for a in articles if a.get("topic") == "weather")))
            macro_s   = float(sum(a.get("sentiment", 0) for a in articles if a.get("topic") == "macro")   / max(1, sum(1 for a in articles if a.get("topic") == "macro")))
            save_daily_sentiment(sentiment, fresh_count, china_s, weather_s, macro_s)
        except Exception as _e:
            print(f"[WARN] No se pudo guardar sentimiento: {_e}")

        if sentiment < -0.05:
            news_signal = {"signal": "BAJISTA", "reason": "Noticias negativas"}
        elif sentiment > 0.05:
            news_signal = {"signal": "ALCISTA", "reason": "Noticias positivas"}
        else:
            news_signal = {"signal": "NEUTRAL", "reason": "Mercado sin direccion"}

        model_signal = load_signals()
        forecast     = load_forecast()
        price_14d    = load_14d_forecast()

        partial_data = {
            "articles":       articles[:10],
            "market":         market,
            "history":        load_history(90),
            "forecast":       forecast,
            "signal":         news_signal,
            "model_signal":   model_signal,
            "alerts":         news_data.get("alerts", []),
            "alert_history":  _load_alert_history()[-10:],
            "price_target_14d": price_14d,
            "wasde_upcoming": load_wasde_upcoming(),
            "accuracy":       compute_signal_accuracy(),
        }
        partial_data["executive_summary"] = generate_executive_summary(partial_data)
        _cache = partial_data
        _last_update = now

    return jsonify(_cache)


@app.route("/api/backtest")
def get_backtest():
    global _backtest_cache, _backtest_ts
    now = time.time()

    with _backtest_lock:
        if now - _backtest_ts <= BACKTEST_TTL and _backtest_cache:
            return jsonify(_backtest_cache)

        try:
            sys.path.insert(0, PROJECT_ROOT)
            from src.model.backtest import run_backtest
            features_path = os.path.join(PROJECT_ROOT, "data", "features.csv")
            result = run_backtest(features_path, MODEL_ARTIFACTS)
            _backtest_cache = {"ok": True,  **result}
        except Exception as e:
            print(f"[ERROR] Backtest: {e}")
            _backtest_cache = {"ok": False, "error": str(e)}

        _backtest_ts = now

    try:
        return jsonify(_backtest_cache)
    except Exception as e:
        print(f"[ERROR] jsonify backtest: {e}")
        return jsonify({"ok": False, "error": f"Error de serialización: {e}"}), 500


@app.route("/api/seasonality")
def get_seasonality():
    """Panel estacional histórico — público (sin API key)."""
    return jsonify(load_seasonality())


@app.route("/api/current_contract")
def api_current_contract():
    """
    GET /api/current_contract
    Retorna el contrato front-month activo de soja CBOT y el spread vs el siguiente.

    Respuesta:
    {
      "ok": true,
      "ticker": "ZS=F",
      "front_contract": "MAY26",
      "front_ticker": "ZSK26.CBT",
      "price": 1166.5,
      "next_contract": "JUL26",
      "next_ticker": "ZSN26.CBT",
      "next_price": 1182.5,
      "spread_usc": 16.0,
      "generated_at": "2026-04-20T14:30:00"
    }
    Cache: 1 hora.
    """
    import json as _json
    global _contract_cache, _contract_ts
    now = time.time()

    with _contract_lock:
        # Servir desde caché en memoria si es fresco
        if now - _contract_ts <= CONTRACT_TTL and _contract_cache:
            return jsonify(_contract_cache)

        # Intentar leer desde archivo en disco (sobrevive reinicios)
        if os.path.exists(CONTRACT_CACHE_PATH):
            try:
                mtime = os.path.getmtime(CONTRACT_CACHE_PATH)
                if now - mtime <= CONTRACT_TTL:
                    with open(CONTRACT_CACHE_PATH, encoding="utf-8") as f:
                        _contract_cache = _json.load(f)
                    _contract_ts = mtime
                    return jsonify(_contract_cache)
            except Exception:
                pass

        # Descargar datos frescos
        try:
            data = get_current_contract_data()
        except Exception as e:
            print(f"[ERROR] /api/current_contract: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

        _contract_cache = data
        _contract_ts    = now

    return jsonify(_contract_cache)


@app.route("/api/paper_trades")
def get_paper_trades():
    """
    GET /api/paper_trades?capital=10000&risk_pct=1
    Retorna portfolio completo de paper trading:
    open_trades, closed_trades, summary, equity_curve.
    """
    try:
        sys.path.insert(0, PROJECT_ROOT)
        capital  = float(request.args.get("capital",  10000))
        risk_pct = float(request.args.get("risk_pct", 1.0))
        from src.trader.paper_trading import get_paper_portfolio
        data = get_paper_portfolio(capital=capital)
        data["ok"] = True
        return jsonify(data)
    except Exception as e:
        print(f"[ERROR] /api/paper_trades: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/trader")
def get_trader():
    """
    GET /api/trader
    Módulo Trader — niveles de riesgo (SL/TP basados en ATR) y
    curva de futuros ZS (term structure).
    """
    try:
        sys.path.insert(0, PROJECT_ROOT)
        capital  = float(request.args.get("capital", 10000))
        risk_pct = float(request.args.get("risk_pct", 1.0))

        from src.trader.risk_manager import get_current_risk_levels
        risk = get_current_risk_levels(capital_usd=capital, risk_pct=risk_pct)

        from src.trader.term_structure import fetch_term_structure
        term = fetch_term_structure()

        # COT delta (si hay datos disponibles)
        cot_data = None
        try:
            cot_path = os.path.join(PROJECT_ROOT, "data", "cot_soybeans.csv")
            if os.path.exists(cot_path):
                cdf = pd.read_csv(cot_path, parse_dates=["Date"]).sort_values("Date")
                if "cot_noncomm_net" in cdf.columns and len(cdf) >= 2:
                    cur  = float(cdf["cot_noncomm_net"].iloc[-1])
                    prev = float(cdf["cot_noncomm_net"].iloc[-8])  # ~1 semana atrás
                    cot_data = {
                        "current_net": round(cur, 0),
                        "delta_1w":    round(cur - prev, 0),
                        "cot_index":   round(float(cdf["cot_index"].iloc[-1]), 1) if "cot_index" in cdf.columns else None,
                    }
        except Exception:
            pass

        return jsonify({"ok": True, "risk_levels": risk, "term_structure": term, "cot_delta": cot_data})
    except Exception as e:
        print(f"[ERROR] /api/trader: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/producer")
def get_producer():
    """
    GET /api/producer
    Módulo Productor — semáforo de venta, precio USD/ton + UYU/ton,
    costo de almacenamiento y contexto para productores de soja en Uruguay.
    """
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.producer.sell_signal import compute_sell_signal

        # Precio actual
        price_usd_bu = None
        try:
            mkt = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw_market.csv"))
            price_usd_bu = float(mkt["Soybeans"].iloc[-1])
        except Exception:
            pass

        if price_usd_bu is None:
            return jsonify({"ok": False, "error": "Sin datos de precio"}), 503

        model_signal  = load_signals()
        price_history = load_history(90)
        wasde_dates   = load_wasde_upcoming()
        seasonality   = load_seasonality().get("months")

        result = compute_sell_signal(
            current_price_usd_bu=price_usd_bu,
            model_signal=model_signal,
            price_history=price_history,
            wasde_dates=wasde_dates,
            seasonality_data=seasonality,
        )

        # Costos de puerto Uruguay (actualizables en .env)
        result["port_costs"] = {
            "nueva_palmira_usd_mt": float(os.getenv("PORT_NUEVA_PALMIRA_USD_MT", "24")),
            "montevideo_usd_mt":    float(os.getenv("PORT_MONTEVIDEO_USD_MT",    "26")),
            "nota": "Costo estimado — actualizar en .env si hay cambios",
        }

        # Señal retención Argentina (caché 12h)
        try:
            from src.data.fetch_argentina import get_argentina_supply_signal
            result["argentina_signal"] = get_argentina_supply_signal()
        except Exception:
            result["argentina_signal"] = None

        result["ok"] = True
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] /api/producer: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/forecast_multihorizon")
def get_forecast_multihorizon():
    """
    GET /api/forecast_multihorizon
    Forecast multi-horizonte: 7d (traders), 30d (base), 90d (productores/forward).
    """
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.model.predict_multihorizon import get_multihorizon_forecast
        data = get_multihorizon_forecast()
        return jsonify(data)
    except Exception as e:
        print(f"[ERROR] /api/forecast_multihorizon: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/argentina_signal")
def get_argentina_signal():
    """
    GET /api/argentina_signal
    Señal de retención de soja argentina (cepo cambiario, brecha, impacto en precio).
    Cache: 12h.
    """
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.data.fetch_argentina import get_argentina_supply_signal
        data = get_argentina_supply_signal()
        data["ok"] = True
        return jsonify(data)
    except Exception as e:
        print(f"[ERROR] /api/argentina_signal: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/cot_delta")
def get_cot_delta():
    """
    GET /api/cot_delta
    Delta semanal de posiciones COT (cambio semana a semana en Managed Money net).
    Señala cuando los grandes fondos están aumentando/reduciendo posición.
    """
    try:
        cot_path = os.path.join(PROJECT_ROOT, "data", "cot_soybeans.csv")
        if not os.path.exists(cot_path):
            return jsonify({"ok": False, "error": "COT data no disponible"}), 503

        df = pd.read_csv(cot_path, parse_dates=["Date"])
        df = df.sort_values("Date").tail(60)

        if "cot_noncomm_net" not in df.columns:
            return jsonify({"ok": False, "error": "Columna cot_noncomm_net no encontrada"}), 503

        # Delta semanal (diferencia entre último y penúltimo registro semanal)
        # COT es semanal; tomamos los últimos registros únicos por semana
        df["week"] = df["Date"].dt.isocalendar().week
        df["year"] = df["Date"].dt.year
        weekly = df.drop_duplicates(subset=["year", "week"], keep="last").tail(10)

        net_values = weekly["cot_noncomm_net"].tolist()
        dates_list = [str(d)[:10] for d in weekly["Date"].tolist()]

        current_net = float(weekly["cot_noncomm_net"].iloc[-1])
        prev_net    = float(weekly["cot_noncomm_net"].iloc[-2]) if len(weekly) >= 2 else current_net
        delta_1w    = current_net - prev_net

        # Delta 4 semanas
        delta_4w = current_net - float(weekly["cot_noncomm_net"].iloc[-5]) if len(weekly) >= 5 else 0

        # COT index si disponible
        cot_index = float(weekly["cot_index"].iloc[-1]) if "cot_index" in weekly.columns else None

        # Señal: extremos históricos en ventana de 52 semanas
        net_52w    = df["cot_noncomm_net"].tail(260)   # ~52 semanas de datos diarios
        pct_rank   = float((net_52w < current_net).mean() * 100)

        if pct_rank >= 80:
            extremo = "LARGO_EXTREMO"
            extremo_label = "Fondos muy comprados (potencial reversión bajista)"
        elif pct_rank <= 20:
            extremo = "CORTO_EXTREMO"
            extremo_label = "Fondos muy vendidos (potencial reversión alcista)"
        else:
            extremo = "NEUTRO"
            extremo_label = "Posicionamiento en rango normal"

        # Señal de delta: aumento sostenido = momentum alcista
        if delta_1w > 5000:
            delta_signal = "COMPRANDO"
            delta_label  = "Fondos aumentando posición larga (alcista)"
        elif delta_1w < -5000:
            delta_signal = "VENDIENDO"
            delta_label  = "Fondos reduciendo posición larga (bajista)"
        else:
            delta_signal = "ESTABLE"
            delta_label  = "Sin cambio significativo en posicionamiento"

        return jsonify({
            "ok":            True,
            "current_net":   round(current_net, 0),
            "delta_1w":      round(delta_1w, 0),
            "delta_4w":      round(delta_4w, 0),
            "cot_index":     round(cot_index, 1) if cot_index else None,
            "pct_rank_52w":  round(pct_rank, 1),
            "extremo":       extremo,
            "extremo_label": extremo_label,
            "delta_signal":  delta_signal,
            "delta_label":   delta_label,
            "history_dates": dates_list,
            "history_net":   [round(v, 0) for v in net_values],
        })
    except Exception as e:
        print(f"[ERROR] /api/cot_delta: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/usda_inspections")
def get_usda_inspections():
    """
    GET /api/usda_inspections
    Seguimiento semanal de inspecciones de exportación de soja USDA.
    Compara el ritmo acumulado vs año anterior y vs estimado USDA anual.
    """
    try:
        insp_path = os.path.join(PROJECT_ROOT, "data", "usda_inspections.csv")
        if not os.path.exists(insp_path):
            return jsonify({"ok": False, "error": "USDA inspections data no disponible"}), 503

        df = pd.read_csv(insp_path, parse_dates=["Date"])
        df = df.sort_values("Date")

        # Últimas 12 semanas
        recent = df.tail(12).copy()

        # Acumulado año marketing (Sep-Ago para soja)
        current_year = pd.Timestamp.now().year
        current_month = pd.Timestamp.now().month
        marketing_start = pd.Timestamp(f"{current_year - 1 if current_month < 9 else current_year}-09-01")
        ytd = df[df["Date"] >= marketing_start]

        cumulative_bu = float(ytd["insp_soy_bu"].sum()) if "insp_soy_bu" in ytd.columns else 0
        cumulative_mt = cumulative_bu / 36744.0 if cumulative_bu > 0 else 0

        # Última semana
        last_row = recent.iloc[-1] if not recent.empty else None
        last_week_bu = float(last_row["insp_soy_bu"]) if last_row is not None and "insp_soy_bu" in recent.columns else 0

        # YoY si hay datos
        yoy_change = None
        if "insp_soy_yoy" in df.columns and not df["insp_soy_yoy"].isna().all():
            yoy_change = round(float(df["insp_soy_yoy"].iloc[-1]) * 100, 1)

        # Historial para gráfico
        history = []
        for _, row in recent.iterrows():
            if "insp_soy_bu" in row.index and pd.notna(row["insp_soy_bu"]):
                history.append({
                    "date":       str(row["Date"])[:10],
                    "weekly_bu":  round(float(row["insp_soy_bu"]), 0),
                    "weekly_mt":  round(float(row["insp_soy_bu"]) / 36744.0, 1),
                })

        return jsonify({
            "ok":                  True,
            "cumulative_mt":       round(cumulative_mt, 0),
            "last_week_bu":        round(last_week_bu, 0),
            "last_week_mt":        round(last_week_bu / 36744.0, 1),
            "yoy_change_pct":      yoy_change,
            "marketing_year_from": str(marketing_start)[:10],
            "history":             history,
        })
    except Exception as e:
        print(f"[ERROR] /api/usda_inspections: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/basis_uruguay")
def get_basis_uruguay():
    """GET /api/basis_uruguay — Basis soja Uruguay/River Plate vs. CBOT."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.data.fetch_basis_uruguay import get_basis_uruguay as _get_basis
        data = _get_basis()
        data["ok"] = True
        return jsonify(data)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/wasde_official")
def get_wasde_official():
    """GET /api/wasde_official — WASDE via USDA FAS PSD API (ending stocks, surprise)."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.data.wasde_api import get_wasde_official as _get_wasde
        data = _get_wasde()
        data["ok"] = True
        return jsonify(data)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/brazil_exports")
def get_brazil_exports():
    """GET /api/brazil_exports — Pace de exportaciones de soja de Brasil vs. proyección USDA."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.data.fetch_brazil_exports import get_brazil_export_pace
        data = get_brazil_export_pace()
        data["ok"] = True
        return jsonify(data)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/china_demand")
def get_china_demand():
    """GET /api/china_demand — Módulo de demanda china (importaciones + crush margin + CNY)."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.data.fetch_china_demand import get_china_demand as _get_china
        data = _get_china()
        data["ok"] = True
        return jsonify(data)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/export_brief")
def export_brief():
    """
    GET /export_brief — Página HTML imprimible del estado actual del sistema.
    Abre en nueva pestaña; el usuario usa Ctrl+P para guardar como PDF.
    """
    try:
        sys.path.insert(0, PROJECT_ROOT)
        import json
        from datetime import date

        # Recopilar datos
        def safe_load(path):
            try:
                full = os.path.join(PROJECT_ROOT, "data", path)
                return json.load(open(full)) if os.path.exists(full) else {}
            except Exception:
                return {}

        china   = safe_load("china_demand.json")
        wasde   = safe_load("wasde_official.json")
        basis   = safe_load("basis_uruguay.json")
        brazil  = safe_load("brazil_exports.json")
        mcomm   = safe_load("multi_commodity.json")
        brkdwn  = safe_load("signal_breakdown.json")
        stress  = safe_load("wasde_stress.json")

        # Señal actual
        sig_df = None
        try:
            import pandas as pd
            sig_df = pd.read_csv(os.path.join(PROJECT_ROOT, "artifacts", "signals.csv"))
            mkt    = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw_market.csv"),
                                  parse_dates=["Date"]).sort_values("Date")
            price  = float(mkt["Soybeans"].iloc[-1])
            signal = str(sig_df["signal"].iloc[-1])
            conf   = float(sig_df["confidence"].iloc[-1])
        except Exception:
            price, signal, conf = 0, "N/D", 0

        signal_color = {"BUY": "#1b8a3e", "SELL": "#c62828"}.get(signal, "#888")
        score = brkdwn.get("composite_score", 50)
        score_color = "#1b8a3e" if score >= 60 else "#c62828" if score <= 40 else "#f0a500"

        # Brief guardado más reciente
        saved_brief = ""
        data_dir = os.path.join(PROJECT_ROOT, "data")
        briefs = sorted([f for f in os.listdir(data_dir) if f.startswith("brief_")], reverse=True)
        if briefs:
            try:
                saved_brief = open(os.path.join(data_dir, briefs[0]), encoding="utf-8").read()
            except Exception:
                pass

        def fmt(v, d=2):
            try: return f"{float(v):.{d}f}"
            except: return "—"

        def sign(v):
            try: return f"+{float(v):.1f}" if float(v) >= 0 else f"{float(v):.1f}"
            except: return "—"

        factors_rows = ""
        for f in brkdwn.get("factors", []):
            sc = f.get("score", 0)
            dir_color = "#1b8a3e" if sc > 0.15 else "#c62828" if sc < -0.15 else "#888"
            weight = f"{int(f.get('weight',0)*100)}%" if f.get("weight",0) > 0 else "info"
            factors_rows += f"""<tr>
                <td><b>{f.get('name','')}</b> <small>({weight})</small></td>
                <td style="color:{dir_color};font-weight:700">{f.get('direction','')}</td>
                <td>{f.get('detail','')}</td>
            </tr>"""

        comm_rows = ""
        for k, c in (mcomm.get("commodities") or {}).items():
            sc = {"BUY":"#1b8a3e","SELL":"#c62828"}.get(c.get("signal",""),"#888")
            comm_rows += f"""<tr>
                <td>{c.get('emoji','')} {c.get('label','')}</td>
                <td><b>{fmt(c.get('price'))}</b> {c.get('unit','')}</td>
                <td style="color:{sc};font-weight:700">{c.get('signal','—')}</td>
                <td>RSI: {fmt(c.get('rsi'),1)}</td>
                <td>{sign(c.get('chg_5d_pct'))}% (5d)</td>
            </tr>"""

        saved_section = f"""
            <div class="section">
                <h2>Brief Semanal (IA)</h2>
                <pre style="white-space:pre-wrap;font-family:inherit;font-size:.88rem;line-height:1.7;
                    background:#f8f9fa;padding:16px;border-radius:8px;">{saved_brief}</pre>
            </div>""" if saved_brief else ""

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgroCast PRO — Brief {date.today().isoformat()}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1a1a2e; max-width: 900px; margin: 0 auto; padding: 32px 24px; }}
  h1 {{ font-size: 1.6rem; font-weight: 900; color: #1b6e3a; border-bottom: 3px solid #1b6e3a; padding-bottom: 8px; margin-bottom: 6px; }}
  h2 {{ font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: .5px; color: #555; margin: 24px 0 10px; border-bottom: 1px solid #eee; padding-bottom: 4px; }}
  .subtitle {{ font-size: .85rem; color: #888; margin-bottom: 24px; }}
  .signal-box {{ display: inline-block; padding: 10px 28px; border-radius: 10px; color: #fff;
    background: {signal_color}; font-size: 1.8rem; font-weight: 900; margin: 16px 0; }}
  .score-row {{ display: flex; align-items: center; gap: 16px; margin: 10px 0 20px; flex-wrap: wrap; }}
  .score-num {{ font-size: 2rem; font-weight: 900; color: {score_color}; }}
  .score-bar {{ flex: 1; min-width: 200px; background: #eee; border-radius: 6px; height: 12px; overflow: hidden; }}
  .score-fill {{ width: {score}%; height: 100%; background: {score_color}; border-radius: 6px; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 20px; }}
  .box {{ background: #f8f9fa; border-radius: 8px; padding: 14px; }}
  .box-label {{ font-size: .7rem; text-transform: uppercase; color: #888; margin-bottom: 4px; }}
  .box-val {{ font-size: 1.15rem; font-weight: 700; color: #1a1a2e; }}
  .box-sub {{ font-size: .75rem; color: #999; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
  th {{ background: #f0f0f0; text-align: left; padding: 7px 10px; font-size: .78rem; color: #555; }}
  td {{ padding: 6px 10px; font-size: .83rem; border-bottom: 1px solid #f0f0f0; }}
  .section {{ margin-bottom: 28px; }}
  .footer {{ margin-top: 32px; padding-top: 16px; border-top: 1px solid #eee; font-size: .72rem; color: #aaa; }}
  .no-print {{ margin-bottom: 20px; }}
  @media print {{
    .no-print {{ display: none !important; }}
    body {{ padding: 0; }}
    a {{ color: inherit !important; text-decoration: none !important; }}
  }}
</style>
</head>
<body>

<div class="no-print">
  <button onclick="window.print()" style="background:#1b6e3a;color:#fff;border:none;padding:10px 24px;
    border-radius:8px;font-size:.9rem;font-weight:700;cursor:pointer;margin-right:10px;">
    🖨 Imprimir / Guardar como PDF
  </button>
  <button onclick="window.close()" style="background:#eee;color:#333;border:none;padding:10px 20px;
    border-radius:8px;font-size:.9rem;cursor:pointer;">Cerrar</button>
</div>

<h1>AgroCast PRO — Brief de Mercado</h1>
<div class="subtitle">Generado el {date.today().strftime('%d/%m/%Y')} · Soja CBOT · Datos en USc/bu</div>

<div class="section">
  <h2>Señal Principal</h2>
  <div class="signal-box">{signal}</div>
  <div class="score-row">
    <div>
      <div style="font-size:.72rem;color:#888">Score de confianza</div>
      <div class="score-num">{score}<span style="font-size:.9rem;font-weight:400;color:#aaa">/100</span></div>
    </div>
    <div class="score-bar"><div class="score-fill"></div></div>
  </div>
  <div class="grid">
    <div class="box"><div class="box-label">Precio actual</div><div class="box-val">{fmt(price)} USc/bu</div></div>
    <div class="box"><div class="box-label">Confianza señal</div><div class="box-val">{conf:.1%}</div></div>
    <div class="box"><div class="box-label">Fecha</div><div class="box-val">{date.today().isoformat()}</div></div>
  </div>
</div>

<div class="section">
  <h2>Breakdown del Score ({brkdwn.get('composite_signal','—')})</h2>
  <table><thead><tr><th>Factor</th><th>Dirección</th><th>Detalle</th></tr></thead>
  <tbody>{factors_rows}</tbody></table>
</div>

<div class="section">
  <h2>Señales Multi-Commodity</h2>
  <table><thead><tr><th>Commodity</th><th>Precio</th><th>Señal</th><th>RSI</th><th>Tendencia</th></tr></thead>
  <tbody>{comm_rows}</tbody></table>
</div>

<div class="section">
  <h2>Contexto Global</h2>
  <div class="grid">
    <div class="box">
      <div class="box-label">Demanda China</div>
      <div class="box-val">{china.get('demand_signal','—')}</div>
      <div class="box-sub">Score: {china.get('demand_score','—')}/100</div>
    </div>
    <div class="box">
      <div class="box-label">WASDE</div>
      <div class="box-val">{wasde.get('signal','—')}</div>
      <div class="box-sub">Stocks: {fmt((wasde.get('world') or {{}}).get('ending_stocks_mmt'))} MMT</div>
    </div>
    <div class="box">
      <div class="box-label">Basis Uruguay</div>
      <div class="box-val">{fmt(basis.get('basis_usd_ton'))} USD/t</div>
      <div class="box-sub">{basis.get('signal','—')} · Z: {fmt(basis.get('basis_zscore'),2)}</div>
    </div>
    <div class="box">
      <div class="box-label">Brasil Exportaciones</div>
      <div class="box-val">{fmt(brazil.get('pct_completed'))}%</div>
      <div class="box-sub">completado · {brazil.get('signal','—')}</div>
    </div>
    <div class="box">
      <div class="box-label">Crush Margin</div>
      <div class="box-val">{fmt((china.get('crush_margin') or {{}}).get('margin_usd_ton'))} USD/t</div>
      <div class="box-sub">{(china.get('crush_margin') or {{}}).get('signal','—')}</div>
    </div>
    <div class="box">
      <div class="box-label">CNY/USD</div>
      <div class="box-val">{fmt(china.get('cny_usd'),3)}</div>
      <div class="box-sub">{china.get('cny_signal','—')}</div>
    </div>
  </div>
</div>

{saved_section}

<div class="footer">
  AgroCast PRO · Brief generado automáticamente · Las señales son herramientas de apoyo a la decisión,
  no constituyen asesoramiento financiero. Datos: CBOT / USDA FAS / Yahoo Finance.
</div>

</body></html>"""

        return html, 200, {"Content-Type": "text/html; charset=utf-8"}
    except Exception as e:
        import traceback
        return f"<pre>Error: {e}\n{traceback.format_exc()}</pre>", 500


@app.route("/api/signal_breakdown")
def get_signal_breakdown_api():
    """GET /api/signal_breakdown — Breakdown explicado del score de confianza."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.trader.signal_breakdown import get_signal_breakdown
        return jsonify(get_signal_breakdown())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/accountability")
def get_accountability():
    """GET /api/accountability — Historial de forecasts vs precios reales."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.trader.accountability import get_accountability_records
        return jsonify(get_accountability_records())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/llm_accountability")
def get_llm_accountability():
    """GET /api/llm_accountability — Hit rate del Brief LLM (snapshot diario + verificación 7d)."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.intel.llm_accountability import get_summary, record_today
        record_today()  # idempotente
        return jsonify(get_summary())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/ml_quality")
def get_ml_quality():
    """GET /api/ml_quality — Métricas honestas del clasificador ML 7d."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.trader.ml_quality import compute_ml_quality
        return jsonify(compute_ml_quality())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/market_synthesis", methods=["GET", "POST"])
def get_market_synthesis():
    """GET → devuelve cache. POST → fuerza re-generación."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.intel.market_synthesis import synthesize, load_synthesis
        if request.method == "POST":
            out = synthesize(force=True)
            if not out:
                return jsonify({"ok": False, "error": "no se pudo generar"}), 500
            return jsonify(out)
        cached = load_synthesis()
        if not cached:
            # Generación lazy en primer GET
            cached = synthesize(force=False) or {}
        if not cached:
            return jsonify({"ok": False, "error": "sin contexto"}), 404
        return jsonify(cached)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/news_intel")
def get_news_intel():
    """GET /api/news_intel — Análisis LLM de noticias agregado por driver."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.intel.aggregator import load_intel
        intel = load_intel()
        if not intel:
            return jsonify({"ok": False, "error": "intel no generado aún"}), 404
        return jsonify(intel)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/next_wasde")
def get_next_wasde():
    """GET /api/next_wasde — Próximas 3 fechas WASDE y días restantes."""
    try:
        from datetime import date, timedelta
        today = date.today()
        dates = []
        for offset in range(0, 6):
            year  = today.year + (today.month + offset - 1) // 12
            month = (today.month + offset - 1) % 12 + 1
            first = date(year, month, 1)
            days_until_tue = (1 - first.weekday()) % 7
            second_tue = first + timedelta(days=days_until_tue) + timedelta(weeks=1)
            if second_tue >= today:
                days_ahead = (second_tue - today).days
                dates.append({
                    "date":       second_tue.isoformat(),
                    "days_ahead": days_ahead,
                    "urgent":     days_ahead <= 2,
                    "soon":       days_ahead <= 7,
                })
            if len(dates) >= 3:
                break
        return jsonify({"ok": True, "next_reports": dates, "today": today.isoformat()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/curve_history")
def get_curve_history():
    """GET /api/curve_history — Últimos N días del front-next spread y roll yield."""
    try:
        import pandas as pd
        path = os.path.join(PROJECT_ROOT, "data", "curve_history.csv")
        if not os.path.exists(path):
            return jsonify({"ok": False, "error": "sin historia de curva aún"}), 404
        df = pd.read_csv(path).tail(120)
        records = df.to_dict("records")
        last = records[-1] if records else {}
        return jsonify({
            "ok": True,
            "n":  len(records),
            "current": last,
            "history": records,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/next_event")
def get_next_event():
    """GET /api/next_event — Próximos eventos USDA (WASDE/Acreage/Stocks) con countdown."""
    try:
        from datetime import date as _d
        sys.path.insert(0, PROJECT_ROOT)
        from src.data.event_calendar import build_event_calendar
        today = _d.today()
        cal   = build_event_calendar(today.year, today.year + 1)
        upcoming = []
        for ev in cal:
            ev_date = _d.fromisoformat(ev["date"])
            if ev_date >= today:
                days_ahead = (ev_date - today).days
                upcoming.append({
                    "date":            ev["date"],
                    "kind":            ev["kind"],
                    "cost_multiplier": ev["cost_multiplier"],
                    "days_ahead":      days_ahead,
                    "urgent":          days_ahead <= 2,
                    "soon":            days_ahead <= 7,
                })
            if len(upcoming) >= 5:
                break
        return jsonify({"ok": True, "today": today.isoformat(), "upcoming": upcoming})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/monthly_forecast")
def get_monthly_forecast():
    """GET /api/monthly_forecast — Forecast robusto ETS/seasonal-naive a 90d."""
    try:
        import pandas as pd
        path = os.path.join(PROJECT_ROOT, "artifacts", "monthly_forecast.csv")
        if not os.path.exists(path):
            return jsonify({"ok": False, "error": "monthly_forecast.csv no generado"}), 404
        df = pd.read_csv(path)
        records = df.to_dict("records")
        method = records[0].get("method", "?") if records else "?"
        return jsonify({
            "ok":       True,
            "method":   method,
            "horizon":  len(records),
            "forecast": records,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/cme")
def get_cme():
    """GET /api/cme — Snapshot + histórico (últimos 60 días) de CME ZS futures."""
    try:
        import pandas as pd
        path = os.path.join(PROJECT_ROOT, "data", "cme_history.csv")
        if not os.path.exists(path):
            return jsonify({"ok": False, "error": "cme_history.csv no generado aún"}), 404
        df = pd.read_csv(path).sort_values("Date")
        if df.empty:
            return jsonify({"ok": False, "error": "sin datos"}), 404
        latest = df.iloc[-1].to_dict()
        history = df.tail(60).to_dict("records")
        # Cambios vs. 5 días atrás (guard contra NaN, None y división por cero)
        oi_change_pct_5d = None
        if len(df) >= 6:
            oi_now = df["total_oi"].iloc[-1]
            oi_5d  = df["total_oi"].iloc[-6]
            if pd.notna(oi_now) and pd.notna(oi_5d) and abs(oi_5d) > 1e-6:
                oi_change_pct_5d = round((oi_now - oi_5d) / oi_5d * 100, 2)
        return jsonify({
            "ok":               True,
            "latest":           latest,
            "oi_change_pct_5d": oi_change_pct_5d,
            "n_days":           len(df),
            "history":          history,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/satellite")
def get_satellite():
    """GET /api/satellite — Clima/stress agregado de regiones sojeras (NASA POWER)."""
    try:
        import pandas as pd
        path = os.path.join(PROJECT_ROOT, "data", "satellite_history.csv")
        if not os.path.exists(path):
            return jsonify({"ok": False, "error": "satellite_history.csv no generado"}), 404
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values(["Date", "region"])
        if df.empty:
            return jsonify({"ok": False, "error": "sin datos"}), 404

        weights = {"us_iowa": 0.35, "br_mt": 0.35, "ar_pampa": 0.20, "uy_oeste": 0.10}
        df["w"] = df["region"].map(weights).fillna(0)
        agg = df.groupby("Date", group_keys=False).apply(lambda g: pd.Series({
            "precip_mm":   float((g["precip_mm"]    * g["w"]).sum() / max(g["w"].sum(), 1e-9)),
            "tmax_c":      float((g["tmax_c"]       * g["w"]).sum() / max(g["w"].sum(), 1e-9)),
            "stress":      float((g["crop_stress_score"] * g["w"]).sum() / max(g["w"].sum(), 1e-9)),
        }), include_groups=False).reset_index().tail(180)

        latest_by_region = (
            df.sort_values("Date").groupby("region").tail(1)
              [["region", "Date", "precip_mm", "tmax_c", "rh_pct", "crop_stress_score"]]
              .to_dict("records")
        )
        for r in latest_by_region:
            r["Date"] = str(r["Date"])[:10]

        return jsonify({
            "ok":         True,
            "by_region":  latest_by_region,
            "aggregated": [
                {
                    "Date":       str(r["Date"])[:10],
                    "precip_mm":  round(float(r["precip_mm"]), 2),
                    "tmax_c":     round(float(r["tmax_c"]),    2),
                    "stress":     round(float(r["stress"]),    3),
                }
                for r in agg.to_dict("records")
            ],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/crop_progress")
def get_crop_progress():
    """GET /api/crop_progress — USDA NASS Crop Progress weekly (soja US)."""
    try:
        import pandas as pd
        path = os.path.join(PROJECT_ROOT, "data", "crop_progress.csv")
        if not os.path.exists(path):
            return jsonify({"ok": False,
                            "error": "crop_progress.csv no generado (requiere USDA_API_KEY)"}), 404
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
        if df.empty:
            return jsonify({"ok": False, "error": "sin datos"}), 404

        records = df.tail(40).to_dict("records")
        for r in records:
            r["Date"] = str(r["Date"])[:10]
        latest = records[-1] if records else {}
        return jsonify({
            "ok":     True,
            "latest": latest,
            "weeks":  len(df),
            "history": records,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/drift_monitor")
def get_drift_monitor():
    """GET /api/drift_monitor — Salud del modelo en producción (rolling 30/60/90d)."""
    try:
        import json as _json
        path = os.path.join(PROJECT_ROOT, "artifacts", "drift_monitor.json")
        # Auto-corre si no existe (1ª vez) o tiene >24h
        needs_refresh = True
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            needs_refresh = (time.time() - mtime) > 24 * 3600
        if needs_refresh:
            try:
                sys.path.insert(0, PROJECT_ROOT)
                from src.model.drift_monitor import run as _run_drift
                _run_drift()
            except Exception as _re:
                print(f"[drift_monitor] auto-refresh fallo: {_re}")

        if not os.path.exists(path):
            return jsonify({"ok": False, "error": "drift_monitor.json no disponible"}), 404
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(_json.load(f))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/lookahead_audit")
def get_lookahead_audit():
    """GET /api/lookahead_audit — resultados de la auditoría OOS por cortes temporales."""
    try:
        import json as _json
        path = os.path.join(PROJECT_ROOT, "artifacts", "lookahead_audit.json")
        if not os.path.exists(path):
            return jsonify({"ok": False,
                            "error": "Aún no corrida. Ejecutá: python -m src.model.audit_lookahead"}), 404
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        data["ok"] = True
        return jsonify(data)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/ensemble")
def get_ensemble():
    """GET /api/ensemble — Señal Bayesian (modelo + LLM ponderados por accuracy)."""
    try:
        import json as _json
        path = os.path.join(PROJECT_ROOT, "artifacts", "ensemble_signal.json")
        if not os.path.exists(path):
            return jsonify({"ok": False, "error": "ensemble_signal.json no generado"}), 404
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        data["ok"] = True
        return jsonify(data)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/multi_commodity")
def get_multi_commodity():
    """GET /api/multi_commodity — Señales técnicas Soja, Maíz y Trigo."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.data.multi_commodity_signals import get_multi_commodity_signals
        return jsonify(get_multi_commodity_signals())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/wasde_stress")
def get_wasde_stress():
    """GET /api/wasde_stress — Stress test WASDE: top 5 reports más volátiles."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.data.wasde_stress_test import get_wasde_stress_test
        return jsonify(get_wasde_stress_test())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/cot_analogs")
def get_cot_analogs():
    """GET /api/cot_analogs — Análogos históricos COT y outcomes de precio."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.trader.cot_analogs import get_cot_analogs as _get_analogs
        data = _get_analogs()
        data["ok"] = True
        return jsonify(data)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/storage_roi")
def get_storage_roi():
    """GET /api/storage_roi?price=320&forecast_return=2.5&financing_rate=8"""
    try:
        price           = float(request.args.get("price", 0))
        forecast_return = request.args.get("forecast_return")
        financing_rate  = request.args.get("financing_rate")

        if price <= 0:
            # Intentar obtener precio actual desde signals.csv
            sig_path = os.path.join(ARTIFACTS_DIR, "signals.csv")
            if os.path.exists(sig_path):
                df = pd.read_csv(sig_path)
                if "Price" in df.columns:
                    price_usc = float(df["Price"].dropna().iloc[-1])
                    price = round((price_usc / 100) * 36.744 - 25, 2)   # USD/ton local

        sys.path.insert(0, PROJECT_ROOT)
        from src.producer.sell_signal import compute_storage_roi
        roi = compute_storage_roi(
            price_usd_ton=price,
            forecast_return_pct=float(forecast_return) if forecast_return else None,
            financing_rate_annual_pct=float(financing_rate) if financing_rate else None,
        )
        return jsonify({"ok": True, "price_usd_ton": price, "roi": roi})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/")
def home():
    with open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.route("/landing")
def landing():
    with open(os.path.join(BASE_DIR, "landing.html"), encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    print("=" * 50)
    print("  AgroCast PRO")
    print("=" * 50)

    # Por defecto corre el pipeline COMPLETO al arrancar (data + modelo +
    # backtest + auditoría). Para arranque rápido (solo servidor con artifacts
    # existentes), exportá AGROCAST_FAST_START=1.
    fast_start = os.environ.get("AGROCAST_FAST_START", "0") == "1"
    has_artifacts = os.path.exists(os.path.join(ARTIFACTS_DIR, "forecast.csv"))

    if fast_start and has_artifacts:
        print("[FAST] AGROCAST_FAST_START=1 — saltando pipeline. Usando artifacts existentes.")
        _check_signal_change()
    elif not has_artifacts:
        # Primera vez: NO hay forecast.csv — sí hay que esperar (sin esto el dashboard
        # carga vacío y todos los endpoints devuelven 404).
        print("[>>] Primera ejecucion — pipeline completo bloqueante (única vez)…")
        run_pipeline_blocking()
        _check_signal_change()
    else:
        # Caso normal: ya hay artifacts. Servimos al usuario YA y refrescamos en background.
        print("[>>] Artifacts existentes — refrescando pipeline en background…")
        run_pipeline_background()

    # Auditoría look-ahead en background (no bloquea el server)
    def _run_audit_bg():
        try:
            audit_proc = subprocess.run(
                [sys.executable, "-m", "src.model.audit_lookahead"],
                cwd=PROJECT_ROOT, timeout=180,
                capture_output=True, text=True,
            )
            if audit_proc.returncode == 0:
                print("[OK] Auditoría completada → artifacts/lookahead_audit.json")
            else:
                print(f"[WARN] Auditoría falló (rc={audit_proc.returncode})")
        except subprocess.TimeoutExpired:
            print("[WARN] Auditoría: timeout 180s — skip")
        except Exception as _ae:
            print(f"[WARN] Auditoría: {_ae}")
    threading.Thread(target=_run_audit_bg, daemon=True).start()

    # ── Scheduler automático ──────────────────────────────────────
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src.infra.scheduler import start_scheduler
        start_scheduler()
    except Exception as _se:
        print(f"[Scheduler] No iniciado: {_se}")

    print(f"\n  Servidor en: http://localhost:{port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)
