"""
src/producer/producer_brief.py
Capa de PRODUCTO para el productor agropecuario.

Este módulo agrega TODA la inteligencia del motor (Intelligence Engine,
modelo ML, forecast, basis, estacionalidad, calendario WASDE, noticias) y la
traduce a un único brief en LENGUAJE DEL PRODUCTOR. El productor no maneja
riesgo de mercado, no conoce CBOT/COT/crush margin: solo quiere saber
"¿cuándo y a cuánto vendo mi soja?".

Filosofía:
  - Una sola respuesta clara: VENDER / ESPERAR / NO_VENDER.
  - Precio en las unidades que el productor reconoce (USD/ton, UYU/ton).
  - Drivers de mercado traducidos a frases simples con íconos.
  - Ventana óptima de venta como fecha de calendario.
  - Precio neto en el bolsillo (descontando flete y gastos).

NO expone: agentes de debate, AUC, drift, COT index, IV z-score, QuantAgent.
Todo eso vive en el "motor interno" (capa 2), de uso propio.
"""

import os
import json
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA = os.path.join(_PROJECT_ROOT, "data")
_ARTIFACTS = os.path.join(_PROJECT_ROOT, "artifacts")


# ─────────────────────────────────────────────────────────────────────────
# Traducción de drivers técnicos → lenguaje del productor
# Cada driver: (ícono+etiqueta corta, explicación de una línea)
# ─────────────────────────────────────────────────────────────────────────
DRIVER_LABELS = {
    "china_demand": ("🇨🇳 Demanda de China", "China es el principal comprador mundial de soja"),
    "supply_global": ("🌍 Oferta mundial", "Cuánta soja hay disponible en el mundo"),
    "usda_report":  ("📋 Reportes USDA", "Datos oficiales de oferta y demanda de EE.UU."),
    "weather_br":   ("☔ Clima en Brasil", "Brasil tiene la cosecha más grande del mundo"),
    "weather_ar":   ("🌦️ Clima en Argentina", "Argentina es un gran exportador vecino"),
    "weather_us":   ("🌤️ Clima en EE.UU.", "EE.UU. define la oferta del segundo semestre"),
    "policy_ar":    ("🏛️ Política en Argentina", "Retenciones y cepo afectan la oferta regional"),
    "policy_us":    ("🏛️ Política en EE.UU.", "Decisiones comerciales de EE.UU."),
    "policy_br":    ("🏛️ Política en Brasil", "Decisiones comerciales de Brasil"),
    "macro_usd":    ("💵 Dólar", "Un dólar fuerte abarata la soja para compradores"),
    "macro_oil":    ("🛢️ Petróleo", "Afecta biocombustibles y costos de flete"),
    "logistics":    ("🚢 Logística y fletes", "Costos de transporte hacia los puertos"),
    "biofuels":     ("⛽ Biocombustibles", "Demanda de aceite de soja para biodiésel"),
    "geopolitics":  ("🌐 Geopolítica", "Conflictos y acuerdos que mueven el mercado"),
    "other":        ("📰 Otros factores", "Otras noticias del mercado"),
}

_MONTHS_ES = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio",
              "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def _fmt_fecha(d: date) -> str:
    """'23 de junio' — formato humano en español."""
    return f"{d.day} de {_MONTHS_ES[d.month]}"


# ─────────────────────────────────────────────────────────────────────────
# Carga del veredicto del Intelligence Engine (fuente de verdad)
# ─────────────────────────────────────────────────────────────────────────
def _load_ie_verdict() -> dict | None:
    path = os.path.join(_DATA, "intelligence_engine_verdict.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            ie = json.load(f)
        fm = ie.get("verdict", {})
        if not isinstance(fm, dict) or "error" in fm:
            return None  # parse falló → caer a ML
        ts = ie.get("timestamp", "")
        age_h = None
        if ts:
            try:
                age_h = (datetime.now() - datetime.fromisoformat(ts)).total_seconds() / 3600
            except Exception:
                pass
        return {
            "signal":      fm.get("verdict"),               # BUY/SELL/HOLD
            "confidence":  fm.get("confidence"),
            "price_range_7d":  fm.get("price_range_7d"),
            "price_range_30d": fm.get("price_range_30d"),
            "producers_action": (fm.get("recommended_action", {}) or {}).get("producers"),
            "bull_bear":   fm.get("bull_bear_balance", {}),
            "timestamp":   ts,
            "age_hours":   round(age_h, 1) if age_h is not None else None,
            "stale":       (age_h is not None and age_h > 72),
        }
    except Exception:
        return None


def _load_ml_signal() -> dict | None:
    path = os.path.join(_ARTIFACTS, "signals.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        last = df.iloc[-1]
        return {
            "signal":     str(last["signal"]),
            "confidence": float(last["confidence"]),
            "date":       str(last.get("Date", ""))[:10],
        }
    except Exception:
        return None


def _current_price_usc_bu() -> float | None:
    """Precio actual del front-month.

    Prioriza el live CME (current_contract.json) SOLO si está fresco (<36h);
    si está stale o no existe, cae al último cierre histórico de raw_market.
    Evita el bug de precio congelado (current_contract.json puede quedar viejo
    al migrar de servidor)."""
    # 1) live CME — solo si es reciente
    try:
        cc = os.path.join(_DATA, "current_contract.json")
        if os.path.exists(cc):
            with open(cc) as f:
                ct = json.load(f)
            gen = ct.get("generated_at", "")
            fresh = False
            if gen:
                try:
                    age_h = (datetime.now() - datetime.fromisoformat(gen)).total_seconds() / 3600
                    fresh = age_h < 36
                except Exception:
                    fresh = False
            if ct.get("ok") and ct.get("price") and fresh:
                return float(ct["price"])
    except Exception:
        pass
    # 2) último cierre histórico (fuente consistente con forecast/tendencia)
    try:
        mkt = pd.read_csv(os.path.join(_DATA, "raw_market.csv"))
        return float(mkt["Soybeans"].iloc[-1])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────
# Conversión de precios (reusa convención del módulo sell_signal)
# ─────────────────────────────────────────────────────────────────────────
BUSHELS_PER_TON = 36.744


def _get_basis() -> float:
    """Basis Uruguay actual (USD/ton). Intenta leer el cálculo real, fallback -25."""
    try:
        bp = os.path.join(_DATA, "basis_uruguay.json")
        if os.path.exists(bp):
            with open(bp) as f:
                b = json.load(f)
            val = b.get("basis_usd_ton") or b.get("basis")
            if val is not None and -80 < float(val) < 20:
                return float(val)
    except Exception:
        pass
    return float(os.getenv("URUGUAY_BASIS_USD_TON", "-25"))


def _get_uyu_rate() -> float:
    try:
        import requests
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5,
                         headers={"User-Agent": "AgroCastPRO/1.0"})
        if r.status_code == 200:
            uyu = r.json().get("rates", {}).get("UYU")
            if uyu and 35 < float(uyu) < 80:
                return round(float(uyu), 2)
    except Exception:
        pass
    return float(os.getenv("UYU_USD_RATE", "42.0"))


def _build_prices(price_usc_bu: float, basis: float, uyu_rate: float) -> dict:
    price_usd_bu = price_usc_bu / 100.0
    chicago_usd_ton = price_usd_bu * BUSHELS_PER_TON
    local_usd_ton = chicago_usd_ton + basis
    return {
        "usc_bu":          round(price_usc_bu, 1),
        "chicago_usd_ton": round(chicago_usd_ton, 1),
        "basis_usd_ton":   round(basis, 1),
        "usd_ton":         round(local_usd_ton, 1),
        "uyu_ton":         round(local_usd_ton * uyu_rate, 0),
        "uyu_per_usd":     uyu_rate,
    }


# ─────────────────────────────────────────────────────────────────────────
# Tendencia de precio (30 días)
# ─────────────────────────────────────────────────────────────────────────
def _price_trend() -> dict:
    try:
        mkt = pd.read_csv(os.path.join(_DATA, "raw_market.csv"))
        s = mkt["Soybeans"].dropna()
        if len(s) < 31:
            return {"pct_30d": None, "direction": "estable", "arrow": "→"}
        now = float(s.iloc[-1])
        past = float(s.iloc[-31])
        pct = (now - past) / past * 100
        if pct > 1.5:
            direction, arrow = "alcista", "↗"
        elif pct < -1.5:
            direction, arrow = "bajista", "↘"
        else:
            direction, arrow = "estable", "→"
        return {"pct_30d": round(pct, 1), "direction": direction, "arrow": arrow}
    except Exception:
        return {"pct_30d": None, "direction": "estable", "arrow": "→"}


# ─────────────────────────────────────────────────────────────────────────
# Ventana óptima de venta (peak del monthly_forecast 90d)
# ─────────────────────────────────────────────────────────────────────────
def _best_window(current_usc_bu: float | None) -> dict | None:
    try:
        df = pd.read_csv(os.path.join(_ARTIFACTS, "monthly_forecast.csv"))
        if df.empty or "forecast" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        # Ignorar los primeros 7 días (ruido inmediato); buscar pico en el resto
        horizon = df.iloc[7:] if len(df) > 14 else df
        idx = horizon["forecast"].idxmax()
        peak = horizon.loc[idx]
        peak_date = peak["Date"].date()
        peak_price = float(peak["forecast"])
        # Ventana = ±4 días alrededor del pico
        win_start = peak_date - timedelta(days=4)
        win_end = peak_date + timedelta(days=4)
        delta_pct = None
        if current_usc_bu:
            delta_pct = round((peak_price - current_usc_bu) / current_usc_bu * 100, 1)
        return {
            "fecha_pico":     str(peak_date),
            "fecha_pico_es":  _fmt_fecha(peak_date),
            "ventana_inicio": str(win_start),
            "ventana_fin":    str(win_end),
            "ventana_es":     f"{_fmt_fecha(win_start)} – {_fmt_fecha(win_end)}",
            "precio_estimado_usc": round(peak_price, 0),
            "delta_pct":      delta_pct,
            "mejora":         (delta_pct is not None and delta_pct > 0.5),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────
# Próximo evento de mercado relevante (WASDE)
# ─────────────────────────────────────────────────────────────────────────
def _next_wasde() -> dict | None:
    """Segundo martes del mes (lógica del proyecto)."""
    try:
        today = date.today()

        def second_tuesday(y, m):
            d = date(y, m, 1)
            # primer martes
            first_tue = d + timedelta(days=(1 - d.weekday()) % 7)
            return first_tue + timedelta(days=7)

        candidates = []
        for off in range(0, 3):
            m = today.month + off
            y = today.year + (m - 1) // 12
            m = (m - 1) % 12 + 1
            candidates.append(second_tuesday(y, m))
        upcoming = [d for d in candidates if d >= today]
        if not upcoming:
            return None
        nxt = min(upcoming)
        days_to = (nxt - today).days
        return {
            "nombre":     "Reporte WASDE del USDA",
            "fecha":      str(nxt),
            "fecha_es":   _fmt_fecha(nxt),
            "dias_para":  days_to,
            "impacto":    "Puede mover el precio ±3% en un día",
            "inminente":  days_to <= 5,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────
# Track record — el historial honesto que construye confianza.
# "Cuando AgroCast dio una señal, ¿acertó?" Es lo único que rompe la
# desconfianza del productor. El dato ya está computado por ie_accountability.
# ─────────────────────────────────────────────────────────────────────────
def _track_record() -> dict | None:
    try:
        from src.intel.ie_accountability import get_verdict_history
        h = get_verdict_history()
    except Exception:
        return None
    if not h or not h.get("ok"):
        return None

    n_eval = h.get("verified_7d") or 0
    prec   = h.get("direction_accuracy_7d")
    if n_eval == 0:
        # Todavía sin señales maduras para evaluar
        return {
            "n_evaluadas": 0,
            "mensaje": "Todavía estamos acumulando historial verificable. "
                       "Cada señal se evalúa a 7 días para medir si acertó.",
            "recientes": [],
            "muestra_chica": True,
        }

    aciertos = round((prec or 0) / 100 * n_eval)
    label_sig = {"SELL": "Anticipamos baja", "BUY": "Anticipamos suba",
                 "HOLD": "Anticipamos estabilidad"}

    recientes = []
    for r in (h.get("recent") or []):
        if not r.get("verified_7d"):
            continue
        ret = r.get("return_7d_pct")
        ok = r.get("direction_correct_7d")
        sig = r.get("verdict")
        # Fecha humana
        try:
            d = date.fromisoformat(r.get("date"))
            fecha = _fmt_fecha(d)
        except Exception:
            fecha = r.get("date", "")
        if ret is not None:
            mov = "bajó" if ret < 0 else ("subió" if ret > 0 else "quedó igual")
            resultado = f"el precio {mov} {abs(ret):.1f}% en 7 días"
        else:
            resultado = "—"
        recientes.append({
            "fecha":    fecha,
            "senal":    label_sig.get(sig, sig),
            "resultado": resultado,
            "acierto":  bool(ok),
        })
        if len(recientes) >= 5:
            break

    muestra_chica = n_eval < 8
    if prec is not None and prec >= 60:
        mensaje = (f"En las últimas {n_eval} señales evaluadas, acertamos la dirección "
                   f"del precio en {aciertos} de {n_eval} ({prec:.0f}%).")
    else:
        mensaje = (f"Llevamos {n_eval} señales evaluadas, con {aciertos} aciertos de "
                   f"dirección ({prec:.0f}%). Seguimos midiéndonos con transparencia.")
    if muestra_chica:
        mensaje += " (Historial todavía corto — lo mostramos con total transparencia.)"

    return {
        "n_evaluadas":   n_eval,
        "aciertos":      aciertos,
        "precision_pct": prec,
        "mensaje":       mensaje,
        "recientes":     recientes,
        "muestra_chica": muestra_chica,
    }


# ─────────────────────────────────────────────────────────────────────────
# Inteligencia de basis — traduce el descuento local a decisión simple
# El productor tiene DOS palancas: el precio de Chicago y el basis local.
# Saber si el basis está caro o barato vs su historia separa al productor
# amateur del profesional. El dato (zscore, pct_rank) ya está calculado.
# ─────────────────────────────────────────────────────────────────────────
def _basis_intelligence() -> dict | None:
    try:
        with open(os.path.join(_DATA, "basis_uruguay.json")) as f:
            b = json.load(f)
    except Exception:
        return None

    basis = b.get("basis_usd_ton")
    pct_rank = b.get("basis_pct_rank")   # 0 = más amplio (peor), 1 = más angosto (mejor)
    avg5 = b.get("basis_5yr_avg")
    if basis is None or pct_rank is None:
        return None

    # Diferencia vs promedio histórico (positivo = mejor que lo normal)
    delta_vs_avg = round(basis - avg5, 1) if avg5 is not None else None

    # pct_rank alto = basis angosto = físico local caro = bueno para vender físico
    if pct_rank >= 0.7:
        semaforo = "green"
        titulo = "El físico uruguayo está caro"
        detalle = ("El descuento local está más angosto que el "
                   f"{round(pct_rank*100)}% de los últimos 5 años. "
                   "Aunque Chicago no esté alto, conviene fijar el grano físico ahora.")
    elif pct_rank <= 0.3:
        semaforo = "red"
        titulo = "El físico uruguayo está barato"
        detalle = ("El descuento local está más amplio que el "
                   f"{round((1-pct_rank)*100)}% de los últimos 5 años"
                   + (f" (recibís ~${abs(delta_vs_avg):.0f}/ton menos que lo normal vs Chicago)" if delta_vs_avg and delta_vs_avg < 0 else "")
                   + ". Si podés esperar, el descuento podría normalizarse.")
    else:
        semaforo = "yellow"
        titulo = "El descuento local está en rango normal"
        detalle = ("El basis está cerca de su promedio histórico. "
                   "La decisión depende más del precio de Chicago que del basis.")

    return {
        "basis_usd_ton":   round(basis, 1),
        "promedio_5y":     round(avg5, 1) if avg5 is not None else None,
        "delta_vs_avg":    delta_vs_avg,
        "percentil":       round(pct_rank * 100),
        "semaforo":        semaforo,
        "titulo":          titulo,
        "detalle":         detalle,
    }


# ─────────────────────────────────────────────────────────────────────────
# Decisión de almacenamiento — ¿conviene guardar o vender?
# Traduce el ROI de almacenamiento (costo silo + financiero + riesgo calidad)
# vs la apreciación proyectada del precio, a una decisión simple.
# ─────────────────────────────────────────────────────────────────────────
def _forecast_return_30d(current_usc_bu: float | None) -> float | None:
    """Retorno proyectado a ~30 días (%) desde el monthly_forecast."""
    if not current_usc_bu:
        return None
    try:
        df = pd.read_csv(os.path.join(_ARTIFACTS, "monthly_forecast.csv"))
        if df.empty or "forecast" not in df.columns:
            return None
        # Tomar el valor a ~30 días (o el último disponible)
        idx = min(29, len(df) - 1)
        px_30 = float(df["forecast"].iloc[idx])
        return round((px_30 - current_usc_bu) / current_usc_bu * 100, 2)
    except Exception:
        return None


def _storage_decision(price_local_usd_ton: float, current_usc_bu: float | None) -> dict | None:
    try:
        from src.producer.sell_signal import compute_storage_roi
    except Exception:
        return None
    ret30 = _forecast_return_30d(current_usc_bu)
    roi = compute_storage_roi(
        price_usd_ton=price_local_usd_ton,
        forecast_return_pct=ret30,
        days_list=[30, 60],
    )
    if not roi:
        return None
    r30 = roi[0]
    costo = r30.get("total_cost_usd")
    # Umbral de suba necesaria para empatar el costo de guardar 30d
    break_even_pct = round((costo / price_local_usd_ton) * 100, 2) if price_local_usd_ton else None
    rec = r30.get("recommendation")
    if rec == "ALMACENAR":
        semaforo, titulo = "green", "Conviene guardar"
    elif rec == "VENDER":
        semaforo, titulo = "red", "No conviene guardar"
    else:
        semaforo, titulo = "yellow", "Guardar es marginal"

    # Mensaje conversacional
    msg = f"Guardar 30 días te cuesta ~${costo:.1f}/ton. "
    if break_even_pct is not None:
        msg += f"Para que valga la pena, el precio tiene que subir más de {break_even_pct:.1f}% en ese plazo. "
    if ret30 is not None:
        if ret30 > 0:
            msg += f"Proyección actual: +{ret30:.1f}%. "
        else:
            msg += f"Proyección actual: {ret30:.1f}% (a la baja). "
    if rec == "VENDER":
        msg += "El costo de guardar supera lo que se espera ganar."
    elif rec == "ALMACENAR":
        msg += "La suba esperada cubre el costo de guardar."
    else:
        msg += "Depende de tu necesidad de liquidez."

    return {
        "semaforo":          semaforo,
        "titulo":            titulo,
        "costo_30d_usd_ton": round(costo, 1) if costo is not None else None,
        "suba_necesaria_pct": break_even_pct,
        "proyeccion_30d_pct": ret30,
        "recomendacion":     rec,
        "mensaje":           msg,
    }


# ─────────────────────────────────────────────────────────────────────────
# Inteligencia climática — riesgo sobre el RINDE del productor.
# Previene el error más caro: vender a futuro una cosecha que el clima no
# entrega. En el Río de la Plata, El Niño = más lluvia = favorable para soja;
# La Niña = sequía = riesgo. (Opuesto al Midwest de EE.UU.)
# ─────────────────────────────────────────────────────────────────────────
def _climate_intelligence() -> dict | None:
    try:
        df = pd.read_csv(os.path.join(_DATA, "climate_forecast.csv"))
        if df.empty:
            return None
        last = df.iloc[-1]
    except Exception:
        return None

    phase = str(last.get("enso_phase_3m") or last.get("enso_phase_now") or "").lower()
    val = last.get("enso_value_3m")
    try:
        val = float(val) if val is not None else None
    except Exception:
        val = None

    if "nino" in phase or "niño" in phase:
        semaforo = "green"
        fase_es = "El Niño"
        titulo = "Clima favorable para tu rinde"
        detalle = ("El pronóstico climático (El Niño) suele traer buenas lluvias a la "
                   "región — favorable para el rinde de soja. Tu cosecha tiene sesgo al alza, "
                   "así que podés fijar precio a futuro con más confianza.")
        forward_max = 80
    elif "nina" in phase or "niña" in phase:
        semaforo = "red"
        fase_es = "La Niña"
        titulo = "Riesgo de sequía sobre tu rinde"
        detalle = ("Atención: el pronóstico (La Niña) trae riesgo de sequía en la región. "
                   "Si el rinde cae, podrías quedar corto. No vendas a futuro más del 60% "
                   "de la cosecha que todavía no levantaste.")
        forward_max = 60
    else:
        semaforo = "yellow"
        fase_es = "Neutral"
        titulo = "Clima sin sesgo claro"
        detalle = ("El pronóstico climático es neutral, sin sesgo marcado sobre el rinde. "
                   "Podés fijar a futuro con prudencia (hasta ~70% de lo no cosechado).")
        forward_max = 70

    return {
        "fase":        fase_es,
        "enso_valor":  round(val, 2) if val is not None else None,
        "semaforo":    semaforo,
        "titulo":      titulo,
        "detalle":     detalle,
        "forward_max_pct": forward_max,
    }


# ─────────────────────────────────────────────────────────────────────────
# Drivers de mercado traducidos (desde news_intel + IE bull/bear)
# ─────────────────────────────────────────────────────────────────────────
def _driver_row(icono: str, etiqueta: str, detalle: str, efecto: str, peso: int) -> dict:
    """Estructura uniforme de un driver. peso = fuerza para ordenar (mayor primero)."""
    return {"icono": icono, "etiqueta": etiqueta, "detalle": detalle,
            "efecto": efecto, "_peso": peso}


def _simple_drivers(max_items: int = 5) -> list:
    """
    Construye los drivers de mercado desde los FUNDAMENTALES reales
    (China, Brasil, WASDE, Argentina) — no desde sentiment de noticias.
    Esto da una lectura estable y coherente con el veredicto.
    🔴 = presiona precio abajo · 🟢 = empuja precio arriba · 🟡 = neutro.
    """
    rows = []

    def _read(fn):
        try:
            with open(os.path.join(_DATA, fn)) as f:
                return json.load(f)
        except Exception:
            return None

    # 1. China — demanda + crush margin (el driver más importante)
    china = _read("china_demand.json")
    if china:
        score = china.get("demand_score")
        crush = (china.get("crush_margin") or {}).get("signal", "")
        crush_val = (china.get("crush_margin") or {}).get("margin_usd_ton")
        if score is not None:
            if score < 40 or crush == "NEGATIVO":
                detalle = "China está comprando menos soja"
                if crush_val is not None:
                    detalle += f" (margen de industrialización negativo)"
                rows.append(_driver_row("🔴", "🇨🇳 Demanda de China", detalle,
                                        "presiona el precio hacia abajo", 100))
            elif score > 60:
                rows.append(_driver_row("🟢", "🇨🇳 Demanda de China",
                                        "China está comprando con fuerza",
                                        "empuja el precio hacia arriba", 100))
            else:
                rows.append(_driver_row("🟡", "🇨🇳 Demanda de China",
                                        "China compra a ritmo normal",
                                        "impacto neutro por ahora", 60))

    # 2. Brasil — ritmo de exportación (más oferta = bajista)
    brazil = _read("brazil_exports.json")
    if brazil:
        sig = (brazil.get("signal") or "").upper()
        if sig in ("HIGH", "RECORD", "FAST"):
            rows.append(_driver_row("🔴", "🇧🇷 Exportaciones de Brasil",
                                    "Brasil exporta a ritmo récord (mucha oferta)",
                                    "presiona el precio hacia abajo", 80))
        elif sig in ("LOW", "SLOW"):
            rows.append(_driver_row("🟢", "🇧🇷 Exportaciones de Brasil",
                                    "Brasil exporta poco (menos oferta)",
                                    "empuja el precio hacia arriba", 80))
        else:
            rows.append(_driver_row("🟡", "🇧🇷 Exportaciones de Brasil",
                                    "Brasil exporta a ritmo normal",
                                    "impacto neutro por ahora", 40))

    # 3. WASDE — balance oferta/demanda global
    wasde = _read("wasde_official.json")
    if wasde:
        sig = (wasde.get("signal") or "").upper()
        if sig in ("BULLISH", "BULL"):
            rows.append(_driver_row("🟢", "📋 Reporte USDA (WASDE)",
                                    "Los datos oficiales apuntan a precios más altos",
                                    "empuja el precio hacia arriba", 70))
        elif sig in ("BEARISH", "BEAR"):
            rows.append(_driver_row("🔴", "📋 Reporte USDA (WASDE)",
                                    "Los datos oficiales apuntan a precios más bajos",
                                    "presiona el precio hacia abajo", 70))
        else:
            rows.append(_driver_row("🟡", "📋 Reporte USDA (WASDE)",
                                    "El balance de oferta y demanda está equilibrado",
                                    "impacto neutro por ahora", 30))

    # 4. Argentina — supply regional (cepo/retenciones)
    arg = _read("argentina_supply.json")
    if arg:
        imp = (arg.get("impacto_precio") or "").upper()
        if imp in ("ALCISTA", "BULLISH"):
            rows.append(_driver_row("🟢", "🇦🇷 Oferta de Argentina",
                                    "Argentina libera menos soja al mercado",
                                    "empuja el precio hacia arriba", 50))
        elif imp in ("BAJISTA", "BEARISH"):
            rows.append(_driver_row("🔴", "🇦🇷 Oferta de Argentina",
                                    "Argentina vende fuerte y suma oferta",
                                    "presiona el precio hacia abajo", 50))
        # NEUTRAL → no agregar (evita saturar)

    # Ordenar por peso (más relevante primero) y limitar
    rows.sort(key=lambda r: -r["_peso"])
    out = []
    for r in rows[:max_items]:
        r.pop("_peso", None)
        out.append(r)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Resolución del MOMENTO (VENDER / ESPERAR / NO_VENDER)
# ─────────────────────────────────────────────────────────────────────────
def _resolve_momento(ie: dict | None, ml: dict | None, trend: dict, best_win: dict | None) -> dict:
    """
    Combina IE (prioritario) + ML + tendencia para una recomendación clara.
    Lógica desde la óptica del productor (que quiere VENDER caro):
      - Señal SELL del mercado = el mercado va a BAJAR = NO conviene esperar
        → si el precio está alto, VENDER ya; si está deprimido, es tarde pero
          igual conviene no retener (va a seguir bajando).
      - Señal BUY = el mercado va a SUBIR = ESPERAR / NO_VENDER todavía.
      - HOLD = ESPERAR.
    """
    # Señal primaria: IE si está fresco, si no ML
    src = None
    signal = None
    conf = None
    if ie and ie.get("signal") and not ie.get("stale"):
        signal = ie["signal"]
        conf = ie.get("confidence")
        src = "Análisis de inteligencia (5 analistas)"
    elif ml and ml.get("signal"):
        signal = ml["signal"]
        conf = ml.get("confidence")
        src = "Modelo cuantitativo"
    elif ie and ie.get("signal"):
        signal = ie["signal"]
        conf = ie.get("confidence")
        src = "Análisis de inteligencia (desactualizado)"

    pct30 = trend.get("pct_30d")
    mejora_ventana = best_win.get("mejora") if best_win else False

    # Defaults
    momento = "ESPERAR"
    color = "yellow"
    titulo = "MOMENTO NEUTRO"
    explicacion = "Condiciones mixtas. Evaluá según tu necesidad de liquidez."

    if signal == "SELL":
        # Mercado anticipa baja. Para el productor: no conviene retener.
        momento = "VENDER"
        color = "red_action"  # rojo de mercado pero acción = vender
        titulo = "CONVIENE FIJAR PRECIO PRONTO"
        if mejora_ventana:
            explicacion = ("El mercado anticipa precios a la baja, pero hay una ventana "
                           "de rebote próxima. Conviene fijar precio en ese rebote, no más tarde.")
        else:
            explicacion = ("El mercado anticipa precios a la baja. No conviene retener la "
                           "cosecha esperando mejores precios — el riesgo es que sigan cayendo.")
    elif signal == "BUY":
        momento = "NO_VENDER"
        color = "green"
        titulo = "CONVIENE ESPERAR"
        explicacion = ("El mercado anticipa precios al alza. Si podés financiar el "
                       "almacenamiento, conviene esperar antes de vender.")
    elif signal == "HOLD":
        if mejora_ventana:
            momento = "ESPERAR"
            color = "yellow"
            titulo = "ESPERAR LA PRÓXIMA VENTANA"
            explicacion = ("Sin dirección clara de corto plazo, pero el modelo proyecta una "
                           "ventana de mejor precio próximamente.")
        else:
            momento = "ESPERAR"
            color = "yellow"
            titulo = "MOMENTO NEUTRO"
            explicacion = "El mercado no tiene dirección clara. No hay urgencia para vender."

    return {
        "momento":     momento,        # VENDER | ESPERAR | NO_VENDER
        "color":       color,
        "titulo":      titulo,
        "explicacion": explicacion,
        "senal_mercado": signal,
        "confianza":   round(conf * 100) if isinstance(conf, (int, float)) else None,
        "fuente":      src,
    }


# ─────────────────────────────────────────────────────────────────────────
# Precio neto en el bolsillo
# ─────────────────────────────────────────────────────────────────────────
def _precio_neto(local_usd_ton: float, uyu_rate: float,
                 flete: float = None, otros: float = None) -> dict:
    flete = float(os.getenv("FLETE_PUERTO_USD_TON", "26")) if flete is None else flete
    otros = float(os.getenv("OTROS_GASTOS_USD_TON", "8")) if otros is None else otros
    neto = local_usd_ton - flete - otros
    return {
        "precio_local_usd": round(local_usd_ton, 1),
        "flete_usd":        round(flete, 1),
        "otros_usd":        round(otros, 1),
        "neto_usd_ton":     round(neto, 1),
        "neto_uyu_ton":     round(neto * uyu_rate, 0),
    }


# ─────────────────────────────────────────────────────────────────────────
# API principal
# ─────────────────────────────────────────────────────────────────────────
def build_producer_brief(flete: float = None, otros: float = None) -> dict:
    """
    Construye el brief completo del productor en lenguaje simple.
    Único punto de entrada para el panel comercial.
    """
    price_usc = _current_price_usc_bu()
    if price_usc is None:
        return {"ok": False, "error": "Sin datos de precio disponibles"}

    basis = _get_basis()
    uyu_rate = _get_uyu_rate()
    prices = _build_prices(price_usc, basis, uyu_rate)
    trend = _price_trend()
    best_win = _best_window(price_usc)
    ie = _load_ie_verdict()
    ml = _load_ml_signal()
    momento = _resolve_momento(ie, ml, trend, best_win)
    drivers = _simple_drivers()
    wasde = _next_wasde()
    neto = _precio_neto(prices["usd_ton"], uyu_rate, flete, otros)
    basis_intel = _basis_intelligence()
    track = _track_record()
    clima = _climate_intelligence()
    storage = _storage_decision(prices["usd_ton"], price_usc)

    # Accionable narrativo: priorizar el texto del IE (ya está en lenguaje productor)
    accionable = None
    if ie and ie.get("producers_action") and not ie.get("stale"):
        accionable = ie["producers_action"]

    # Margen sobre la cosecha del productor (si configuró "Mi Campo")
    margen = None
    try:
        from src.producer.producer_profile import load_profile, compute_margin
        profile = load_profile()
        if profile:
            margen = compute_margin(
                profile,
                precio_neto_usd_ton=neto["neto_usd_ton"],
                precio_local_usd_ton=prices["usd_ton"],
                uyu_rate=uyu_rate,
            )
    except Exception:
        margen = None

    # Capa de margen sobre el momento: la gestión de margen pesa más que
    # especular por el techo. Enriquecemos la recomendación con el margen.
    if margen and margen.get("margen_pct") is not None:
        mp = margen["margen_pct"]
        sig = momento.get("senal_mercado")
        if mp >= 15:
            momento["nota_margen"] = (
                f"Tu margen es muy bueno ({mp:+.0f}% sobre el costo). "
                f"Fijar al menos una parte asegura un año rentable.")
        elif sig == "SELL" and mp >= 3:
            momento["nota_margen"] = (
                f"Ya estás {mp:+.0f}% sobre tu costo. Con el mercado a la baja, "
                f"asegurar este margen vale más que esperar un rebote incierto.")
        elif sig == "BUY" and mp < 3:
            momento["nota_margen"] = (
                f"Estás casi en tu costo ({mp:+.0f}%). El mercado proyecta suba: "
                f"si podés financiar el almacenamiento, esperar mejora tu margen.")
        elif mp < -3:
            momento["nota_margen"] = (
                f"Atención: a precio de hoy estás {mp:+.0f}% bajo tu costo. "
                f"Vender realiza una pérdida — evaluá esperar o coberturas.")

    return {
        "ok": True,
        "as_of": date.today().isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "precio_hoy": prices,
        "tendencia": trend,
        "momento": momento,
        "margen": margen,
        "basis": basis_intel,
        "track_record": track,
        "clima": clima,
        "almacenamiento": storage,
        "ventana_optima": best_win,
        "accionable_detallado": accionable,
        "drivers_simple": drivers,
        "proximo_evento": wasde,
        "precio_neto": neto,
        "_meta": {
            "ie_disponible": ie is not None,
            "ie_age_horas": ie.get("age_hours") if ie else None,
            "ie_stale": ie.get("stale") if ie else None,
            "ml_disponible": ml is not None,
        },
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(build_producer_brief())
