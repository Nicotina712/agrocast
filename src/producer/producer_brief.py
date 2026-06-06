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

    # Accionable narrativo: priorizar el texto del IE (ya está en lenguaje productor)
    accionable = None
    if ie and ie.get("producers_action") and not ie.get("stale"):
        accionable = ie["producers_action"]

    return {
        "ok": True,
        "as_of": date.today().isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "precio_hoy": prices,
        "tendencia": trend,
        "momento": momento,
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
