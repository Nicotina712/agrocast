"""
src/producer/decision_engine.py
Motor de decisión del PRODUCTOR — sin adivinar la dirección del precio.

Responde las decisiones reales de quien vende soja física (no especula):
  1. ¿Guardar o vender?   → la curva de futuros dice cuánto paga el mercado
                             por esperar; se compara con silo + financiamiento
                             + merma, y se mide el riesgo de que guardar salga peor.
  2. ¿Cuánto fijar ya?     → margen asegurable con el precio de la nueva cosecha
                             vs riesgo calibrado de quedar debajo del costo a
                             cosecha y riesgo de producción (ENSO).
  3. ¿Qué viene?           → fechas que de verdad mueven el precio.

Fundamento (scripts/research_producer_engines.py, 2026-10-03):
  - La dirección a 14-30d no es predecible con nuestros datos (auditoría del
    clasificador), pero el RIESGO sí: vol 30d fwd corr ~0.38 con EWMA/vol60d.
  - Las bandas normales con EWMA quedan angostas (80% nominal → 73-76% real):
    acá se usan los cuantiles EMPÍRICOS de los retornos estandarizados.
  - Los futuros son la expectativa del mercado para cada mes de entrega: el
    "carry" (diferencia contra el contrato más cercano) es el precio de esperar.
    No es un pronóstico de AgroCast.
  - WASDE no mueve más que un día normal (0.95x, n=120); Acreage 1.46x,
    Stocks trimestrales 1.28x.

Supuestos por defecto (editables, Fase 3 los lleva a "Mi Campo"):
  silo 3 USD/ton/mes · financiamiento 8% anual · merma 0.2%/mes.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.infra.cache_meta import is_fresh, stamp

_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA  = os.path.join(_ROOT, "data")
_CURVE_PATH = os.path.join(_DATA, "futures_curve.json")
_CURVE_TTL_H = 6

BU_PER_TON = 36.744
MONTH_CODES = {1: "F", 3: "H", 5: "K", 7: "N", 8: "Q", 9: "U", 11: "X"}
MESES = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "setiembre", "octubre", "noviembre", "diciembre"]

DEFAULT_COSTS = {
    "silo_usd_ton_mes":   3.0,    # silo bolsa / acopio
    "tasa_anual":         0.08,   # costo de oportunidad / financiamiento
    "merma_pct_mes":      0.002,  # pérdida de peso/calidad
}
HARVEST_MONTH_UY = 5            # cosecha de soja en Uruguay: abril-mayo
MIN_GAIN_USD_TON = 2.0          # ganancia esperada mínima para que guardar "pague"


# ── 1. Curva de futuros ─────────────────────────────────────────────────────
def _upcoming_contracts(n: int = 9, today: date | None = None) -> list[tuple[int, int]]:
    today = today or date.today()
    out, y, m = [], today.year, today.month
    while len(out) < n:
        if m in MONTH_CODES and (y, m) != (today.year, today.month) or \
           (m in MONTH_CODES and (y, m) == (today.year, today.month) and today.day < 14):
            out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def get_futures_curve(force: bool = False) -> list[dict]:
    """Contratos de soja CBOT próximos: [{contrato, ticker, anio, mes, usc_bu, usd_ton}]."""
    if not force and is_fresh(_CURVE_PATH, _CURVE_TTL_H):
        try:
            with open(_CURVE_PATH, encoding="utf-8") as f:
                return json.load(f)["curve"]
        except Exception:
            pass
    import yfinance as yf
    curve = []
    for y, m in _upcoming_contracts():
        tk = f"ZS{MONTH_CODES[m]}{str(y)[-2:]}.CBT"
        try:
            h = yf.Ticker(tk).history(period="5d")["Close"].dropna()
            if len(h):
                px = float(h.iloc[-1])
                curve.append({"contrato": f"{MESES[m][:3].upper()}{str(y)[-2:]}", "ticker": tk,
                              "anio": y, "mes": m, "usc_bu": round(px, 2),
                              "usd_ton": round(px / 100 * BU_PER_TON, 2)})
        except Exception:
            continue
    if curve:
        os.makedirs(_DATA, exist_ok=True)
        with open(_CURVE_PATH, "w", encoding="utf-8") as f:
            json.dump(stamp({"curve": curve}), f, indent=1)
    elif os.path.exists(_CURVE_PATH):          # fallback a la última curva conocida
        with open(_CURVE_PATH, encoding="utf-8") as f:
            curve = json.load(f).get("curve", [])
    return curve


# ── 2. Riesgo calibrado (sin dirección) ─────────────────────────────────────
class RiskModel:
    """Distribución del precio a h días: centro = precio de referencia (sin
    drift propio), ancho = vol actual (EWMA+60d) × forma empírica histórica."""

    def __init__(self):
        df = pd.read_csv(os.path.join(_DATA, "raw_market.csv"), parse_dates=["Date"])
        df = df.sort_values("Date").dropna(subset=["Soybeans"]).reset_index(drop=True)
        r = np.log(df["Soybeans"]).diff()
        r[r.abs() > 0.08] = np.nan                       # saltos de rollover
        ewma = np.sqrt((r.fillna(0) ** 2).ewm(alpha=0.06).mean())
        v60 = r.rolling(60, min_periods=40).std()
        self.sigma_d = (0.5 * ewma + 0.5 * v60).bfill()
        self.px = df["Soybeans"]
        self.sigma_now = float(self.sigma_d.iloc[-1])
        self._z_cache: dict[int, np.ndarray] = {}

    def z_dist(self, h: int) -> np.ndarray:
        """Retornos log a h días estandarizados por la vol ex-ante, demeaned
        (se quita la tendencia histórica: no apostamos a una dirección)."""
        h = int(max(5, min(h, 300)))
        if h not in self._z_cache:
            lr = np.log(self.px.shift(-h) / self.px)
            z = (lr / (self.sigma_d * np.sqrt(h))).dropna().values
            self._z_cache[h] = z - np.median(z)
        return self._z_cache[h]

    def quantiles(self, center: float, h_days: int, qs=(0.10, 0.50, 0.90)) -> dict:
        z = self.z_dist(h_days)
        s = self.sigma_now * np.sqrt(max(h_days, 1))
        return {f"q{int(q*100)}": round(center * float(np.exp(np.quantile(z, q) * s)), 1) for q in qs}

    def prob_below(self, threshold: float, center: float, h_days: int) -> float:
        z = self.z_dist(h_days)
        s = self.sigma_now * np.sqrt(max(h_days, 1))
        return float(np.mean(center * np.exp(z * s) < threshold))


# ── helpers ─────────────────────────────────────────────────────────────────
def _days_until(y: int, m: int, day: int = 15) -> int:
    return max((date(y, m, day) - date.today()).days, 1)


def _local_reference() -> dict:
    """Precio local real (Revista Verde) + base contra Chicago."""
    try:
        with open(os.path.join(_DATA, "basis_uruguay.json"), encoding="utf-8") as f:
            b = json.load(f)
    except Exception:
        b = {}
    return {"local_usd_ton": b.get("fob_usd_ton"), "cbot_usd_ton": b.get("cbot_usd_ton"),
            "basis_usd_ton": b.get("basis_usd_ton"), "campania": b.get("rv_campaign"),
            "fecha": b.get("rv_date"), "fuente": b.get("source")}


def _enso_outlook() -> dict:
    try:
        c = pd.read_csv(os.path.join(_DATA, "climate_forecast.csv")).iloc[-1]
        phase = str(c.get("enso_phase_3m") or c.get("enso_phase_now") or "neutral")
        val = c.get("enso_value_3m")
        return {"fase": phase, "valor_3m": None if pd.isna(val) else round(float(val), 2)}
    except Exception:
        return {"fase": "desconocida", "valor_3m": None}


# ── 3. ¿Guardar o vender? ───────────────────────────────────────────────────
def storage_analysis(local_now: float, curve: list[dict], risk: RiskModel,
                     costs: dict, start_idx: int = 0, start_days: int = 0) -> dict:
    """Valor de guardar desde el contrato `start_idx` hasta cada contrato posterior."""
    if not curve or start_idx >= len(curve) - 1:
        return {"ok": False, "error": "curva de futuros insuficiente"}
    base = curve[start_idx]
    opciones = []
    for c in curve[start_idx + 1:]:
        if (c["anio"], c["mes"]) > (base["anio"] + 1, base["mes"]):
            break                                         # horizonte máx ~12 meses
        months = ((c["anio"] - base["anio"]) * 12 + (c["mes"] - base["mes"]))
        carry = c["usd_ton"] - base["usd_ton"]
        silo = costs["silo_usd_ton_mes"] * months
        fin = local_now * costs["tasa_anual"] * months / 12
        merma = local_now * costs["merma_pct_mes"] * months
        costo = silo + fin + merma
        neta = carry - costo
        h = start_days + int(months * 30.4)
        center = local_now + carry
        # riesgo: precio local al vender ese mes vs vender hoy (neto de costos)
        p_peor = risk.prob_below(local_now + costo, center, h)
        q = risk.quantiles(center, h)
        opciones.append({
            "hasta": f"{MESES[c['mes']]} {c['anio']}", "contrato": c["contrato"], "meses": months,
            "paga_mercado_usd_ton": round(carry, 1), "costo_guardar_usd_ton": round(costo, 1),
            "detalle_costo": {"silo": round(silo, 1), "financiamiento": round(fin, 1), "merma": round(merma, 1)},
            "ganancia_esperada_usd_ton": round(neta, 1),
            # La curva no incluye la mejora estacional de la base local (no hay
            # historia real de base UY todavía): cuánto tendría que mejorar
            # para que guardar pague. El productor lo compara con su zona.
            "mejora_base_necesaria_usd_ton": round(max(0.0, -neta), 1),
            "prob_peor_que_vender_hoy": round(p_peor * 100),
            "precio_local_q10": q["q10"], "precio_local_q50": q["q50"], "precio_local_q90": q["q90"],
        })
    if not opciones:
        return {"ok": False, "error": "sin contratos posteriores en 12 meses"}
    best = max(opciones, key=lambda o: o["ganancia_esperada_usd_ton"])
    paga = best["ganancia_esperada_usd_ton"] >= MIN_GAIN_USD_TON
    return {"ok": True, "desde": f"{MESES[base['mes']]} {base['anio']}", "precio_local_base": round(local_now, 1),
            "opciones": opciones, "mejor": best, "guardar_paga": paga,
            "recomendacion": "GUARDAR" if paga else "VENDER"}


def _storage_text(s: dict, prefijo: str) -> str:
    if not s.get("ok"):
        return "Sin datos de la curva de futuros para evaluar."
    b = s["mejor"]
    if s["guardar_paga"]:
        return (f"{prefijo}guardar hasta {b['hasta']} paga: el mercado paga +{b['paga_mercado_usd_ton']:.0f} USD/ton "
                f"y guardar te cuesta {b['costo_guardar_usd_ton']:.0f} → ganancia esperada "
                f"+{b['ganancia_esperada_usd_ton']:.0f} USD/ton. Riesgo: {b['prob_peor_que_vender_hoy']}% de que "
                f"termine peor que vender {('en cosecha' if 'cosech' in prefijo else 'hoy')}.")
    return (f"{prefijo}guardar no paga por la curva: lo máximo que paga el mercado por esperar "
            f"(+{b['paga_mercado_usd_ton']:.0f} USD/ton hasta {b['hasta']}) no cubre el costo de guardar "
            f"({b['costo_guardar_usd_ton']:.0f} USD/ton). Solo pagaría si la base local mejora más de "
            f"{b['mejora_base_necesaria_usd_ton']:.0f} USD/ton en ese plazo.")


# ── 4. ¿Cuánto fijar ya? (pre-venta de la nueva cosecha) ────────────────────
def pricing_analysis(profile: dict | None, local: dict, curve: list[dict], risk: RiskModel,
                     gastos_usd_ton: float = 0.0) -> dict:
    today = date.today()
    hy = today.year + (1 if today.month > HARVEST_MONTH_UY else 0)
    h_days = _days_until(hy, HARVEST_MONTH_UY)
    harvest_c = next((c for c in curve if (c["anio"], c["mes"]) == (hy, HARVEST_MONTH_UY)), None)

    # Precio de la nueva cosecha: Revista Verde si cotiza esa campaña; si no,
    # futuro de mayo + base actual.
    camp_nueva = f"{hy-1}/{hy}"
    if local.get("local_usd_ton") and local.get("campania") == camp_nueva:
        fwd, fuente = float(local["local_usd_ton"]), f"Revista Verde ({camp_nueva}, {local.get('fecha')})"
    elif harvest_c and local.get("basis_usd_ton") is not None:
        fwd = harvest_c["usd_ton"] + float(local["basis_usd_ton"])
        fuente = f"Futuro {harvest_c['contrato']} + base actual"
    else:
        return {"ok": False, "error": "sin precio de referencia para la nueva cosecha"}

    # Precio NETO en el bolsillo (flete + gastos), igual que la banda de margen.
    fwd_bruto = fwd
    fwd = fwd - (gastos_usd_ton or 0.0)
    out = {"ok": True, "campania": camp_nueva, "precio_fijable_usd_ton": round(fwd, 1),
           "precio_fijable_bruto_usd_ton": round(fwd_bruto, 1), "gastos_usd_ton": round(gastos_usd_ton or 0, 1),
           "fuente_precio": fuente,
           "dias_a_cosecha": h_days, "rango_a_cosecha": risk.quantiles(fwd, h_days)}
    enso = _enso_outlook()
    out["clima"] = enso
    # Tope de pre-venta por riesgo de producción: no comprometer grano que
    # quizás no se coseche. La Niña = riesgo de seca en UY/AR (2008/09, 2017/18,
    # 2022/23). Supuesto de literatura: pendiente validar con rindes DIEA/MGAP.
    tope = 0.30 if "nina" in enso["fase"] else 0.50
    out["tope_preventa_pct"] = int(tope * 100)

    be = (profile or {}).get("break_even_usd_ton")
    if not be:
        out.update({"sugerido_pct": None,
                    "mensaje": "Cargá tu costo en Mi Campo para saber cuánto conviene fijar."})
        return out
    margen = fwd - be
    p_below = risk.prob_below(be, fwd, h_days)
    # Lo que le importa al productor no es solo no perder plata, sino no
    # resignar buena parte del margen que hoy puede asegurar.
    p_half = risk.prob_below(be + margen / 2, fwd, h_days) if margen > 0 else 1.0
    out.update({"break_even_usd_ton": be, "margen_asegurable_usd_ton": round(margen, 1),
                "margen_asegurable_pct": round(margen / be * 100, 1),
                "prob_debajo_costo_a_cosecha": round(p_below * 100),
                "prob_perder_mitad_margen": round(p_half * 100)})
    if margen <= 0:
        sug, msg = 0, (f"Hoy el precio de la nueva cosecha ({fwd:.0f} USD/ton) no cubre tu costo ({be:.0f}). "
                       f"Fijar ahora asegura pérdida: conviene esperar.")
    else:
        # Más riesgo de quedar debajo del costo → más conviene asegurar.
        # Piso 30% del tope (disciplina de venta escalonada), techo = tope.
        # Más riesgo de resignar margen → más conviene asegurar. Piso 30% del
        # tope (disciplina de venta escalonada), techo = tope.
        frac = min(1.0, max(0.3, p_half / 0.30))
        sug = int(round(tope * frac * 100 / 5) * 5)
        msg = (f"Podés asegurar {margen:+.0f} USD/ton ({margen/be*100:+.0f}% sobre tu costo). "
               f"Si esperás a la cosecha: {p_half*100:.0f}% de probabilidad de perder más de la mitad de ese "
               f"margen y {p_below*100:.0f}% de quedar debajo de tu costo. "
               f"Sugerido: fijar hasta {sug}% de tu producción esperada ahora "
               f"(tope {int(tope*100)}% por riesgo de producción{' — La Niña' if tope < 0.5 else ''}).")
    prod = (profile or {}).get("produccion_total_ton")
    out.update({"sugerido_pct": sug, "sugerido_ton": round(prod * sug / 100) if prod else None, "mensaje": msg})
    return out


# ── 5. ¿Qué viene? ──────────────────────────────────────────────────────────
_EVENT_IMPACT = {
    "ACREAGE": ("Informe de área sembrada EE.UU.", 1.46),
    "STOCKS":  ("Stocks trimestrales EE.UU.", 1.28),
    "WASDE":   ("Informe WASDE", 0.95),
}


def upcoming_events(n: int = 4) -> list[dict]:
    try:
        from src.data.event_calendar import build_event_calendar
        cal = build_event_calendar(date.today().year, date.today().year + 1)
    except Exception:
        return []
    out = []
    for e in sorted(cal, key=lambda x: x["date"]):
        d = pd.Timestamp(e["date"]).date()
        if d < date.today() or e["kind"] not in _EVENT_IMPACT:
            continue
        name, mult = _EVENT_IMPACT[e["kind"]]
        out.append({"fecha": d.isoformat(), "evento": name, "movimiento_vs_dia_normal": mult,
                    "relevante": mult >= 1.2})
        if len(out) >= n:
            break
    return out


# ── API pública ─────────────────────────────────────────────────────────────
def build_decision(profile: dict | None = None, costs: dict | None = None,
                   gastos_usd_ton: float = 0.0) -> dict:
    if profile is None:
        try:
            from src.producer.producer_profile import load_profile
            profile = load_profile()
        except Exception:
            profile = None
    costs = {**DEFAULT_COSTS, **(costs or {})}
    curve = get_futures_curve()
    risk = RiskModel()
    local = _local_reference()
    out = {"ok": True, "generado": datetime.now().isoformat(timespec="seconds"),
           "supuestos_costos": costs, "precio_local": local, "curva": curve,
           "vol_anual_pct": round(risk.sigma_now * np.sqrt(252) * 100, 1)}

    # Guardar o vender. Revista Verde cotiza por campaña: si cotiza la NUEVA
    # cosecha, ese precio es para entrega a cosecha (se compara con el futuro
    # de mayo); si cotiza la vieja, es el disponible (vs el contrato más cercano).
    today = date.today()
    hy = today.year + (1 if today.month > HARVEST_MONTH_UY else 0)
    hi = next((i for i, c in enumerate(curve) if (c["anio"], c["mes"]) == (hy, HARVEST_MONTH_UY)), None)
    lp = local.get("local_usd_ton")
    if lp and curve:
        lp = float(lp)
        if local.get("campania") == f"{hy-1}/{hy}" and hi is not None:
            basis = lp - curve[hi]["usd_ton"]
            local_harvest, spot, spot_est = lp, curve[0]["usd_ton"] + basis, True
        else:
            basis = lp - curve[0]["usd_ton"]
            spot, spot_est = lp, False
            local_harvest = curve[hi]["usd_ton"] + basis if hi is not None else None
        out["base_usada_usd_ton"] = round(basis, 1)

        g = storage_analysis(spot, curve, risk, costs)
        g["texto"] = _storage_text(g, "Si tenés soja guardada: ")
        g["precio_estimado"] = spot_est
        if spot_est:
            g["nota"] = ("No hay cotización local de soja vieja; precio disponible estimado con "
                         "el contrato más cercano y la base de la nueva cosecha.")
        out["guardar_hoy"] = g
        if local_harvest is not None:
            sc = storage_analysis(local_harvest, curve, risk, costs, start_idx=hi,
                                  start_days=_days_until(hy, HARVEST_MONTH_UY))
            sc["texto"] = _storage_text(sc, f"Si cosechás en {MESES[HARVEST_MONTH_UY]} {hy}: ")
            sc["nota"] = "Con la curva de hoy; se actualiza todos los días."
            out["guardar_en_cosecha"] = sc

    out["fijar_precio"] = pricing_analysis(profile, local, curve, risk, gastos_usd_ton)
    out["eventos"] = upcoming_events()
    return out


if __name__ == "__main__":
    print(json.dumps(build_decision(), indent=1, ensure_ascii=False, default=str))
