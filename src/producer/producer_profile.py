"""
src/producer/producer_profile.py
Perfil "Mi Campo" — los pocos números del productor que personalizan todo.

El productor carga UNA vez (onboarding ~30s):
  - hectáreas sembradas
  - rinde esperado (ton/ha)
  - costo de producción (en USD/ton o USD/ha)

Con eso el sistema calcula su PRODUCCIÓN TOTAL y su BREAK-EVEN, y puede
traducir cada precio de mercado en MARGEN concreto (USD/ton, % y plata
total sobre su cosecha). Es el gancho comercial: el productor deja de ver
un precio abstracto y ve "estoy ganando $X por hectárea".

Persistencia: JSON único (MVP single-tenant). A futuro se puede keyear por
producer_id para multi-usuario.
"""

import os
import json
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROFILE_PATH = os.path.join(_PROJECT_ROOT, "data", "producer_profile.json")


def load_profile() -> dict | None:
    """Carga el perfil del productor. None si no fue configurado todavía."""
    if not os.path.exists(_PROFILE_PATH):
        return None
    try:
        with open(_PROFILE_PATH, "r", encoding="utf-8") as f:
            p = json.load(f)
        return p if p else None
    except Exception:
        return None


def save_profile(hectareas: float, rinde_ton_ha: float,
                 costo: float, costo_mode: str = "ton",
                 campania: str | None = None) -> dict:
    """
    Guarda/actualiza el perfil. Valida y deriva campos.

    costo_mode: "ton" (USD/ton) o "ha" (USD/ha). Si es "ha", el break-even
    por tonelada se deriva dividiendo por el rinde.
    """
    hectareas    = max(0.0, float(hectareas))
    rinde_ton_ha = max(0.0, float(rinde_ton_ha))
    costo        = max(0.0, float(costo))
    costo_mode   = costo_mode if costo_mode in ("ton", "ha") else "ton"

    # Break-even por tonelada (el número que se compara con el precio)
    if costo_mode == "ha":
        break_even_usd_ton = (costo / rinde_ton_ha) if rinde_ton_ha > 0 else None
        costo_usd_ha = costo
    else:
        break_even_usd_ton = costo
        costo_usd_ha = costo * rinde_ton_ha if rinde_ton_ha > 0 else None

    produccion_total_ton = hectareas * rinde_ton_ha

    profile = {
        "hectareas":            round(hectareas, 1),
        "rinde_ton_ha":         round(rinde_ton_ha, 2),
        "costo_input":          round(costo, 2),
        "costo_mode":           costo_mode,
        "break_even_usd_ton":   round(break_even_usd_ton, 1) if break_even_usd_ton else None,
        "costo_usd_ha":         round(costo_usd_ha, 1) if costo_usd_ha else None,
        "produccion_total_ton": round(produccion_total_ton, 1),
        "campania":             campania,
        "updated_at":           datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(_PROFILE_PATH), exist_ok=True)
    with open(_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return profile


def compute_margin(profile: dict, precio_neto_usd_ton: float,
                   precio_local_usd_ton: float | None = None,
                   uyu_rate: float | None = None) -> dict | None:
    """
    Traduce un precio en MARGEN concreto sobre la cosecha del productor.

    Compara el precio NETO (lo que realmente embolsa, ya descontado flete y
    gastos) contra el break-even (costo de producción). Ese es el margen real.

    Retorna:
      break_even_usd_ton, margen_usd_ton, margen_pct, en_ganancia (bool),
      margen_total_usd (sobre toda la producción), margen_usd_ha,
      precio_para_break_even (qué precio neto necesita para empatar),
      semaforo (verde/amarillo/rojo) + mensaje.
    """
    if not profile:
        return None
    be = profile.get("break_even_usd_ton")
    if be is None or be <= 0:
        return None

    prod_total = profile.get("produccion_total_ton") or 0
    rinde      = profile.get("rinde_ton_ha") or 0

    margen_usd_ton = precio_neto_usd_ton - be
    margen_pct     = (margen_usd_ton / be * 100) if be > 0 else None
    margen_total   = margen_usd_ton * prod_total if prod_total else None
    margen_usd_ha  = margen_usd_ton * rinde if rinde else None

    # Semáforo de margen
    if margen_pct is None:
        semaforo, msg = "gray", "Sin datos suficientes para calcular el margen."
    elif margen_pct >= 15:
        semaforo = "green"
        msg = f"Estás ganando bien: {margen_pct:+.0f}% sobre tu costo. Buen momento para asegurar margen."
    elif margen_pct >= 3:
        semaforo = "green"
        msg = f"Estás en ganancia: {margen_pct:+.0f}% sobre tu costo."
    elif margen_pct >= -3:
        semaforo = "yellow"
        msg = f"Estás cerca del equilibrio ({margen_pct:+.0f}%). El precio apenas cubre tu costo."
    else:
        semaforo = "red"
        msg = f"Estás por debajo de tu costo ({margen_pct:+.0f}%). Vender hoy realiza una pérdida."

    return {
        "break_even_usd_ton":     round(be, 1),
        "precio_neto_usd_ton":    round(precio_neto_usd_ton, 1),
        "margen_usd_ton":         round(margen_usd_ton, 1),
        "margen_pct":             round(margen_pct, 1) if margen_pct is not None else None,
        "margen_total_usd":       round(margen_total, 0) if margen_total is not None else None,
        "margen_usd_ha":          round(margen_usd_ha, 1) if margen_usd_ha is not None else None,
        "margen_total_uyu":       round(margen_total * uyu_rate, 0) if (margen_total is not None and uyu_rate) else None,
        "produccion_total_ton":   prod_total,
        "en_ganancia":            (margen_usd_ton > 0),
        "semaforo":               semaforo,
        "mensaje":                msg,
        # margen también al precio local (FOB), informativo
        "margen_local_usd_ton":   round(precio_local_usd_ton - be, 1) if precio_local_usd_ton else None,
    }
