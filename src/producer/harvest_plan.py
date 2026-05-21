"""
src/producer/harvest_plan.py
Plan de Cosecha Inteligente — Pre-commitment selling plan for soybean producers.

Academic basis:
  - Vollmer et al. (2019, Agricultural Economics): Pre-commitment is the #1
    intervention against disposition effect in farmers.
  - Irwin & Good (2006, AJAE): Disciplined layered selling adds $8-12/acre
    vs actual farmer behavior.
  - Tomek & Peterson (2001, J. Futures Markets): Systematic plans outperform
    discretionary timing.

How it works:
  1. Producer defines: total tons, cost of production, price targets
  2. System generates a layered plan with 5 tranches (configurable)
  3. Each tranche has a trigger type:
     - price_target: fires when local price ≥ threshold
     - seasonal_window: fires during historically favorable months
     - model_signal: fires when semáforo = VENDER
     - time_deadline: fires on a specific date (e.g., end of campaign)
     - manual: producer decides
  4. When a trigger fires → Telegram/WhatsApp alert
  5. Producer confirms execution → system logs and tracks weighted avg price

Persistence: data/harvest_plans.json (array of plans, one active per user)
"""

import json
import os
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PLANS_PATH = os.path.join(_PROJECT_ROOT, "data", "harvest_plans.json")
_HISTORY_PATH = os.path.join(_PROJECT_ROOT, "data", "harvest_plan_history.json")

# Bushels per metric ton
BUSHELS_PER_TON = 36.744

# Default tranche template (disciplined 5-tranche plan from Tomek & Peterson)
DEFAULT_TRANCHES = [
    {"pct": 20, "trigger": "price_target", "label": "Primer objetivo", "description": "Primer tramo: asegurar piso de rentabilidad"},
    {"pct": 20, "trigger": "price_target", "label": "Segundo objetivo", "description": "Segundo tramo: capturar rally intermedio"},
    {"pct": 20, "trigger": "seasonal_window", "label": "Ventana estacional", "description": "Venta en ventana históricamente favorable (Mar-May)"},
    {"pct": 20, "trigger": "model_signal", "label": "Señal AgroCast", "description": "Venta cuando el semáforo marca VENDER"},
    {"pct": 20, "trigger": "time_deadline", "label": "Cierre de campaña", "description": "Último tramo antes de fin de campaña (evita deterioro)"},
]

# Historically favorable months for selling soybeans (Southern Hemisphere harvest: Mar-Jun)
FAVORABLE_MONTHS = {3, 4, 5}  # Mar, Apr, May — peak harvest + demand


def _load_plans() -> list:
    if not os.path.exists(_PLANS_PATH):
        return []
    try:
        with open(_PLANS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_plans(plans: list) -> None:
    os.makedirs(os.path.dirname(_PLANS_PATH), exist_ok=True)
    with open(_PLANS_PATH, "w", encoding="utf-8") as f:
        json.dump(plans, f, ensure_ascii=False, indent=2, default=str)


def _load_history() -> list:
    if not os.path.exists(_HISTORY_PATH):
        return []
    try:
        with open(_HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(history: list) -> None:
    os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
    with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history[-50:], f, ensure_ascii=False, indent=2, default=str)


def create_plan(
    crop_tons: float,
    cost_of_production_usd_ton: float,
    target_avg_price_usd_ton: Optional[float] = None,
    campaign: Optional[str] = None,
    custom_tranches: Optional[list] = None,
    storage_cost_pct_annual: float = 6.0,
    financing_rate_pct_annual: float = 0.0,
) -> dict:
    """
    Create a new harvest selling plan.

    Args:
        crop_tons: Total expected harvest in metric tons
        cost_of_production_usd_ton: Break-even cost per ton
        target_avg_price_usd_ton: Desired average selling price (optional)
        campaign: Campaign name, e.g. "2025/26" (auto-generated if None)
        custom_tranches: Override default 5-tranche plan
        storage_cost_pct_annual: Annual storage cost as % of grain value
        financing_rate_pct_annual: Annual financing rate for stored grain

    Returns:
        The created plan dict
    """
    plans = _load_plans()

    # Deactivate existing active plans
    for p in plans:
        if p.get("active"):
            p["active"] = False

    if campaign is None:
        year = date.today().year
        month = date.today().month
        campaign = f"{year}/{year+1}" if month >= 4 else f"{year-1}/{year}"

    # Build tranches from template
    template = custom_tranches if custom_tranches else DEFAULT_TRANCHES
    tranches = []
    for i, t in enumerate(template):
        tranche = {
            "id": i + 1,
            "pct": t.get("pct", 20),
            "tons": round(crop_tons * t.get("pct", 20) / 100, 1),
            "trigger": t.get("trigger", "manual"),
            "label": t.get("label", f"Tramo {i+1}"),
            "description": t.get("description", ""),
            # Trigger-specific config
            "price_target_usd_ton": t.get("price_target_usd_ton"),
            "target_month": t.get("target_month"),
            "deadline_date": t.get("deadline_date"),
            # Execution tracking
            "status": "pending",  # pending, triggered, executed, skipped
            "triggered_at": None,
            "executed_at": None,
            "execution_price_usd_ton": None,
            "alert_sent": False,
        }
        tranches.append(tranche)

    # Auto-calculate price targets if target_avg_price given
    if target_avg_price_usd_ton and not custom_tranches:
        _auto_set_price_targets(tranches, cost_of_production_usd_ton,
                                target_avg_price_usd_ton)

    plan = {
        "id": f"plan_{date.today().isoformat()}_{len(plans)+1}",
        "campaign": campaign,
        "created_at": datetime.now().isoformat(),
        "active": True,
        "crop_tons": crop_tons,
        "cost_of_production_usd_ton": round(cost_of_production_usd_ton, 2),
        "target_avg_price_usd_ton": round(target_avg_price_usd_ton, 2) if target_avg_price_usd_ton else None,
        "storage_cost_pct_annual": storage_cost_pct_annual,
        "financing_rate_pct_annual": financing_rate_pct_annual,
        "tranches": tranches,
        "metrics": {
            "tons_sold": 0,
            "tons_pending": crop_tons,
            "weighted_avg_price": None,
            "total_revenue_usd": 0,
            "vs_cost_of_production_pct": None,
            "vs_always_sell_pct": None,
        },
    }

    plans.append(plan)
    _save_plans(plans)
    return plan


def _auto_set_price_targets(tranches: list, cost: float, target: float):
    """Auto-distribute price targets across tranches.
    Strategy: target from break-even+margin to target+upside.
    """
    margin = target - cost
    if margin <= 0:
        margin = cost * 0.05  # 5% minimum

    for t in tranches:
        if t["trigger"] == "price_target":
            if t["id"] == 1:
                # First target: cost + 50% of margin (secure profitability)
                t["price_target_usd_ton"] = round(cost + margin * 0.5, 2)
            elif t["id"] == 2:
                # Second target: target price (the goal)
                t["price_target_usd_ton"] = round(target, 2)
        elif t["trigger"] == "seasonal_window":
            t["target_month"] = 4  # April default
        elif t["trigger"] == "time_deadline":
            # End of campaign: June 30 of current year
            year = date.today().year
            if date.today().month >= 7:
                year += 1
            t["deadline_date"] = f"{year}-06-30"


def check_triggers(current_price_usd_ton: float, sell_signal: str = "ESPERAR") -> dict:
    """
    Check all tranches of the active plan against current conditions.
    Called periodically (e.g., daily in pipeline or on API request).

    Args:
        current_price_usd_ton: Current local soybean price
        sell_signal: Current semáforo signal (VENDER/ESPERAR/RETENER)

    Returns:
        Dict with triggered tranches and alerts to send
    """
    plans = _load_plans()
    active = next((p for p in plans if p.get("active")), None)
    if not active:
        return {"ok": False, "reason": "no_active_plan"}

    today = date.today()
    today_month = today.month
    alerts = []
    modified = False

    for tranche in active["tranches"]:
        if tranche["status"] != "pending":
            continue

        triggered = False
        reason = ""

        if tranche["trigger"] == "price_target":
            target = tranche.get("price_target_usd_ton")
            if target and current_price_usd_ton >= target:
                triggered = True
                reason = f"Precio actual ${current_price_usd_ton:.0f}/ton ≥ objetivo ${target:.0f}/ton"

        elif tranche["trigger"] == "seasonal_window":
            target_month = tranche.get("target_month")
            if target_month:
                if today_month == target_month:
                    triggered = True
                    reason = f"Ventana estacional activa (mes {today_month})"
            elif today_month in FAVORABLE_MONTHS:
                triggered = True
                reason = f"Ventana estacional favorable (Mar-May)"

        elif tranche["trigger"] == "model_signal":
            if sell_signal == "VENDER":
                triggered = True
                reason = f"Semáforo AgroCast: VENDER"

        elif tranche["trigger"] == "time_deadline":
            deadline = tranche.get("deadline_date")
            if deadline:
                dl = date.fromisoformat(deadline)
                days_left = (dl - today).days
                if days_left <= 0:
                    triggered = True
                    reason = f"Plazo cumplido: {deadline}"
                elif days_left <= 7:
                    # Pre-alert: deadline approaching
                    if not tranche.get("alert_sent"):
                        alerts.append({
                            "tranche_id": tranche["id"],
                            "type": "deadline_approaching",
                            "message": f"⏰ Tramo {tranche['id']} ({tranche['label']}): "
                                       f"faltan {days_left} días para el cierre de campaña. "
                                       f"{tranche['tons']} ton pendientes.",
                            "days_left": days_left,
                        })

        if triggered:
            tranche["status"] = "triggered"
            tranche["triggered_at"] = datetime.now().isoformat()
            modified = True
            alerts.append({
                "tranche_id": tranche["id"],
                "type": "trigger_fired",
                "message": (
                    f"🌾 PLAN DE COSECHA — Tramo {tranche['id']} activado\n\n"
                    f"📋 {tranche['label']}: {tranche['description']}\n"
                    f"📦 Toneladas: {tranche['tons']:.0f} ton ({tranche['pct']}% de cosecha)\n"
                    f"💰 Precio actual: ${current_price_usd_ton:.0f}/ton\n"
                    f"📌 Razón: {reason}\n\n"
                    f"¿Confirmar venta? Responde /confirmar_{tranche['id']} o /posponer_{tranche['id']}"
                ),
                "tons": tranche["tons"],
                "price": current_price_usd_ton,
                "reason": reason,
            })

    if modified:
        _save_plans(plans)

    return {
        "ok": True,
        "plan_id": active["id"],
        "alerts": alerts,
        "plan_summary": get_plan_summary(active),
    }


def confirm_execution(tranche_id: int, actual_price_usd_ton: float) -> dict:
    """
    Confirm that a triggered tranche was executed at a given price.
    Updates plan metrics (weighted avg price, revenue, etc.)
    """
    plans = _load_plans()
    active = next((p for p in plans if p.get("active")), None)
    if not active:
        return {"ok": False, "error": "no_active_plan"}

    tranche = next((t for t in active["tranches"] if t["id"] == tranche_id), None)
    if not tranche:
        return {"ok": False, "error": f"tranche {tranche_id} not found"}

    if tranche["status"] not in ("triggered", "pending"):
        return {"ok": False, "error": f"tranche {tranche_id} status={tranche['status']}, cannot confirm"}

    # Mark executed
    tranche["status"] = "executed"
    tranche["executed_at"] = datetime.now().isoformat()
    tranche["execution_price_usd_ton"] = round(actual_price_usd_ton, 2)

    # Recalculate metrics
    _recalc_metrics(active)

    _save_plans(plans)

    # Log to history
    history = _load_history()
    history.append({
        "plan_id": active["id"],
        "tranche_id": tranche_id,
        "date": date.today().isoformat(),
        "tons": tranche["tons"],
        "price_usd_ton": round(actual_price_usd_ton, 2),
        "campaign": active["campaign"],
    })
    _save_history(history)

    return {
        "ok": True,
        "tranche": tranche,
        "metrics": active["metrics"],
    }


def skip_tranche(tranche_id: int, reason: str = "") -> dict:
    """Skip/postpone a triggered tranche."""
    plans = _load_plans()
    active = next((p for p in plans if p.get("active")), None)
    if not active:
        return {"ok": False, "error": "no_active_plan"}

    tranche = next((t for t in active["tranches"] if t["id"] == tranche_id), None)
    if not tranche:
        return {"ok": False, "error": f"tranche {tranche_id} not found"}

    tranche["status"] = "pending"  # Reset to pending so it can retrigger
    tranche["triggered_at"] = None
    tranche["alert_sent"] = False
    if reason:
        tranche["skip_reason"] = reason

    _save_plans(plans)
    return {"ok": True, "tranche": tranche}


def _recalc_metrics(plan: dict):
    """Recalculate plan metrics after an execution."""
    executed = [t for t in plan["tranches"] if t["status"] == "executed"]
    if not executed:
        plan["metrics"]["tons_sold"] = 0
        plan["metrics"]["weighted_avg_price"] = None
        plan["metrics"]["total_revenue_usd"] = 0
        return

    total_tons = sum(t["tons"] for t in executed)
    total_revenue = sum(t["tons"] * t["execution_price_usd_ton"] for t in executed)
    wavg = total_revenue / total_tons if total_tons > 0 else 0

    plan["metrics"]["tons_sold"] = round(total_tons, 1)
    plan["metrics"]["tons_pending"] = round(plan["crop_tons"] - total_tons, 1)
    plan["metrics"]["weighted_avg_price"] = round(wavg, 2)
    plan["metrics"]["total_revenue_usd"] = round(total_revenue, 2)

    cost = plan.get("cost_of_production_usd_ton", 0)
    if cost > 0:
        plan["metrics"]["vs_cost_of_production_pct"] = round(
            (wavg - cost) / cost * 100, 2
        )


def get_plan_summary(plan: Optional[dict] = None) -> dict:
    """Get a summary of the plan suitable for dashboard/API."""
    if plan is None:
        plans = _load_plans()
        plan = next((p for p in plans if p.get("active")), None)
        if not plan:
            return {"ok": False, "active": False, "message": "Sin plan activo"}

    executed = [t for t in plan["tranches"] if t["status"] == "executed"]
    triggered = [t for t in plan["tranches"] if t["status"] == "triggered"]
    pending = [t for t in plan["tranches"] if t["status"] == "pending"]

    # Progress bar
    pct_sold = (plan["metrics"]["tons_sold"] / plan["crop_tons"] * 100) if plan["crop_tons"] > 0 else 0

    # Carrying cost so far
    carrying_cost = _estimate_carrying_cost(plan)

    return {
        "ok": True,
        "active": True,
        "plan_id": plan["id"],
        "campaign": plan["campaign"],
        "crop_tons": plan["crop_tons"],
        "cost_of_production": plan["cost_of_production_usd_ton"],
        "progress": {
            "pct_sold": round(pct_sold, 1),
            "tons_sold": plan["metrics"]["tons_sold"],
            "tons_pending": plan["metrics"]["tons_pending"],
            "weighted_avg_price": plan["metrics"]["weighted_avg_price"],
            "total_revenue_usd": plan["metrics"]["total_revenue_usd"],
            "vs_cost_pct": plan["metrics"].get("vs_cost_of_production_pct"),
        },
        "tranches": [
            {
                "id": t["id"],
                "label": t["label"],
                "pct": t["pct"],
                "tons": t["tons"],
                "trigger": t["trigger"],
                "status": t["status"],
                "price_target": t.get("price_target_usd_ton"),
                "execution_price": t.get("execution_price_usd_ton"),
                "executed_at": t.get("executed_at"),
            }
            for t in plan["tranches"]
        ],
        "counts": {
            "executed": len(executed),
            "triggered": len(triggered),
            "pending": len(pending),
            "total": len(plan["tranches"]),
        },
        "carrying_cost_usd_ton": carrying_cost,
    }


def _estimate_carrying_cost(plan: dict) -> float:
    """Estimate accumulated carrying cost for unsold tons."""
    created = datetime.fromisoformat(plan["created_at"])
    days_held = (datetime.now() - created).days
    if days_held <= 0:
        return 0.0

    price_ref = plan.get("cost_of_production_usd_ton", 400)
    annual_rate = plan.get("storage_cost_pct_annual", 6.0) / 100
    financing_rate = plan.get("financing_rate_pct_annual", 0) / 100
    total_rate = annual_rate + financing_rate
    daily_cost = price_ref * total_rate / 365
    return round(daily_cost * days_held, 2)


def send_plan_alerts(alerts: list) -> int:
    """Send triggered alerts via Telegram. Returns count of alerts sent."""
    try:
        from src.alerts.telegram_bot import _send_message
    except ImportError:
        print("[HarvestPlan] telegram_bot not available")
        return 0

    sent = 0
    for alert in alerts:
        msg = alert.get("message", "")
        if msg and _send_message(msg):
            sent += 1
            # Mark alert as sent
            plans = _load_plans()
            active = next((p for p in plans if p.get("active")), None)
            if active:
                for t in active["tranches"]:
                    if t["id"] == alert.get("tranche_id"):
                        t["alert_sent"] = True
                _save_plans(plans)

    return sent


def get_plan_performance_vs_benchmark(plan: Optional[dict] = None) -> dict:
    """
    Compare plan performance against benchmarks:
    - always-sell (sell 100% at campaign start)
    - best-price (sell 100% at season high)
    - avg-price (seasonal average)
    """
    if plan is None:
        plans = _load_plans()
        plan = next((p for p in plans if p.get("active")), None)
        if not plan:
            return {"ok": False}

    wavg = plan["metrics"].get("weighted_avg_price")
    if not wavg or plan["metrics"]["tons_sold"] == 0:
        return {"ok": True, "message": "Sin ejecuciones aún", "benchmarks": {}}

    # Load price history for benchmark comparison
    try:
        raw_path = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")
        import pandas as pd
        df = pd.read_csv(raw_path, parse_dates=["Date"])
        df = df.sort_values("Date").tail(120)  # Last ~6 months

        # Convert to USD/ton
        prices_usd_ton = df["Soybeans"].values * BUSHELS_PER_TON / 100  # cents/bu → USD/ton

        avg_price = float(np.mean(prices_usd_ton))
        max_price = float(np.max(prices_usd_ton))
        min_price = float(np.min(prices_usd_ton))
        first_price = float(prices_usd_ton[0])  # "always-sell" = first available

        return {
            "ok": True,
            "plan_weighted_avg": wavg,
            "benchmarks": {
                "always_sell": {
                    "price": round(first_price, 2),
                    "vs_plan_pct": round((wavg - first_price) / first_price * 100, 2),
                },
                "season_average": {
                    "price": round(avg_price, 2),
                    "vs_plan_pct": round((wavg - avg_price) / avg_price * 100, 2),
                },
                "season_high": {
                    "price": round(max_price, 2),
                    "vs_plan_pct": round((wavg - max_price) / max_price * 100, 2),
                },
                "season_low": {
                    "price": round(min_price, 2),
                    "vs_plan_pct": round((wavg - min_price) / min_price * 100, 2),
                },
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
