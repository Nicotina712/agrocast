"""
src/intel/ie_accountability.py
Historial y accountability del Intelligence Engine (debate multi-agente).

Cada vez que el IE produce un veredicto, se guarda un snapshot.
Después de N días, se verifica vs precio real y se calcula el hit rate.

Este módulo provee:
  - save_verdict_snapshot(): guarda el veredicto del debate con precio actual
  - evaluate_verdicts(): verifica veredictos maduros vs precio real
  - get_verdict_history(): retorna historial + stats para dashboard
  - get_feedback_for_debate(): genera texto con track record para inyectar en el debate
"""

import json
import os
from datetime import date, datetime, timedelta

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HISTORY_PATH = os.path.join(_PROJECT_ROOT, "data", "ie_verdict_history.json")
_RAW_PATH = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")

_HORIZON_7D = 7
_HORIZON_14D = 14
_NEUTRAL_BAND = 0.02  # +/-2% = neutral zone


def _load_history() -> list:
    if not os.path.exists(_HISTORY_PATH):
        return []
    try:
        with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(history: list) -> None:
    os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
    with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history[-365:], f, ensure_ascii=False, indent=2)


def _get_price_on_date(target_date: str) -> float | None:
    """Get actual soybean price on or after target_date."""
    if not os.path.exists(_RAW_PATH):
        return None
    try:
        df = pd.read_csv(_RAW_PATH, parse_dates=["Date"]).sort_values("Date")
        target = pd.Timestamp(target_date)
        future = df[df["Date"] >= target]
        if future.empty:
            return None
        return float(future["Soybeans"].iloc[0])
    except Exception:
        return None


def save_verdict_snapshot(debate_result: dict) -> dict | None:
    """
    Guarda un snapshot del veredicto del IE. Idempotente por día.
    Se llama después de cada debate exitoso.
    """
    verdict = debate_result.get("verdict", {})
    if not verdict or verdict.get("error"):
        return None

    today = date.today().isoformat()
    history = _load_history()

    # Dedupe por día
    if any(h["date"] == today for h in history):
        return next(h for h in history if h["date"] == today)

    price = debate_result.get("current_price", 0)
    if not price:
        return None

    # Extract key fields
    snap = {
        "date": today,
        "timestamp": debate_result.get("timestamp", datetime.now().isoformat()),
        "price_at_verdict": round(price, 2),
        "verdict": verdict.get("verdict", "HOLD"),
        "confidence": verdict.get("confidence", 0),
        "reasoning": verdict.get("reasoning", "")[:300],
        "price_range_7d": verdict.get("price_range_7d", {}),
        "price_range_30d": verdict.get("price_range_30d", {}),
        "bull_conviction": debate_result.get("agents", {}).get("bull", {}).get("conviction", "?"),
        "bear_conviction": debate_result.get("agents", {}).get("bear", {}).get("conviction", "?"),
        "position_sizing": verdict.get("position_sizing", "?"),
        # Verification fields (filled later)
        "verified_7d": False,
        "verified_14d": False,
        "price_at_7d": None,
        "price_at_14d": None,
        "return_7d_pct": None,
        "return_14d_pct": None,
        "direction_correct_7d": None,
        "direction_correct_14d": None,
        "range_hit_7d": None,
    }

    history.append(snap)
    _save_history(history)
    print(f"[IE-Accountability] Verdict snapshot saved: {today} | {snap['verdict']} | conf={snap['confidence']}")
    return snap


def evaluate_verdicts() -> int:
    """
    Verifica veredictos con suficiente madurez vs precio real.
    Retorna cuántos verificó.
    """
    history = _load_history()
    if not history:
        return 0

    today = date.today()
    n_done = 0

    for snap in history:
        # Check 7d horizon
        if not snap.get("verified_7d"):
            snap_date = date.fromisoformat(snap["date"])
            if (today - snap_date).days >= _HORIZON_7D:
                target = (snap_date + timedelta(days=_HORIZON_7D)).isoformat()
                actual = _get_price_on_date(target)
                if actual is not None:
                    base = snap["price_at_verdict"]
                    ret = (actual / base - 1.0) * 100
                    snap["verified_7d"] = True
                    snap["price_at_7d"] = round(actual, 2)
                    snap["return_7d_pct"] = round(ret, 2)

                    # Direction check
                    verdict = snap["verdict"]
                    if verdict in ("STRONG_BUY", "BUY"):
                        snap["direction_correct_7d"] = ret > 0
                    elif verdict in ("STRONG_SELL", "SELL"):
                        snap["direction_correct_7d"] = ret < 0
                    else:  # HOLD
                        snap["direction_correct_7d"] = abs(ret) <= _NEUTRAL_BAND * 100

                    # Range check
                    range_7d = snap.get("price_range_7d", {})
                    if range_7d.get("low") and range_7d.get("high"):
                        snap["range_hit_7d"] = range_7d["low"] <= actual <= range_7d["high"]

                    n_done += 1

        # Check 14d horizon
        if not snap.get("verified_14d"):
            snap_date = date.fromisoformat(snap["date"])
            if (today - snap_date).days >= _HORIZON_14D:
                target = (snap_date + timedelta(days=_HORIZON_14D)).isoformat()
                actual = _get_price_on_date(target)
                if actual is not None:
                    base = snap["price_at_verdict"]
                    ret = (actual / base - 1.0) * 100
                    snap["verified_14d"] = True
                    snap["price_at_14d"] = round(actual, 2)
                    snap["return_14d_pct"] = round(ret, 2)
                    n_done += 1

    if n_done:
        _save_history(history)
    return n_done


def get_verdict_history() -> dict:
    """Retorna historial completo + estadísticas para dashboard/API."""
    evaluate_verdicts()  # auto-verify pending
    history = _load_history()

    verified_7d = [h for h in history if h.get("verified_7d")]
    dir_correct = [h for h in verified_7d if h.get("direction_correct_7d")]
    range_hits = [h for h in verified_7d if h.get("range_hit_7d")]

    n_v = len(verified_7d)
    return {
        "ok": True,
        "total_verdicts": len(history),
        "verified_7d": n_v,
        "pending": len(history) - n_v,
        "direction_accuracy_7d": round(len(dir_correct) / n_v * 100, 1) if n_v else None,
        "range_accuracy_7d": round(len(range_hits) / n_v * 100, 1) if n_v else None,
        "avg_confidence": round(sum(h.get("confidence", 0) for h in history) / len(history), 2) if history else None,
        "verdict_distribution": _count_verdicts(history),
        "recent": history[-10:][::-1],
    }


def _count_verdicts(history: list) -> dict:
    counts = {}
    for h in history:
        v = h.get("verdict", "UNKNOWN")
        counts[v] = counts.get(v, 0) + 1
    return counts


def get_feedback_for_debate() -> str:
    """
    Genera texto de feedback para inyectar en el contexto del debate.
    Los agentes ven su track record y pueden calibrar mejor.
    """
    history = _load_history()
    if not history:
        return ""

    evaluate_verdicts()
    history = _load_history()  # reload after evaluation

    verified = [h for h in history if h.get("verified_7d")]
    if not verified:
        # Show recent unverified for context
        recent = history[-5:]
        if not recent:
            return ""
        lines = [
            "\n=== HISTORIAL DE VEREDICTOS PREVIOS (sin verificar aún) ===",
            f"Total veredictos emitidos: {len(history)} | Verificados: 0 (menos de 7 días)",
        ]
        for h in reversed(recent):
            lines.append(
                f"  {h['date']}: {h['verdict']} (conf={h['confidence']}) @ ${h['price_at_verdict']}"
            )
        lines.append(
            "NOTA: Estos veredictos aún no han sido verificados contra el precio real. "
            "No hay feedback disponible todavía.\n"
        )
        return "\n".join(lines)

    # Compute stats
    n_v = len(verified)
    dir_correct = sum(1 for h in verified if h.get("direction_correct_7d"))
    range_hits = sum(1 for h in verified if h.get("range_hit_7d"))
    avg_ret = sum(h.get("return_7d_pct", 0) for h in verified) / n_v if n_v else 0

    # Count by verdict type
    buy_correct = sum(1 for h in verified if h["verdict"] in ("BUY", "STRONG_BUY") and h.get("direction_correct_7d"))
    buy_total = sum(1 for h in verified if h["verdict"] in ("BUY", "STRONG_BUY"))
    sell_correct = sum(1 for h in verified if h["verdict"] in ("SELL", "STRONG_SELL") and h.get("direction_correct_7d"))
    sell_total = sum(1 for h in verified if h["verdict"] in ("SELL", "STRONG_SELL"))
    hold_total = sum(1 for h in verified if h["verdict"] == "HOLD")

    lines = [
        "\n=== TRACK RECORD DEL SISTEMA (feedback verificado) ===",
        f"Total veredictos: {len(history)} | Verificados a 7d: {n_v}",
        f"Accuracy direccional (7d): {dir_correct}/{n_v} = {dir_correct/n_v*100:.0f}%",
        f"Accuracy de rango (7d): {range_hits}/{n_v} = {range_hits/n_v*100:.0f}%" if n_v else "",
        f"Retorno promedio 7d post-veredicto: {avg_ret:+.2f}%",
    ]

    if buy_total:
        lines.append(f"BUY/STRONG_BUY: {buy_correct}/{buy_total} correctos ({buy_correct/buy_total*100:.0f}%)")
    if sell_total:
        lines.append(f"SELL/STRONG_SELL: {sell_correct}/{sell_total} correctos ({sell_correct/sell_total*100:.0f}%)")
    if hold_total:
        lines.append(f"HOLD: {hold_total} veces")

    # Last 5 verified for context
    lines.append("\nÚltimos 5 veredictos verificados:")
    for h in reversed(verified[-5:]):
        hit = "✓" if h.get("direction_correct_7d") else "✗"
        lines.append(
            f"  {h['date']}: {h['verdict']} @ ${h['price_at_verdict']:.0f} → "
            f"7d: {h.get('return_7d_pct', '?'):+.1f}% [{hit}]"
        )

    # Calibration note
    if n_v >= 5:
        if dir_correct / n_v < 0.4:
            lines.append(
                "\n⚠️ CALIBRACIÓN: El track record muestra accuracy por debajo del 40%. "
                "Los agentes deben ser más conservadores y favorecer HOLD cuando la evidencia es ambigua."
            )
        elif dir_correct / n_v > 0.7:
            lines.append(
                "\n✓ CALIBRACIÓN: El track record muestra buena accuracy (>70%). "
                "Pueden mantener el nivel actual de convicción."
            )

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== IE Accountability ===")
    print(f"Evaluate: {evaluate_verdicts()} verified")
    print(json.dumps(get_verdict_history(), indent=2, ensure_ascii=False))
    print("\n--- Feedback for debate ---")
    print(get_feedback_for_debate())
