"""
src/intel/market_hours.py
Central gate for Anthropic API calls — limits LLM operations to 1x/day
during soybean market hours to control costs.

CBOT Soybean Futures (ZS):
  - Electronic trading: Sun 7pm–Fri 1:20pm CT (nearly 24h)
  - Pit session: Mon–Fri 9:30am–1:15pm CT
  - Relevant window for daily analysis: Mon–Fri during US business hours

We define "market open" as Mon–Fri, 08:00–18:00 CT (America/Chicago).
This covers the full pit session + pre/post-market action.

Cost control: each LLM component checks `can_run_llm(component_name)`.
Returns True only ONCE per calendar day per component, and only on
weekdays during market hours. Subsequent calls that day return False.
"""

import json
import os
from datetime import datetime, date

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GATE_PATH = os.path.join(_PROJECT_ROOT, "data", "llm_run_gate.json")


def _now_ct() -> datetime:
    """Current time in US Central (CBOT timezone)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Chicago"))
    except Exception:
        # Fallback: UTC-5 approximation (CT without DST awareness)
        from datetime import timezone, timedelta
        ct = timezone(timedelta(hours=-5))
        return datetime.now(ct)


def _load_gate() -> dict:
    if not os.path.exists(_GATE_PATH):
        return {}
    try:
        with open(_GATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_gate(gate: dict) -> None:
    os.makedirs(os.path.dirname(_GATE_PATH), exist_ok=True)
    with open(_GATE_PATH, "w", encoding="utf-8") as f:
        json.dump(gate, f, ensure_ascii=False, indent=2)


def is_market_open() -> bool:
    """True if CBOT soybean market is open (Mon–Fri, 08:00–18:00 CT)."""
    now = _now_ct()
    # 0=Monday, 6=Sunday
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    return 8 <= now.hour < 18


def can_run_llm(component: str) -> bool:
    """
    Returns True if this LLM component should run NOW.

    Rules:
      1. Must be a weekday (Mon–Fri)
      2. Must be during market hours (08:00–18:00 CT)
      3. Must not have already run today

    Components: "news_analyst", "market_synthesis", "intelligence_engine",
                "weekly_brief", "market_synthesis_trader"
    """
    now = _now_ct()

    # Rule 1 & 2: market hours only
    if not is_market_open():
        return False

    # Rule 3: once per day per component
    gate = _load_gate()
    today = now.date().isoformat()
    last_run = gate.get(component, {}).get("last_date")

    if last_run == today:
        return False  # Already ran today

    return True


def mark_llm_ran(component: str) -> None:
    """Record that this component ran today. Call after successful execution."""
    now = _now_ct()
    gate = _load_gate()
    gate[component] = {
        "last_date": now.date().isoformat(),
        "last_time": now.strftime("%H:%M:%S"),
    }
    _save_gate(gate)


def get_gate_status() -> dict:
    """Returns current gate status for all components (for dashboard/debug)."""
    now = _now_ct()
    gate = _load_gate()
    today = now.date().isoformat()

    return {
        "market_open": is_market_open(),
        "current_time_ct": now.strftime("%Y-%m-%d %H:%M CT"),
        "weekday": now.strftime("%A"),
        "components": {
            comp: {
                "last_run": info.get("last_date"),
                "last_time": info.get("last_time"),
                "ran_today": info.get("last_date") == today,
                "can_run": can_run_llm(comp),
            }
            for comp, info in gate.items()
        },
    }


if __name__ == "__main__":
    print(f"Market open: {is_market_open()}")
    print(f"Time CT: {_now_ct().strftime('%Y-%m-%d %H:%M CT (%A)')}")
    for comp in ["news_analyst", "market_synthesis", "intelligence_engine", "weekly_brief"]:
        print(f"  {comp}: can_run={can_run_llm(comp)}")
