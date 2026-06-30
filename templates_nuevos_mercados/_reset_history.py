"""
_reset_history.py — Limpia el historial previo a las mejoras del 2026-05-26.

Acciones:
  1. Renombra paper_trades.jsonl  → paper_trades_pre_fix_20260526.jsonl  (backup)
  2. Crea paper_trades.jsonl vacío (historial limpio desde cero)
  3. Limpia live_state.json:
       - borra paper_position (posición paper fantasma)
       - borra signal_cooldown_until (cooldown obsoleto)
       - resetea llm_calls / cycles a 0 (contador limpio)
       - conserva date para que el rate-limit diario funcione bien

Ejecutar UNA sola vez:  py -3 _reset_history.py
No toca: live_log.jsonl, live_signal.json, robot.pid
"""

import json, os, shutil
from datetime import date
from pathlib import Path

HERE      = Path(__file__).parent
BASE      = HERE.parent / "artifacts"
SUFFIX    = "pre_fix_20260526"

INSTRUMENTS = [
    "btcusd", "ethusd", "us500", "ustec",
    "us30", "wti_n6", "brent_n6", "uk100", "xauusd",
]

def reset_instrument(sym: str):
    art = BASE / sym
    if not art.exists():
        print(f"  {sym:12s}  (sin carpeta artifacts — skip)")
        return

    results = []

    # ── 1. paper_trades.jsonl ──────────────────────────────────────────────────
    pt = art / "paper_trades.jsonl"
    if pt.exists():
        n_trades = sum(1 for l in pt.read_text(encoding="utf-8").splitlines() if l.strip())
        backup   = art / f"paper_trades_{SUFFIX}.jsonl"
        shutil.copy2(pt, backup)            # copia → backup
        pt.write_text("", encoding="utf-8") # vacía el original
        results.append(f"paper_trades: {n_trades} trades -> backup + reset")
    else:
        results.append("paper_trades: no existía")

    # ── 2. live_state.json ─────────────────────────────────────────────────────
    ls = art / "live_state.json"
    if ls.exists():
        try:
            state = json.loads(ls.read_text(encoding="utf-8"))
        except Exception:
            state = {}

        cleaned = []
        for key in ("paper_position", "signal_cooldown_until"):
            if key in state:
                del state[key]
                cleaned.append(key)

        # Reset contadores diarios
        state["llm_calls"] = 0
        state["cycles"]    = 0
        state["date"]      = date.today().isoformat()

        ls.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        results.append(f"live_state: reset ({', '.join(cleaned) or 'solo contadores'})")
    else:
        results.append("live_state: no existía")

    print(f"  {sym:12s}  " + " | ".join(results))


def main():
    print("=" * 65)
    print("_reset_history.py — limpieza pre-fix 2026-05-26")
    print("=" * 65)
    for sym in INSTRUMENTS:
        reset_instrument(sym)
    print()
    print("Listo. Backups guardados como paper_trades_pre_fix_20260526.jsonl")
    print("Historial limpio. Los robots arrancan con slate en blanco.")


if __name__ == "__main__":
    main()
