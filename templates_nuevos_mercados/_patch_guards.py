"""
Patch all live_runner.py files to add:
  B) MT5 open-position guard  — skip LLM if a position is already open
  C) Post-signal cooldown     — skip LLM for 2 cycles after every non-FLAT signal

Run once:  py -3 _patch_guards.py
"""

import os, re

HERE = os.path.dirname(os.path.abspath(__file__))

INSTRUMENTS = [
    "XAUUSD","BTCUSD","ETHUSD","US500","USTEC",
    "US30","WTI_N6","BRENT_N6","UK100",
]

# ── Block to INSERT after the LLM gate early-return ──────────────────────────
GUARD_BLOCK = '''
    # ── B. MT5 position guard ─────────────────────────────────────────────────
    # If there is already an open position for this instrument, skip new signal.
    try:
        from mt5_bridge import get_positions as _get_pos
        _open = [p for p in (_get_pos(SYMBOL) or []) if p.get("magic") == OUR_MAGIC]
        if _open:
            _p = _open[0]
            print(f"[POSITION GUARD] Already open: {_p.get('type','?')} #{_p.get('ticket','?')} "
                  f"@ {_p.get('price_open','?')} | P&L: {_p.get('profit',0):.2f} — skip LLM")
            _log("position_guard", {"ticket": _p.get("ticket"), "profit": _p.get("profit", 0)})
            return {"status": "position_open", "ticket": _p.get("ticket")}
    except Exception as _pg_err:
        print(f"[POSITION GUARD] check failed: {_pg_err}")

    # ── C. Post-signal cooldown ───────────────────────────────────────────────
    # After any non-FLAT signal, wait 2 × CYCLE_MINUTES before new signal.
    _cd_until = state.get("signal_cooldown_until")
    if _cd_until:
        try:
            from datetime import datetime as _dt
            _cd_dt = _dt.fromisoformat(_cd_until)
            if _dt.now() < _cd_dt:
                _mins_cd = int((_cd_dt - _dt.now()).total_seconds() / 60)
                print(f"[COOLDOWN] {_mins_cd} min remaining after last signal — skip LLM")
                _log("cooldown", {"cooldown_until": _cd_until, "mins_left": _mins_cd})
                return {"status": "cooldown", "cooldown_until": _cd_until}
        except Exception:
            pass
'''

# ── Replacement for the state-save block after signal synthesis ───────────────
# We add cooldown_until when signal is non-FLAT.
# Pattern handles two styles:
#   Style A (single-line): state["llm_calls"] += 2; state["cycles"] += 1; ...
#   Style B (multi-line):  state["llm_calls"] += 2\n    state["cycles"] += 1\n ...

SAVE_PATCH = '''
    # set post-signal cooldown (Option C)
    if signal["signal"] != "FLAT":
        from datetime import datetime as _dt2, timedelta as _td2
        state["signal_cooldown_until"] = (_dt2.now() + _td2(minutes=CYCLE_MINUTES * 2)).isoformat()
    else:
        state.pop("signal_cooldown_until", None)
    _save_state(state)'''


def patch_file(path):
    with open(path, encoding="utf-8") as f:
        src = f.read()

    original = src

    # ── 1. Add get_positions to import (if not already there) ─────────────────
    if "get_positions" not in src:
        # Match the mt5_bridge import block and append get_positions
        src = re.sub(
            r"(from mt5_bridge import \([^)]*?)(place_order,)",
            r"\1get_positions,\n    place_order,",
            src,
            count=1,
            flags=re.DOTALL,
        )

    # ── 2. Insert guard block after LLM gate ──────────────────────────────────
    # Marker: the two lines right after `if not can_call:`
    GATE_MARKER_A = '        print(f"LLM gate: {reason}"); return {"status":"llm_gate","reason":reason}'
    GATE_MARKER_B = '        print(f"LLM gate: {reason}")\n        return {"status": "llm_gate", "reason": reason}'

    if "[POSITION GUARD]" not in src:          # skip if already patched
        for marker in (GATE_MARKER_A, GATE_MARKER_B):
            if marker in src:
                src = src.replace(marker, marker + GUARD_BLOCK, 1)
                break

    # ── 3. Insert cooldown setter right after last_signal_time assignment ────────
    # All styles share this line — use it as anchor and replace the following
    # _save_state(state) call (the one in the signal synthesis block).
    ANCHOR = 'state["last_signal_time"] = signal["timestamp"]'
    if 'state["signal_cooldown_until"] =' not in src and ANCHOR in src:
        # Replace only the first _save_state(state) that comes after the anchor
        idx = src.index(ANCHOR)
        save_idx = src.index("    _save_state(state)", idx)
        src = src[:save_idx] + SAVE_PATCH + src[save_idx + len("    _save_state(state)"):]

    if src != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        return True
    return False


def main():
    print("Patching live_runner.py files...\n")
    ok = fail = skip = 0
    for sym in INSTRUMENTS:
        path = os.path.join(HERE, sym, "live_runner.py")
        if not os.path.exists(path):
            print(f"  {sym:12s} MISSING — skipped")
            skip += 1
            continue
        try:
            changed = patch_file(path)
            if changed:
                # Quick sanity check
                with open(path, encoding="utf-8") as f: content = f.read()
                has_guard    = "[POSITION GUARD]" in content
                has_cooldown = "signal_cooldown_until" in content
                status = "OK" if (has_guard and has_cooldown) else "PARTIAL"
                print(f"  {sym:12s} PATCHED  guard={has_guard} cooldown={has_cooldown}  [{status}]")
                ok += 1
            else:
                print(f"  {sym:12s} already patched — skipped")
                skip += 1
        except Exception as e:
            print(f"  {sym:12s} ERROR: {e}")
            fail += 1

    print(f"\nDone: {ok} patched, {skip} skipped, {fail} errors")


if __name__ == "__main__":
    main()
