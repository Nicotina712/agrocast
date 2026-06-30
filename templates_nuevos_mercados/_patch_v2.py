"""
_patch_v2.py — Fix #1 + Fix #2 for all live_runner.py and config.py

Fix #1: Paper position guard — block new LLM signals while a paper trade is
        still open (SL/TP not yet hit).  Tracked in live_state.json.

Fix #2: MIN_SL_PCT enforcement — after synthesis, if SL is tighter than the
        per-instrument minimum %, widen it to the minimum.

Run once:  py -3 _patch_v2.py
"""

import os, re

HERE = os.path.dirname(os.path.abspath(__file__))

PAPER_INSTR = ["BTCUSD", "US500", "USTEC", "US30", "WTI_N6", "XAUUSD"]
LIVE_INSTR  = ["ETHUSD", "BRENT_N6", "UK100"]
ALL_INSTR   = PAPER_INSTR + LIVE_INSTR

# Per-instrument minimum SL as % of entry price
MIN_SL_PCT = {
    "BTCUSD":  0.80,
    "US500":   0.30,
    "USTEC":   0.30,
    "US30":    0.35,
    "WTI_N6":  0.20,
    "XAUUSD":  0.15,
    "ETHUSD":  0.80,
    "BRENT_N6": 0.20,
    "UK100":   0.20,
}

# ─────────────────────────────────────────────────────────────────────────────
# Fix #1 — Paper guard block  (inserted before fund_ctx in run_cycle)
# ─────────────────────────────────────────────────────────────────────────────

PAPER_GUARD = '''
    # ── D. Paper position guard ───────────────────────────────────────────────
    # Block new signals while a paper trade is still open (no real MT5 position
    # exists for paper instruments, so we track open trades in live_state.json).
    if not execute:
        _pp = state.get("paper_position")
        if _pp:
            _pp_sig   = _pp.get("signal")
            _pp_entry = float(_pp.get("entry") or 0)
            _pp_sl    = float(_pp.get("sl")    or 0)
            _pp_tp    = float(_pp.get("tp")    or 0)
            _pp_ts    = str(_pp.get("timestamp", "?"))[:16]
            _cur      = float((summary.get("current_state") or {}).get("close") or 0)
            if _pp_sig == "LONG":
                if _cur >= _pp_tp:    _pp_out = "WIN"
                elif _cur <= _pp_sl:  _pp_out = "LOSS"
                else:                 _pp_out = "OPEN"
            else:
                if _cur <= _pp_tp:    _pp_out = "WIN"
                elif _cur >= _pp_sl:  _pp_out = "LOSS"
                else:                 _pp_out = "OPEN"
            if _pp_out == "OPEN":
                print(f"[PAPER GUARD] {_pp_sig} open since {_pp_ts} | "
                      f"entry={_pp_entry} SL={_pp_sl} TP={_pp_tp} now={_cur:.4f} — skip LLM")
                _log("paper_guard", {"signal": _pp_sig, "entry": _pp_entry,
                                     "sl": _pp_sl, "tp": _pp_tp,
                                     "current_price": _cur, "outcome": "OPEN"})
                return {"status": "paper_position_open"}
            else:
                print(f"[PAPER GUARD] {_pp_sig} closed → {_pp_out} | New signal allowed.")
                _log("paper_guard", {"signal": _pp_sig, "entry": _pp_entry,
                                     "sl": _pp_sl, "tp": _pp_tp,
                                     "current_price": _cur, "outcome": _pp_out})
                state.pop("paper_position", None)
                _save_state(state)

'''

# ─────────────────────────────────────────────────────────────────────────────
# Fix #1 — Paper position save  (inserted before _save_state in synthesis block)
# ─────────────────────────────────────────────────────────────────────────────

PP_SAVE = '''\
    # Track paper position for paper guard (Fix #1)
    if signal["signal"] != "FLAT" and not execute:
        state["paper_position"] = {
            "signal":    signal["signal"],
            "entry":     signal.get("entry"),
            "sl":        signal.get("sl"),
            "tp":        signal.get("tp"),
            "timestamp": signal["timestamp"],
        }
    elif signal["signal"] == "FLAT":
        state.pop("paper_position", None)
'''

# ─────────────────────────────────────────────────────────────────────────────
# Fix #2 — SL enforcement block  (inserted before state["llm_calls"] += 2)
# ─────────────────────────────────────────────────────────────────────────────

def sl_enforce(pct):
    return f"""\

    # ── E. Minimum SL distance enforcement (Fix #2) ──────────────────────────
    if signal["signal"] != "FLAT":
        _e   = float(signal.get("entry") or 0)
        _sl  = float(signal.get("sl")    or 0)
        _slp = abs(_e - _sl) / _e * 100 if _e else 0
        if _slp < {pct}:
            _min_d = _e * {pct} / 100
            if signal["signal"] == "LONG":
                signal["sl"] = round(_e - _min_d, 5)
            else:
                signal["sl"] = round(_e + _min_d, 5)
            signal["sl_enforced"] = True
            print(f"[SL GUARD] SL too tight ({{_slp:.3f}}% < {pct}%). "
                  f"Adjusted: {{_sl:.5f}} -> {{signal['sl']:.5f}}")
"""


# ─────────────────────────────────────────────────────────────────────────────
# config.py patch
# ─────────────────────────────────────────────────────────────────────────────

def patch_config(sym, path):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "MIN_SL_PCT" in src:
        return False, "already patched"

    pct = MIN_SL_PCT[sym]
    m = re.search(r"^(EXECUTE_TRADES\s*=.*)$", src, re.MULTILINE)
    if not m:
        return False, "EXECUTE_TRADES line not found"

    new_line = f"\nMIN_SL_PCT           = {pct}    # minimum SL as % of entry price"
    src = src[:m.end()] + new_line + src[m.end():]
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# live_runner.py patch
# ─────────────────────────────────────────────────────────────────────────────

def patch_runner(sym, path, is_paper):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    orig = src

    # ── 1. Add MIN_SL_PCT to config import ────────────────────────────────────
    if "MIN_SL_PCT" not in src:
        src = re.sub(
            r"(    EXECUTE_TRADES as _CFG_EXECUTE,?)",
            r"\1\n    MIN_SL_PCT,",
            src, count=1
        )
        if "MIN_SL_PCT" not in src:
            print(f"  [{sym}] WARNING: could not add MIN_SL_PCT to import")

    # ── 2. Paper guard: insert before fund_ctx ────────────────────────────────
    FUND_CTX = "    fund_ctx = _build_fundamental_context()"
    if is_paper and "[PAPER GUARD]" not in src:
        if FUND_CTX in src:
            src = src.replace(FUND_CTX, PAPER_GUARD + FUND_CTX, 1)
        else:
            print(f"  [{sym}] WARNING: fund_ctx anchor not found for paper guard")

    # ── 3. SL enforcement: insert before state["llm_calls"] += 2 ─────────────
    LLMC_ANCHOR = '\n\n    state["llm_calls"] += 2'
    if "[SL GUARD]" not in src:
        if LLMC_ANCHOR in src:
            enf = sl_enforce(MIN_SL_PCT[sym])
            # Insert enforcement between blank line and state["llm_calls"]
            src = src.replace(LLMC_ANCHOR, enf + '\n    state["llm_calls"] += 2', 1)
        else:
            print(f"  [{sym}] WARNING: llm_calls anchor not found for SL enforcement")

    # ── 4. Paper position save: before _save_state in synthesis block ─────────
    COOLDOWN_SAVE = '    state.pop("signal_cooldown_until", None)\n    _save_state(state)'
    if is_paper and "Track paper position" not in src:
        if COOLDOWN_SAVE in src:
            src = src.replace(
                COOLDOWN_SAVE,
                '    state.pop("signal_cooldown_until", None)\n' + PP_SAVE + '    _save_state(state)',
                1
            )
        else:
            print(f"  [{sym}] WARNING: cooldown_save anchor not found for paper_position save")

    if src != orig:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("_patch_v2.py — paper guard + MIN_SL enforcement")
    print("=" * 60)

    cfg_ok = cfg_skip = cfg_err = 0
    run_ok = run_skip = run_err = 0

    for sym in ALL_INSTR:
        is_paper = sym in PAPER_INSTR

        # ── config.py ─────────────────────────────────────────────────────────
        cfg_path = os.path.join(HERE, sym, "config.py")
        if not os.path.exists(cfg_path):
            print(f"  {sym:12s} [config]  MISSING")
            cfg_err += 1
        else:
            changed, msg = patch_config(sym, cfg_path)
            if changed:
                print(f"  {sym:12s} [config]  PATCHED  MIN_SL_PCT={MIN_SL_PCT[sym]}")
                cfg_ok += 1
            else:
                print(f"  {sym:12s} [config]  skip ({msg})")
                cfg_skip += 1

        # ── live_runner.py ────────────────────────────────────────────────────
        run_path = os.path.join(HERE, sym, "live_runner.py")
        if not os.path.exists(run_path):
            print(f"  {sym:12s} [runner]  MISSING")
            run_err += 1
        else:
            try:
                changed = patch_runner(sym, run_path, is_paper)
                # Verify
                with open(run_path, encoding="utf-8") as f:
                    content = f.read()
                has_import  = "MIN_SL_PCT" in content
                has_sl      = "[SL GUARD]" in content
                has_guard   = ("[PAPER GUARD]" in content) if is_paper else True
                has_pp_save = ("paper_position" in content) if is_paper else True
                ok = has_import and has_sl and has_guard and has_pp_save
                if changed:
                    status = "PATCHED" if ok else "PARTIAL"
                    print(f"  {sym:12s} [runner]  {status}  "
                          f"import={has_import} sl={has_sl} "
                          + (f"guard={has_guard} pp_save={has_pp_save}" if is_paper else "(live — no guard needed)"))
                    run_ok += 1
                else:
                    print(f"  {sym:12s} [runner]  skip (unchanged)")
                    run_skip += 1
            except Exception as e:
                print(f"  {sym:12s} [runner]  ERROR: {e}")
                import traceback; traceback.print_exc()
                run_err += 1

    print()
    print(f"config.py:     {cfg_ok} patched, {cfg_skip} skipped, {cfg_err} errors")
    print(f"live_runner.py: {run_ok} patched, {run_skip} skipped, {run_err} errors")


if __name__ == "__main__":
    main()
