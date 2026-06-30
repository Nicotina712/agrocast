"""GAPENGINE — live runner (gaps overnight en mega-caps, solo LONG).
Loop cada POLL_SECONDS. Logica:
  1. Al open NY (08:30-08:50 CT): para cada ticker, gap = open_hoy / close_ayer - 1 (D1).
     |gap| >= 3% -> LONG al mercado, SL 1.5xATR diario, sin TP.
  2. Desde 14:30 CT: cerrar posiciones con >= 5 dias habiles de hold.
Estado en live_state.json: posiciones propias {ticket: entry_date}, dia ya chequeado.
"""
import os, sys, io, json, time, argparse, traceback
from datetime import datetime, date, timedelta, timezone

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import MetaTrader5 as mt5

from config import (
    ARTIFACTS_DIR, LIVE_LOG_FILE, LIVE_STATE_FILE, SIGNAL_FILE, PAPER_LOG_FILE,
    BASKET, GAP_MIN_PCT, SL_ATR_MULT, HOLD_TRADING_DAYS, MAX_CONCURRENT,
    SESSION_OPEN_CT, SESSION_CLOSE_CT, ENTRY_WINDOW_MIN, EXIT_AFTER_CT,
    EXECUTE_TRADES, OUR_MAGIC, POLL_SECONDS, calc_lots,
)


def _now_ct():
    return datetime.now(timezone.utc) + timedelta(hours=-5)


def _log(etype, data):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    entry = {"timestamp": datetime.now().isoformat(), "ct": _now_ct().strftime("%H:%M"),
             "type": etype, **data}
    try:
        with open(LIVE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
    print(f"[{entry['ct']} CT] [{etype}] {json.dumps(data, default=str)[:150]}")


def _load_state():
    if os.path.exists(LIVE_STATE_FILE):
        try:
            return json.loads(open(LIVE_STATE_FILE, encoding="utf-8").read())
        except Exception:
            pass
    return {}


def _save_state(s):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(LIVE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, default=str)


def _daily(sym, n=20):
    b = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 0, n)
    if b is None or len(b) < 16:
        return None
    return pd.DataFrame(b)


def _atr_d(df, span=14):
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.ewm(span=span, adjust=False).mean().iloc[-1])


def _my_positions():
    out = []
    for p in (mt5.positions_get() or []):
        if p.magic == OUR_MAGIC:
            out.append(p)
    return out


def _trading_days_since(d0: date, d1: date) -> int:
    n = 0
    d = d0
    while d < d1:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def check_entries(state, execute):
    ct = _now_ct()
    today = ct.date().isoformat()
    open_min = SESSION_OPEN_CT[0] * 60 + SESSION_OPEN_CT[1]
    cur = ct.hour * 60 + ct.minute
    if not (open_min <= cur <= open_min + ENTRY_WINDOW_MIN):
        return
    if state.get("entries_checked") == today:
        return
    state["entries_checked"] = today
    _save_state(state)

    held = {p.symbol for p in _my_positions()}
    slots = MAX_CONCURRENT - len(held)
    candidates = []
    for sym in BASKET:
        if sym in held:
            continue
        mt5.symbol_select(sym, True)
        df = _daily(sym)
        if df is None:
            continue
        # ultima barra D1 = hoy (recien abierta); anterior = ayer
        d_today = pd.to_datetime(df["time"].iloc[-1], unit="s").date()
        if d_today != ct.date():            # la barra de hoy aun no existe
            continue
        o_today = float(df["open"].iloc[-1])
        c_prev = float(df["close"].iloc[-2])
        if o_today <= 0 or c_prev <= 0:
            continue
        gap = (o_today / c_prev - 1) * 100
        if abs(gap) < GAP_MIN_PCT:
            continue
        atr = _atr_d(df.iloc[:-1])           # ATR hasta ayer (sin la barra de hoy)
        if atr <= 0:
            continue
        candidates.append(dict(sym=sym, gap=gap, atr=atr, o=o_today))
    if not candidates:
        _log("no_gaps", {"date": today, "checked": len(BASKET)})
        return
    # priorizar gaps mas grandes si hay mas candidatos que slots
    candidates.sort(key=lambda x: -abs(x["gap"]))
    for cand in candidates[:max(0, slots)]:
        sym, gap, atr = cand["sym"], cand["gap"], cand["atr"]
        t = mt5.symbol_info_tick(sym)
        if not t or not t.ask:
            _log("no_tick_skip", {"sym": sym}); continue
        entry = t.ask
        sl = round(entry - SL_ATR_MULT * atr, 2)
        lots = calc_lots(entry - sl)
        kind = "PEAD-drift" if gap > 0 else "panic-fade"
        # ── FASE 2 SOMBRA: debate multi-agente opina y queda registrado, NO veta.
        # Tras 20-30 gaps se compara veredicto vs resultado y se decide darle poder.
        try:
            from gap_debate import shadow_debate
            _dbt = shadow_debate(sym, gap)
            _log("shadow_debate", _dbt)
        except Exception as _de:
            _log("shadow_debate", {"ticker": sym, "verdict": "UNAVAILABLE", "error": str(_de)[:100]})
        sig = {"timestamp": datetime.now().isoformat(), "symbol": sym, "signal": "LONG",
               "entry": entry, "sl": sl, "tp": None, "lots": lots, "gap_pct": round(gap, 2),
               "basis": f"gap {gap:+.1f}% -> LONG ({kind}), hold {HOLD_TRADING_DAYS}d"}
        with open(SIGNAL_FILE, "w", encoding="utf-8") as f:
            json.dump(sig, f, indent=2)
        with open(PAPER_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(sig, default=str) + "\n")
        _log("signal", sig)
        if execute:
            from src.intraday.data.mt5_bridge import place_order
            res = place_order(direction="BUY", symbol=sym, volume=lots,
                              sl=sl, tp=None, comment="GAPENGINE",
                              magic=OUR_MAGIC, dry_run=False)
            _log("execution", {"sym": sym, "lots": lots, "ok": res.get("ok"),
                               "price": res.get("price"), "retcode": res.get("retcode")})
            if res.get("ok"):
                state.setdefault("holds", {})[str(res.get("order"))] = {
                    "sym": sym, "entry_date": today, "gap": round(gap, 2)}
                _save_state(state)


def check_exits(state, execute):
    ct = _now_ct()
    cur = ct.hour * 60 + ct.minute
    exit_min = EXIT_AFTER_CT[0] * 60 + EXIT_AFTER_CT[1]
    close_min = SESSION_CLOSE_CT[0] * 60 + SESSION_CLOSE_CT[1]
    if not (exit_min <= cur <= close_min):
        return
    holds = state.get("holds", {})
    pos = _my_positions()
    live_tickets = {str(p.ticket): p for p in pos}
    # limpiar holds de posiciones ya cerradas (SL)
    for tk in list(holds.keys()):
        if tk not in live_tickets:
            _log("hold_closed_by_sl", {"ticket": tk, **holds[tk]})
            holds.pop(tk)
    changed = False
    for tk, info in list(holds.items()):
        d0 = date.fromisoformat(info["entry_date"])
        tdays = _trading_days_since(d0, ct.date())
        if tdays >= HOLD_TRADING_DAYS:
            p = live_tickets.get(tk)
            if p is None:
                holds.pop(tk); changed = True; continue
            if execute:
                from src.intraday.data.mt5_bridge import close_position
                res = close_position(ticket=p.ticket, symbol=p.symbol,
                                     comment="GAPENGINE_hold5", magic=OUR_MAGIC, dry_run=False)
                _log("hold_exit", {"sym": p.symbol, "ticket": tk, "tdays": tdays,
                                   "ok": res.get("ok"), "pnl_before": p.profit})
                if res.get("ok"):
                    holds.pop(tk); changed = True
    if changed:
        state["holds"] = holds
        _save_state(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    if not mt5.initialize():
        print("MT5 init fallo"); return
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACTS_DIR, "robot.pid"), "w") as f:
        f.write(str(os.getpid()))
    print(f"GAPENGINE loop (poll {POLL_SECONDS}s) | basket={len(BASKET)} | magic={OUR_MAGIC} | execute={EXECUTE_TRADES}")
    state = _load_state()
    while True:
        try:
            ct = _now_ct()
            if ct.weekday() < 5:
                check_entries(state, EXECUTE_TRADES)
                check_exits(state, EXECUTE_TRADES)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] {e}"); traceback.print_exc()
            try:
                _log("loop_error", {"error": str(e)[:200], "exc_type": type(e).__name__})
            except Exception:
                pass
        if not args.loop:
            break
        time.sleep(POLL_SECONDS)
    mt5.shutdown()


if __name__ == "__main__":
    main()
