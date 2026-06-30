"""EVENTBREAK live runner — breakout de evento macro bidireccional (DEMO, opera de verdad).
Loop cada POLL_SECONDS. Para cada evento de macro_events.json cuyo minuto ya paso y esta dentro
de la ventana WATCH_MIN: ancla precio+ATR, arma gatillos ±K_ATR*ATR, y al primer cruce dispara
orden real (BUY arriba / SELL abajo) con SL/TP por ATR. Un trade por (evento,simbolo).
"""
import os, sys, io, json, time, argparse, traceback
from datetime import datetime, timezone, timedelta

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

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from config import (
    ARTIFACTS_DIR, LIVE_LOG_FILE, LIVE_STATE_FILE, SIGNAL_FILE, PAPER_LOG_FILE, EVENTS_FILE,
    K_ATR, SL_ATR_MULT, TP_ATR_MULT, WATCH_MIN, ATR_TIMEFRAME, ATR_BARS,
    CT_OFFSET_HOURS, CAPITAL_USD, MAX_RISK_PCT, RISK_CAP_PCT, EXECUTE_TRADES, OUR_MAGIC, POLL_SECONDS,
    ARM_DELAY_MIN, MAX_HOLD_MIN, MAX_CONCURRENT,
)

_TF = {"15m": mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30, "1h": mt5.TIMEFRAME_H1}


def _experiment_active(name):
    """Kill-switch: lee data/experiments.json. True si el experimento sigue activo."""
    try:
        import json as _j
        p = os.path.join(_MVP_ROOT, "data", "experiments.json")
        return bool(_j.load(open(p, encoding="utf-8")).get("experiments", {}).get(name, {}).get("active"))
    except Exception:
        return False


def _log(etype, data):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    entry = {"timestamp": datetime.now().isoformat(), "type": etype, **data}
    try:
        with open(LIVE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
    print(f"[{entry['timestamp'][11:19]}] [{etype}] {json.dumps(data, default=str)[:160]}")


def _load_state():
    if os.path.exists(LIVE_STATE_FILE):
        try:
            return json.load(open(LIVE_STATE_FILE, encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(s):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    json.dump(s, open(LIVE_STATE_FILE, "w", encoding="utf-8"), indent=2, default=str)


def _events():
    try:
        return json.load(open(EVENTS_FILE, encoding="utf-8")).get("events", [])
    except Exception:
        return []


def _atr_now(sym):
    b = mt5.copy_rates_from_pos(sym, _TF[ATR_TIMEFRAME], 0, ATR_BARS)
    if b is None or len(b) < 30:
        return None
    df = pd.DataFrame(b)
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.ewm(span=14, adjust=False).mean().iloc[-1])


def _calc_lots(sym, stop_price_dist):
    """Sizing generico por symbol_info: lots = riesgo_usd / (stop_dist * valor_por_precio_por_lote)."""
    try:
        from portfolio_guard import live_risk_usd
        risk_usd = live_risk_usd(CAPITAL_USD, MAX_RISK_PCT)
    except Exception:
        risk_usd = CAPITAL_USD * MAX_RISK_PCT
    si = mt5.symbol_info(sym)
    if not si or stop_price_dist <= 0:
        return None
    vpt = (si.trade_tick_value / si.trade_tick_size) if si.trade_tick_size else 1.0
    risk_per_lot = stop_price_dist * vpt
    if risk_per_lot <= 0:
        return si.volume_min
    raw = risk_usd / risk_per_lot
    step = si.volume_step or 0.01
    lots = round(round(raw / step) * step, 2)
    return float(max(si.volume_min, min(si.volume_max, lots)))


def _ev_utc(ev):
    eh, em = map(int, ev["time_ct"].split(":"))
    ct = datetime.strptime(ev["date"], "%Y-%m-%d").replace(hour=eh, minute=em)
    return ct.replace(tzinfo=timezone.utc) - timedelta(hours=CT_OFFSET_HOURS)


def _process(state, execute):
    now = datetime.now(timezone.utc)
    for ev in _events():
        try:
            evu = _ev_utc(ev)
        except Exception:
            continue
        if now < evu + timedelta(minutes=ARM_DELAY_MIN):
            continue                              # esperar ARM_DELAY: dejar pasar el barrido/spike inicial
        if now > evu + timedelta(minutes=WATCH_MIN):
            continue                              # ventana cerrada
        for sym in ev.get("symbols", []):
            key = f"{ev['date']}|{ev['name']}|{sym}"
            st = state.get(key, {})
            if st.get("status") in ("TRADED", "EXPIRED"):
                continue
            mt5.symbol_select(sym, True)
            tick = mt5.symbol_info_tick(sym)
            if not tick or tick.bid <= 0:
                continue                          # mercado cerrado / sin quote
            price = (tick.bid + tick.ask) / 2
            # 1) anclar al primer poll tras el evento
            if "anchor" not in st:
                atr = _atr_now(sym)
                if not atr or atr <= 0:
                    continue
                st = {"status": "ARMED", "anchor": price, "atr": atr,
                      "long_t": price + K_ATR * atr, "short_t": price - K_ATR * atr}
                state[key] = st; _save_state(state)
                _log("armed", {"event": ev["name"], "date": ev["date"], "sym": sym,
                               "anchor": round(price, 5), "atr": round(atr, 5),
                               "long_t": round(st["long_t"], 5), "short_t": round(st["short_t"], 5)})
                continue
            # 2) ya hay posicion nuestra en el simbolo? (evitar doble)
            pos = mt5.positions_get(symbol=sym) or []
            if any(p.magic == OUR_MAGIC for p in pos):
                continue
            # 2b) cap de posiciones concurrentes (evita ráfaga correlacionada de un solo evento)
            _allpos = mt5.positions_get() or []
            if sum(1 for p in _allpos if p.magic == OUR_MAGIC) >= MAX_CONCURRENT:
                _log("concurrent_cap", {"sym": sym, "event": ev["name"], "open": MAX_CONCURRENT,
                                        "reason": "max posiciones EVENTBREAK simultaneas alcanzado"})
                continue
            # 3) chequear breakout
            atr = st["atr"]
            direction = None
            if price >= st["long_t"]:
                direction = "LONG"
            elif price <= st["short_t"]:
                direction = "SHORT"
            if direction is None:
                continue
            entry = st["long_t"] if direction == "LONG" else st["short_t"]
            sl = entry - SL_ATR_MULT * atr if direction == "LONG" else entry + SL_ATR_MULT * atr
            tp = entry + TP_ATR_MULT * atr if direction == "LONG" else entry - TP_ATR_MULT * atr
            stop_dist = abs(entry - sl)
            lots = _calc_lots(sym, stop_dist)
            # CAP de riesgo por trade: si el lote-minimo fuerza riesgo > RISK_CAP_PCT del balance, SKIP
            # (oil en cuenta chica: vol_min 1.0 = ~13% -> se saltea; auto-habilita en cuenta grande).
            if lots:
                _si = mt5.symbol_info(sym); _a = mt5.account_info()
                _vpt = (_si.trade_tick_value / _si.trade_tick_size) if (_si and _si.trade_tick_size) else 1.0
                _risk = lots * stop_dist * _vpt
                if _a and _risk > RISK_CAP_PCT * _a.balance:
                    _log("risk_cap_skip", {"sym": sym, "event": ev["name"], "lots": lots,
                                           "risk_usd": round(_risk, 2), "cap_pct": RISK_CAP_PCT,
                                           "reason": f"lote min arriesga ${_risk:.0f} > {RISK_CAP_PCT*100:.0f}% balance"})
                    print(f"[RISK CAP] {sym} riesgo ${_risk:.0f} > cap -> skip")
                    st["status"] = "EXPIRED"; state[key] = st; _save_state(state); continue
            sig = {"timestamp": datetime.now().isoformat(), "symbol": sym, "event": ev["name"],
                   "signal": direction, "entry": round(entry, 5), "sl": round(sl, 5),
                   "tp": round(tp, 5), "lots": lots, "atr": round(atr, 5),
                   "basis": f"event-breakout {ev['name']} {direction} (k{K_ATR})"}
            with open(SIGNAL_FILE, "w", encoding="utf-8") as f:
                json.dump(sig, f, indent=2, default=str)
            with open(PAPER_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(sig, default=str) + "\n")
            _log("signal", sig)
            if execute and not _experiment_active("eventbreak"):
                _log("kill_switch", {"sym": sym, "reason": "experimento eventbreak APAGADO por kill-switch -> no opero"})
                print("[KILL-SWITCH] eventbreak apagado -> no ejecuto"); st["status"] = "TRADED"; state[key] = st; _save_state(state); continue
            if execute and lots and lots > 0:
                from src.intraday.data.mt5_bridge import place_order
                res = place_order(direction="BUY" if direction == "LONG" else "SELL", symbol=sym,
                                  volume=lots, sl=round(sl, 5), tp=round(tp, 5),
                                  comment=(f"EVB {ev['name']}")[:24], magic=OUR_MAGIC, dry_run=False)
                ok = bool(res.get("ok")) if isinstance(res, dict) else False
                _log("execution", {"sym": sym, "dir": direction, "lots": lots, "ok": ok,
                                   "retcode": res.get("retcode") if isinstance(res, dict) else None,
                                   "price": res.get("price") if isinstance(res, dict) else None,
                                   "result": res})
                st["status"] = "TRADED" if ok else "ARMED"
            else:
                st["status"] = "TRADED"   # sin ejecucion real, marcar para no repetir
            st["traded_dir"] = direction
            state[key] = st; _save_state(state)


def _expire_old(state):
    """Marca EXPIRED los eventos cuya ventana cerro sin breakout."""
    now = datetime.now(timezone.utc)
    changed = False
    for ev in _events():
        try:
            evu = _ev_utc(ev)
        except Exception:
            continue
        if now <= evu + timedelta(minutes=WATCH_MIN):
            continue
        for sym in ev.get("symbols", []):
            key = f"{ev['date']}|{ev['name']}|{sym}"
            st = state.get(key)
            if st and st.get("status") == "ARMED":
                st["status"] = "EXPIRED"; changed = True
                _log("expired", {"event": ev["name"], "date": ev["date"], "sym": sym,
                                 "reason": "sin breakout en la ventana"})
    if changed:
        _save_state(state)


def _timestop_check():
    """Time-stop: cierra posiciones EVENTBREAK abiertas mas de MAX_HOLD_MIN (corta sangrados largos)."""
    try:
        from src.intraday.data.mt5_bridge import close_position
        for p in (mt5.positions_get() or []):
            if p.magic != OUR_MAGIC:
                continue
            tk = mt5.symbol_info_tick(p.symbol)
            if not tk:
                continue
            age_min = (tk.time - p.time) / 60
            if age_min >= MAX_HOLD_MIN:
                res = close_position(ticket=p.ticket, symbol=p.symbol, magic=OUR_MAGIC, dry_run=False)
                ok = res.get("ok") if isinstance(res, dict) else None
                _log("timestop_close", {"sym": p.symbol, "ticket": p.ticket, "age_min": round(age_min),
                                        "pnl": round(p.profit + p.swap, 2), "ok": ok})
                print(f"[TIME-STOP] cierro {p.symbol} #{p.ticket} ({age_min:.0f}min) P&L ${p.profit+p.swap:+.2f}")
    except Exception as e:
        try:
            _log("loop_error", {"error": "timestop " + str(e)[:150]})
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    if not mt5.initialize():
        print("MT5 init fallo"); return
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACTS_DIR, "robot.pid"), "w") as f:
        f.write(str(os.getpid()))
    execute = EXECUTE_TRADES
    print(f"EVENTBREAK loop (poll {POLL_SECONDS}s) | execute={execute} | magic={OUR_MAGIC} | k{K_ATR} SL{SL_ATR_MULT}/TP{TP_ATR_MULT}")
    state = _load_state()
    while True:
        try:
            _process(state, execute)
            _expire_old(state)
            _timestop_check()
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
