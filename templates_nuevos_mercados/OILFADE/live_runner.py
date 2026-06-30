"""OILFADE — Oil Flush-Fade live runner (hibrido mecanica + noticias).
Loop cada POLL_SECONDS. En cada barra 15m CERRADA nueva de WTI/BRENT:
  1. ¿La barra cerrada es un shock? (rango>4x mediana 400, cuerpo>=50%)
  2. ¿Hay confirmacion cross-asset en la ultima hora (otros simbolos con shock)?
     -> SI: skip (noticia real, el shock continua; no fadear)
  3. ¿Noticias frescas de alta magnitud? -> VETO (capa inteligencia)
  4. Si huerfano y sin veto: FADE (contra el shock) SL 1.5xATR / TP 3xATR.
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

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from config import (
    ARTIFACTS_DIR, LIVE_LOG_FILE, LIVE_STATE_FILE, SIGNAL_FILE, PAPER_LOG_FILE, NEWS_DIR,
    OIL_SYMBOLS, SENTINELS, SHOCK_RANGE_MULT, MEDIAN_WINDOW, BODY_MIN_FRAC,
    CONFIRM_LOOKBACK_BARS, COOLDOWN_BARS, SL_ATR_MULT, TP_ATR_MULT,
    MAX_TRADES_PER_DAY, NEWS_MAX_AGE_H, NEWS_MAG_MIN, NEWS_CONF_MIN,
    EXECUTE_TRADES, OUR_MAGIC, POLL_SECONDS, calc_lots,
)


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
            with open(LIVE_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(s):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(LIVE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, default=str)


def _bars(sym, n):
    b = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, n)
    if b is None or len(b) < min(n, 50):
        return None
    return pd.DataFrame(b)


def _is_shock(df, i):
    """Criterio EXACTO del estudio sobre la barra i (debe haber >=MEDIAN_WINDOW previas)."""
    h, lo, c, o = df["high"], df["low"], df["close"], df["open"]
    rng = (h - lo) / c
    med = rng.iloc[max(0, i - MEDIAN_WINDOW):i].median()
    if not med or np.isnan(med) or rng.iloc[i] <= SHOCK_RANGE_MULT * med:
        return None
    body = c.iloc[i] - o.iloc[i]
    if abs(body) < BODY_MIN_FRAC * (h.iloc[i] - lo.iloc[i]):
        return None
    return 1 if body > 0 else -1   # direccion del shock


def _atr(df, i, span=14):
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.ewm(span=span, adjust=False).mean().iloc[i])


def _sentinel_confirm(ts_epoch):
    """¿Algun vigia tuvo shock en la ultima hora (4 barras cerradas)? -> noticia real."""
    confirms = []
    for s in SENTINELS + OIL_SYMBOLS:
        df = _bars(s, MEDIAN_WINDOW + 12)
        if df is None:
            continue
        for k in range(2, 2 + CONFIRM_LOOKBACK_BARS):     # barras cerradas recientes
            i = len(df) - k
            if i < MEDIAN_WINDOW // 2:
                break
            if abs(int(df["time"].iloc[i]) - ts_epoch) > 3900 + 900 * CONFIRM_LOOKBACK_BARS:
                break
            if _is_shock(df, i) is not None:
                confirms.append(s)
                break
    return confirms


def _news_veto(sym):
    """Capa inteligencia: articulo fresco de alta magnitud = shock con causa -> no fadear."""
    name = "WTI_N6" if "WTI" in sym else "BRENT_N6"
    p = os.path.join(NEWS_DIR, f"{name}_latest.json")
    if not os.path.exists(p):
        return None
    try:
        data = json.loads(open(p, encoding="utf-8").read())
        fetched = data.get("fetched_at", "")
        age_h = (datetime.now() - datetime.fromisoformat(fetched)).total_seconds() / 3600
        if age_h > NEWS_MAX_AGE_H:
            return None      # datos viejos: solo proxy mecanico
        for a in data.get("articles", []):
            intel = a.get("intel", a)
            if (intel.get("magnitude", 0) >= NEWS_MAG_MIN and
                    intel.get("confidence", 0) >= NEWS_CONF_MIN):
                return {"title": str(a.get("title", ""))[:120], "age_h": round(age_h, 1)}
    except Exception:
        return None
    return None


def check_symbol(sym, state, execute):
    df = _bars(sym, MEDIAN_WINDOW + 60)
    if df is None or len(df) < MEDIAN_WINDOW + 10:
        return
    i = len(df) - 2                      # ultima barra CERRADA
    bar_ts = int(df["time"].iloc[i])
    key = f"last_bar_{sym}"
    if state.get(key) == bar_ts:
        return                           # ya evaluada
    state[key] = bar_ts; _save_state(state)

    sgn = _is_shock(df, i)
    if sgn is None:
        return
    atr = _atr(df, i)
    close = float(df["close"].iloc[i])
    _log("shock_detected", {"sym": sym, "dir": "UP" if sgn > 0 else "DOWN",
                            "close": close, "atr": round(atr, 4),
                            "range_pct": round(float((df['high'].iloc[i]-df['low'].iloc[i])/close*100), 2)})

    # cooldown por simbolo
    last_i = state.get(f"last_trade_bar_{sym}")
    if last_i and (bar_ts - last_i) < COOLDOWN_BARS * 900:
        _log("cooldown_skip", {"sym": sym}); return
    # max trades/dia
    today = date.today().isoformat()
    if state.get("trades_date") != today:
        state["trades_date"] = today; state["trades_today"] = 0
    if state.get("trades_today", 0) >= MAX_TRADES_PER_DAY:
        _log("daily_cap_skip", {"sym": sym}); return
    # ya hay posicion nuestra en el simbolo?
    pos = mt5.positions_get(symbol=sym) or []
    if any(p.magic == OUR_MAGIC for p in pos):
        _log("position_open_skip", {"sym": sym}); return

    # 1) confirmacion cross-asset (pasado-solo) = noticia real -> NO fadear
    confirms = _sentinel_confirm(bar_ts)
    if confirms:
        _log("confirmed_skip", {"sym": sym, "confirmed_by": confirms,
                                "reason": "shock con confirmacion cross-asset = noticia real, no fadear"})
        return
    # 2) capa de noticias
    veto = _news_veto(sym)
    if veto:
        _log("news_veto", {"sym": sym, **veto,
                           "reason": "articulo fresco de alta magnitud = shock con causa"})
        return

    # 3) FADE
    d = "SHORT" if sgn > 0 else "LONG"
    sgn_t = 1 if d == "LONG" else -1
    sl = close - sgn_t * SL_ATR_MULT * atr
    tp = close + sgn_t * TP_ATR_MULT * atr
    stop_pts = abs(close - sl)
    lots = calc_lots(stop_pts)
    sig = {"timestamp": datetime.now().isoformat(), "symbol": sym, "signal": d,
           "entry": close, "sl": round(sl, 3), "tp": round(tp, 3), "lots": lots,
           "atr": round(atr, 4), "basis": f"flush-fade huerfano (0 confirmaciones, sin news-veto)"}
    with open(SIGNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(sig, f, indent=2)
    with open(PAPER_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(sig, default=str) + "\n")
    _log("signal", sig)

    if execute:
        from src.intraday.data.mt5_bridge import place_order
        res = place_order(direction="BUY" if d == "LONG" else "SELL", symbol=sym,
                          volume=lots, sl=round(sl, 3), tp=round(tp, 3),
                          comment="OILFADE", magic=OUR_MAGIC, dry_run=False)
        _log("execution", {"sym": sym, "dir": d, "lots": lots, "ok": res.get("ok"),
                           "price": res.get("price"), "retcode": res.get("retcode")})
        if res.get("ok"):
            state["trades_today"] = state.get("trades_today", 0) + 1
            state[f"last_trade_bar_{sym}"] = bar_ts
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
    execute = EXECUTE_TRADES
    print(f"OILFADE loop (poll {POLL_SECONDS}s) | execute={execute} | magic={OUR_MAGIC}")
    state = _load_state()
    while True:
        try:
            wd = datetime.now(timezone.utc).weekday()
            if wd < 5:                      # oil cerrado el finde
                for sym in OIL_SYMBOLS:
                    check_symbol(sym, state, execute)
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
