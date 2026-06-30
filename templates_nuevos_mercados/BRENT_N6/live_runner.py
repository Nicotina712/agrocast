"""
BRENT_N6 Crude Oil â€” Live Runner
Mon-Fri, prime session 08:00-13:30 CT (NYMEX session).

Usage:
  python live_runner.py              # one cycle
  python live_runner.py --loop       # continuous
  python live_runner.py --diagnose   # MT5 test
  python live_runner.py --execute    # live orders
"""

import os, sys
# ── RETIRADO 2026-06-26 — guard anti-zombie. Sale de inmediato sin operar, lo lance quien lo lance.
# Motivo: sin edge mecánico intradía (ventana líquida vol-matched: trend +0.025/PF1.02, reversión +0.10/PF1.08,
# ambos ≈planos vs WTI trend +0.25/PF1.20). La literatura lo confirma: WTI lidera price-discovery (info-share >80%)
# y liquidez; Brent es follower → momentum propio débil. Redundante con WTI (corr ~0.9). Era el último LLM-primary.
# NO revivir sin re-evaluar. Ver memoria: llm-primary-research.
print("BRENT_N6 RETIRADO — runner deshabilitado (sin edge mecánico; Brent follower de WTI). Capítulo LLM-primary cerrado.")
sys.exit(0)

import io, json, time, argparse, traceback
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

from config import (
    SYMBOL, CT_OFFSET_HOURS, PRIME_OPEN_CT, PRIME_CLOSE_CT,
    TRADE_WEEKENDS, NO_NEW_SIGNALS_MINS,
    TIMEFRAME, N_BARS_LIVE, MIN_BARS_REQ,
    CYCLE_MINUTES, MAX_LLM_CALLS_PER_DAY,
    NO_TRADE_OPEN_MINS,
    EXECUTE_TRADES as _CFG_EXECUTE,
    MIN_SL_PCT,
    MAX_ENTRY_SLIP_PCT,
    ORDER_DEVIATION,
    COOLDOWN_BARS,
    MAX_LOSSES_PER_DAY,
    H1_TREND_FILTER,
    DEFAULT_VOLUME,
    SIGNAL_FILE, PAPER_LOG_FILE, LIVE_STATE_FILE, LIVE_LOG_FILE,
    ARTIFACTS_DIR,
)
from mt5_bridge import (
    initialize as mt5_init, shutdown as mt5_shutdown, is_connected as mt5_connected,
    fetch_mt5_bars, get_account_info, get_positions,
    place_order, close_position, diagnose as mt5_diagnose,
)
from microstructure import build_intraday_features, summarize_bars
from agents import call_trend_agent, call_risk_agent, synthesize_signal

OUR_MAGIC    = 20260608


def _experiment_active(name):
    """Kill-switch: lee data/experiments.json. True si el experimento sigue activo."""
    try:
        import json as _j
        p = os.path.join(_MVP_ROOT, "data", "experiments.json")
        return bool(_j.load(open(p, encoding="utf-8")).get("experiments", {}).get(name, {}).get("active"))
    except Exception:
        return False
_PRIME_START = (PRIME_OPEN_CT.hour,  PRIME_OPEN_CT.minute)
_PRIME_END   = (PRIME_CLOSE_CT.hour, PRIME_CLOSE_CT.minute)


def _now_ct():
    return datetime.now(timezone.utc) + timedelta(hours=CT_OFFSET_HOURS)

def _is_prime(ct=None):
    if ct is None: ct = _now_ct()
    t = ct.hour*60+ct.minute
    return (_PRIME_START[0]*60+_PRIME_START[1]) <= t <= (_PRIME_END[0]*60+_PRIME_END[1])

def _mins_to_close(ct=None):
    if ct is None: ct = _now_ct()
    return max(0, _PRIME_END[0]*60+_PRIME_END[1] - ct.hour*60-ct.minute)

def _is_trading_day():
    if TRADE_WEEKENDS: return True
    wd = _now_ct().weekday()
    if wd < 5: return True
    if wd == 5: return False
    return _now_ct().hour >= 17  # Sunday: ICMarkets opens ~17:00 CT


def _load_state():
    if os.path.exists(LIVE_STATE_FILE):
        try:
            with open(LIVE_STATE_FILE, encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    return {"date":"","llm_calls":0,"cycles":0,"last_signal_time":None}

def _save_state(s):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(LIVE_STATE_FILE,"w",encoding="utf-8") as f: json.dump(s,f,indent=2,default=str)

def _can_call_llm(state):
    today = date.today().isoformat()
    if state.get("date") != today:
        state.update({"date":today,"llm_calls":0,"cycles":0})
    if state["llm_calls"] >= MAX_LLM_CALLS_PER_DAY:
        return False, f"max {MAX_LLM_CALLS_PER_DAY} calls/day reached"
    return True, "ok"


def _log(etype, data):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    entry = {"timestamp":datetime.now().isoformat(),"ct_time":_now_ct().strftime("%H:%M"),"type":etype,**data}
    try:
        with open(LIVE_LOG_FILE,"a",encoding="utf-8") as f:
            f.write(json.dumps(entry,ensure_ascii=False,default=str)+"\n")
    except Exception: pass
    skip = ("trend_analysis","risk_analysis")
    print(f"[{entry['ct_time']} CT] [{etype}] {json.dumps({k:v for k,v in data.items() if k not in skip},default=str)}")

def _save_signal(sig):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(SIGNAL_FILE,"w",encoding="utf-8") as f: json.dump(sig,f,indent=2,ensure_ascii=False,default=str)

def _log_paper(sig, summary):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    entry = {
        "timestamp":sig.get("timestamp"),"symbol":SYMBOL,
        "signal":sig.get("signal"),"entry":sig.get("entry"),
        "sl":sig.get("sl"),"tp":sig.get("tp"),"lots":sig.get("lots"),
        "confidence":sig.get("confidence"),"rr":sig.get("rr"),
        "risk_usd":sig.get("risk_usd"),"reasoning":sig.get("reasoning"),
        "oil_regime":sig.get("oil_regime"),"opec_dynamic":sig.get("opec_dynamic"),
        "price_at_signal":summary.get("current_state",{}).get("close"),
    }
    with open(PAPER_LOG_FILE,"a",encoding="utf-8") as f:
        f.write(json.dumps(entry,ensure_ascii=False,default=str)+"\n")


def _build_fundamental_context():
    ct = _now_ct()
    return {
        "eia_inventory":  "unknown â€” check EIA Wednesday 10:30 ET",
        "api_report":     "unknown â€” API Tuesday ~16:30 ET",
        "opec_stance":    "unknown â€” OPEC+ production decisions",
        "geopolitical_risk":"unknown â€” Middle East tensions, Russia supply",
        "usd_index":      "unknown â€” WTI inverse corr with DXY",
        "china_demand":   "unknown â€” China PMI, economic activity",
        "session_note":   f"Brent ICE London session | CT: {ct.strftime('%H:%M')} | weekday: {ct.weekday()}",
        "key_levels_note":"Watch: $70, $75, $80, $85, $90, $95 per barrel",
    }


def run_cycle(execute=False):
    ct = _now_ct()
    is_prime_now = _is_prime(ct)
    mins_left    = _mins_to_close(ct)
    is_wknd      = ct.weekday() >= 5

    print(f"\n{'='*60}")
    print(f"BRENT_N6 Crude Oil â€” Cycle at {ct.strftime('%Y-%m-%d %H:%M')} CT")
    print(f"Prime: {is_prime_now} | Weekend: {is_wknd} | Mins left: {mins_left}")
    print("="*60)

    if not _is_trading_day():
        print("Weekend â€” WTI market closed. Skip.")
        return {"status":"weekend"}

    if not mt5_connected():
        if not mt5_init():
            _log("error",{"msg":"MT5 connection failed"}); return {"status":"mt5_error"}

    acct = get_account_info()
    print(f"Account: ${acct.get('balance','?'):.2f} | Equity: ${acct.get('equity','?'):.2f}")

    bars = fetch_mt5_bars(TIMEFRAME, N_BARS_LIVE, SYMBOL)
    if bars is None or len(bars) < MIN_BARS_REQ:
        _log("error",{"msg":f"Not enough bars: {len(bars) if bars is not None else 0}"}); return {"status":"no_data"}
    print(f"Bars: {len(bars)} x {TIMEFRAME} | Latest: {bars.index[-1]}")

    feats   = build_intraday_features(bars, TIMEFRAME)
    summary = summarize_bars(bars, feats)
    price   = summary["current_state"]["close"]

    _log("cycle",{"price":price,"is_prime":is_prime_now,"is_weekend":is_wknd,"mins_left":mins_left,"bars_count":len(bars)})

    # â”€â”€ A2. ATR volatility filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        import sys as _sys, os as _os_pg
        _pg_root = _os_pg.path.dirname(_os_pg.path.dirname(_os_pg.path.abspath(__file__)))
        if _pg_root not in _sys.path: _sys.path.insert(0, _pg_root)
        from portfolio_guard import atr_vol_filter as _avf
        _vf = _avf(bars)
        if _vf:
            print(f"[VOL FILTER] {_vf['reason']}")
            _log("vol_filter", _vf)
            return {"status": "vol_filter", **_vf}
    except Exception:
        pass

    # -- A2c. STORM BREAKER (validado 2026-06-11): regimen post-tormenta cross-asset
    # es loteria negativa para entradas trend (bloque WTI -0.18/ETH -0.14/CHINA50 -0.10 expR).
    # BTC y UK100 EXENTOS (sus buckets post-tormenta son positivos).
    try:
        from portfolio_guard import storm_breaker_check as _sbc
        _sb = _sbc(window_hours=12.0)
        if _sb:
            print(f"[STORM BREAKER] {{_sb['reason']}}")
            _log("storm_breaker", _sb)
            return {{"status": "storm_breaker", **_sb}}
    except Exception:
        pass

    # -- A2b. Market dormancy check — ahorrar calls cuando el mercado no se mueve --
    # Si el rango medio de las últimas 3 barras < 0.08% del precio → mercado dormido.
    # No gastar call LLM en mercados planos; reservar presupuesto para horas activas.
    try:
        from portfolio_guard import market_dormancy_check as _mdc
        _dormant = _mdc(bars)
        if _dormant:
            print(f"[DORMANT] {_dormant['reason']}")
            _log("dormant", _dormant)
            return {"status": "dormant", **_dormant}
    except Exception:
        pass

    if not is_prime_now:
        print(f"Outside prime ({_PRIME_START[0]:02d}:{_PRIME_START[1]:02d}-{_PRIME_END[0]:02d}:{_PRIME_END[1]:02d} CT). Monitor only.")
        _log("outside_prime", {"mins_left": mins_left})
        return {"status":"outside_prime","price":price}

    # ── Bloqueo apertura prime (NO_TRADE_OPEN_MINS) ───────────────────────────
    # Los primeros 30min de prime (06:00-06:30 CT) tienen z>1.5 el 50% del tiempo.
    # Análisis ATR histórico confirmó que la apertura ICE London es el periodo
    # más peligroso: spikes de 2-3x ATR por liquidez delgada y flujos institucionales.
    prime_open_mins = _PRIME_START[0] * 60 + _PRIME_START[1]
    ct_mins         = ct.hour * 60 + ct.minute
    mins_since_open = (ct_mins - prime_open_mins) % 1440
    if mins_since_open < NO_TRADE_OPEN_MINS:
        remaining = NO_TRADE_OPEN_MINS - mins_since_open
        print(f"[OPEN GUARD] Primeros {NO_TRADE_OPEN_MINS}min de prime — apertura ICE peligrosa. Faltan {remaining}min.")
        _log("open_guard", {"mins_since_open": mins_since_open, "mins_remaining": remaining,
                             "reason": f"no nuevas entradas en los primeros {NO_TRADE_OPEN_MINS}min de prime"})
        return {"status": "open_guard", "price": price}

    if mins_left < NO_NEW_SIGNALS_MINS:
        print(f"Last {NO_NEW_SIGNALS_MINS} min of prime. No new signals.")
        _log("too_late", {"mins_left": mins_left})
        return {"status":"too_late","mins_left":mins_left}

    state = _load_state()
    can_call, reason = _can_call_llm(state)
    if not can_call:
        print(f"LLM gate: {reason}")
        _log("llm_gate", {"reason": reason, "state_date": state.get("date"), "llm_calls": state.get("llm_calls")})
        return {"status":"llm_gate","reason":reason}
    # -- A2c. Economic calendar filter — no entrar cerca de eventos macro -------
    # EIA (mie 09:30 CT), API (mar 16:30 CT), NFP (1er vie 07:30 CT) ±30min
    try:
        from portfolio_guard import economic_calendar_block as _ecb
        _cal = _ecb(SYMBOL, ct, window_mins=30)
        if _cal:
            print(f"[CALENDAR] {_cal['reason']}")
            _log("calendar_block", _cal)
            return {"status": "calendar_block", **_cal}
    except Exception:
        pass

    # -- A3. H1 Trend Filter — bloquear entradas contra la tendencia horaria ----
    # Si H1 EMA9 < EMA21: tendencia bajista → vetar LONGs (solo SHORTs o FLAT)
    # Si H1 EMA9 > EMA21: tendencia alcista → vetar SHORTs (solo LONGs o FLAT)
    # Objetivo: evitar "oversold bounce" LONGs en dias claramente bajistas (WTI -4%)
    if H1_TREND_FILTER:
        try:
            import numpy as _np
            _h1 = fetch_mt5_bars("1h", 30, SYMBOL)
            if _h1 is not None and len(_h1) >= 21:
                _h1_close = _h1["close"].values if hasattr(_h1, "values") else _h1.iloc[:,3].values
                _ema9  = float(sum(_h1_close[-9:])  / 9)   # aproximacion simple EMA rapida
                _ema21 = float(sum(_h1_close[-21:]) / 21)
                _h1_trend = "BEAR" if _ema9 < _ema21 else "BULL"
                _h1_price = float(_h1_close[-1])
                print(f"[H1 TREND] EMA9={_ema9:.2f} EMA21={_ema21:.2f} → {_h1_trend} | precio={_h1_price:.2f}")
                _log("h1_trend", {"ema9": round(_ema9,2), "ema21": round(_ema21,2),
                                   "trend": _h1_trend, "price_h1": round(_h1_price,2)})
                state["h1_trend"] = _h1_trend
                state["h1_ema9"]  = round(_ema9, 2)
                state["h1_ema21"] = round(_ema21, 2)
        except Exception as _h1e:
            print(f"[H1 TREND] Error (non-blocking): {_h1e}")
            state["h1_trend"] = None

    # -- A4. Daily loss gate — kill switch tras MAX_LOSSES_PER_DAY SL hits -----
    # Cuenta SL hits del dia en el live_log. Si ya alcanzó el límite, parar.
    try:
        import json as _json2
        from datetime import date as _date2
        _today2  = _date2.today().isoformat()
        _log_path = LIVE_LOG_FILE
        _sl_hits = 0
        if os.path.exists(_log_path):
            with open(_log_path, encoding="utf-8", errors="replace") as _lf:
                for _ll in _lf:
                    try:
                        _le = _json2.loads(_ll)
                        if (_le.get("timestamp","")[:10] == _today2 and
                                _le.get("type") == "execution" and
                                _le.get("result",{}).get("ok") == True):
                            _sl_hits += 1
                    except: pass
        from datetime import datetime as _dtn2, timezone as _tz2
        _dt_now  = _dtn2.now(_tz2.utc)
        _dt_bod  = _dt_now.replace(hour=0, minute=0, second=0, microsecond=0)
        import MetaTrader5 as _mt5_dlg
        _deals_today = _mt5_dlg.history_deals_get(_dt_bod, _dt_now) or []
        _sl_today = sum(1 for _d in _deals_today
                        if _d.entry == 1 and 'BRENT' in _d.symbol
                        and (_d.profit + _d.commission + _d.swap) < 0)
        if _sl_today >= MAX_LOSSES_PER_DAY:
            print(f"[DAILY LOSS GATE] {_sl_today} SL hits hoy >= limite {MAX_LOSSES_PER_DAY} — parar por hoy")
            _log("daily_loss_gate", {"sl_hits_today": _sl_today, "limit": MAX_LOSSES_PER_DAY})
            _save_state(state)
            return {"status": "daily_loss_gate", "sl_hits": _sl_today}
    except Exception as _dlg_e:
        print(f"[DAILY LOSS GATE] Error (non-blocking): {_dlg_e}")

    # -- B. MT5 position guard + News conflict check (Option B) ---------------
    # If there is already an open position, skip new signal.
    # BUT first check if news contradicts the open position -> early exit.
    try:
        from mt5_bridge import get_positions as _get_pos
        _all_pos = _get_pos(SYMBOL) or []
        _open = [p for p in _all_pos if p.get("magic") == OUR_MAGIC]
        if _open:
            _p = _open[0]
            _ptype  = "BUY" if _p.get("type") in (0, "BUY", "buy") else "SELL"
            _ticket = _p.get("ticket")
            _profit = _p.get("profit", 0)
            print(f"[POSITION GUARD] Open: {_ptype} #{_ticket} @ {_p.get('price_open','?')} | P&L: {_profit:.2f}")
            try:
                import sys as _sys2, os as _os2
                _ni_root = _os2.path.dirname(_os2.path.dirname(_os2.path.abspath(__file__)))
                if _ni_root not in _sys2.path: _sys2.path.insert(0, _ni_root)
                from templates_nuevos_mercados.news_intelligence.news_monitor import check_news_conflict
                _news_chk = check_news_conflict(SYMBOL, position_type=_ptype)
                _age = f"{_news_chk.get('data_age_h','?')}h" if _news_chk.get('data_age_h') is not None else "no-data"
                print(f"[NEWS MONITOR] exit={_news_chk['exit']} score={_news_chk.get('score',0):.2f} age={_age} | {_news_chk['reason'][:80]}")
                if _news_chk["exit"]:
                    _log("news_exit_triggered", {"ticket":_ticket,"position_type":_ptype,"profit":_profit,"news_score":_news_chk["score"],"reason":_news_chk["reason"][:200],"n_conflicts":len(_news_chk["conflicts"])})
                    print("[NEWS MONITOR] >>> CIERRE ANTICIPADO por conflicto de noticias <<<")
                    _close_res = close_position(ticket=_ticket, symbol=SYMBOL, comment="news_exit", magic=OUR_MAGIC, dry_run=not execute)
                    _log("news_exit_result", _close_res)
                    if _close_res.get("ok"):
                        print(f"[NEWS MONITOR] Posicion cerrada OK | P&L={_profit:.2f} | dry_run={not execute}")
                    else:
                        print(f"[NEWS MONITOR] Cierre FALLO: {_close_res.get('error') or _close_res.get('comment')}")
                    return {"status":"news_exit","ticket":_ticket,"close_result":_close_res}
            except Exception as _ne:
                print(f"[NEWS MONITOR] Error (no critico): {_ne}")
            # -- FASE 2: Trailing stop + breakeven sobre la posicion abierta ----
            try:
                from portfolio_guard import manage_open_position as _mop
                from mt5_bridge import modify_position_sl as _msl
                _atr_now = float((bars["high"] - bars["low"]).tail(14).mean())
                _trail = _mop(_p, _atr_now, close_position, modify_sl_fn=_msl,
                              breakeven_at_r=1.0, trail_at_r=999, trail_distance_r=0.5)
                if _trail:
                    print(f"[TRAILING] {_trail['action']} ticket={_trail['ticket']} "
                          f"SL {_trail['old_sl']}->{_trail['new_sl']} (profit {_trail['profit_r']}R) ok={_trail['ok']}")
                    _log("trailing_stop", _trail)
            except Exception as _te:
                print(f"[TRAILING] Error (no critico): {_te}")
            _log("position_guard", {"ticket":_ticket,"profit":_profit})
            return {"status":"position_open","ticket":_ticket}
    except Exception as _pg_err:
        print(f"[POSITION GUARD] check failed: {_pg_err}")
        _log("position_guard_error", {"error":str(_pg_err)})
        return {"status":"position_guard_error","error":str(_pg_err)}

    # â”€â”€ B2. Cluster cap â€” mÃ¡x 1 posiciÃ³n abierta por cluster correlacionado â”€â”€
    # Cluster cap movido a seccion G (post-LLM): direction-aware para energy.

    # â”€â”€ C. Post-signal cooldown â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # After any non-FLAT signal, wait 2 Ã— CYCLE_MINUTES before new signal.
    _cd_until = state.get("signal_cooldown_until")
    if _cd_until:
        try:
            from datetime import datetime as _dt
            _cd_dt = _dt.fromisoformat(_cd_until)
            if _dt.now() < _cd_dt:
                _mins_cd = int((_cd_dt - _dt.now()).total_seconds() / 60)
                print(f"[COOLDOWN] {_mins_cd} min remaining after last signal â€” skip LLM")
                _log("cooldown", {"cooldown_until": _cd_until, "mins_left": _mins_cd})
                return {"status": "cooldown", "cooldown_until": _cd_until}
        except Exception:
            pass



    # â”€â”€ D. Paper position guard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                      f"entry={_pp_entry} SL={_pp_sl} TP={_pp_tp} now={_cur:.4f} â€” skip LLM")
                _log("paper_guard", {"signal": _pp_sig, "entry": _pp_entry,
                                     "sl": _pp_sl, "tp": _pp_tp,
                                     "current_price": _cur, "outcome": "OPEN"})
                return {"status": "paper_position_open"}
            else:
                print(f"[PAPER GUARD] {_pp_sig} closed â†’ {_pp_out} | New signal allowed.")
                _log("paper_guard", {"signal": _pp_sig, "entry": _pp_entry,
                                     "sl": _pp_sl, "tp": _pp_tp,
                                     "current_price": _cur, "outcome": _pp_out})
                state.pop("paper_position", None)
                _save_state(state)

    fund_ctx = _build_fundamental_context()

    # LLM RESILIENTE: si cae (sin créditos/timeout) -> trend/risk vacíos -> synthesize da
    # FLAT -> sin veto -> se toma la señal mecánica igual. El portfolio NO queda ciego.
    trend, risk = {}, {}
    print("Calling Trend+Risk Agent (BRENT, veto)...")
    try:
        trend = call_trend_agent(summary, fund_ctx)
        if "error" in trend: raise RuntimeError(trend["error"])
        risk = call_risk_agent(summary, trend, fund_ctx)
        if "error" in risk: raise RuntimeError(risk["error"])
    except Exception as _exc:
        trend, risk = {}, {}
        _log("agent_error", {"error": str(_exc)[:120], "fallback": "mecánico sin veto"})
        print(f"[RESILIENCIA] LLM caído ({str(_exc)[:50]}) -> mecánica sin veto")

    viable = risk.get("trade_viable", False)
    print(f"Risk: viable={viable} | {risk.get('volatility_regime')} | ATR: ${risk.get('atr_5m')} | {risk.get('veto_reason') or 'No veto'}")

    signal = synthesize_signal(trend, risk)
    signal.update({"timestamp":datetime.now().isoformat(),"symbol":SYMBOL,"price":price})

    # ── FILTRO DE CONTEXTO ACTIVO (BRENT←XAGUSD plata, Δ+0.16 en mecánica). Activado 2026-06-15
    # por decisión del usuario (respaldo cualitativo: el trade −$82 del 06-15 fue aligned=False y perdió).
    # Si el short/long va CONTRA el momentum del complejo commodities → FLAT (no operar).
    if signal["signal"] != "FLAT":
        try:
            from portfolio_guard import context_aligned as _ctxa
            _aligned = _ctxa("XAGUSD", signal["signal"])
            _log("ctx_filter", {"signal": signal["signal"], "proxy": "XAGUSD", "aligned": _aligned})
            if _aligned is False:   # None = sin datos → no bloquear
                print(f"[CTX FILTER] {signal['signal']} contra momentum XAGUSD -> FLAT")
                signal["signal"] = "FLAT"
        except Exception:
            pass

    # -- E0. H1 Trend Filter veto (sobre señal del LLM) -----------------------
    if H1_TREND_FILTER and signal["signal"] != "FLAT":
        _h1_t = state.get("h1_trend")
        if _h1_t == "BEAR" and signal["signal"] == "LONG":
            print(f"[H1 VETO] LONG vetado — H1 BAJISTA (EMA9={state.get('h1_ema9')} < EMA21={state.get('h1_ema21')})")
            _log("h1_trend_veto", {"signal":"LONG","h1_trend":"BEAR",
                                    "ema9":state.get("h1_ema9"),"ema21":state.get("h1_ema21")})
            signal["signal"] = "FLAT"
        elif _h1_t == "BULL" and signal["signal"] == "SHORT":
            print(f"[H1 VETO] SHORT vetado — H1 ALCISTA (EMA9={state.get('h1_ema9')} > EMA21={state.get('h1_ema21')})")
            _log("h1_trend_veto", {"signal":"SHORT","h1_trend":"BULL",
                                    "ema9":state.get("h1_ema9"),"ema21":state.get("h1_ema21")})
            signal["signal"] = "FLAT"
    # â”€â”€ E. Minimum SL distance enforcement (Fix #2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if signal["signal"] != "FLAT":
        _e   = float(signal.get("entry") or 0)
        _sl  = float(signal.get("sl")    or 0)
        _slp = abs(_e - _sl) / _e * 100 if _e else 0
        if _slp < 0.2:
            _min_d = _e * 0.2 / 100
            if signal["signal"] == "LONG":
                signal["sl"] = round(_e - _min_d, 5)
            else:
                signal["sl"] = round(_e + _min_d, 5)
            signal["sl_enforced"] = True
            signal["sl_original"] = _sl
            print(f"[SL GUARD] SL too tight ({_slp:.3f}% < 0.2%). "
                  f"Adjusted: {_sl:.5f} -> {signal['sl']:.5f}")
            # Recompute R:R with expanded SL â€” veto if no longer viable
            _new_risk   = abs(_e - float(signal["sl"]))
            _new_reward = abs(float(signal.get("tp", 0)) - _e)
            _real_rr    = round(_new_reward / _new_risk, 2) if _new_risk > 0 else 0
            signal["rr"] = _real_rr
            if _real_rr < 2.0:
                print(f"[SL GUARD] Real R:R after SL expansion = {_real_rr:.2f} < 2.0 â€” VETO")
                signal["signal"] = "FLAT"
                signal["veto_reason"] = f"SL expansion destroyed R:R: {_real_rr:.2f}"


    # -- F. Counter-trend quality filter (BRENT/WTI) --------------------------
    # Contrariar el trend dominante es valido SOLO con alta conviccion.
    # MEDIUM confidence en contra del trend = ruido, no vale el riesgo.
    # HIGH confidence en contra del trend = apuesta con fundamento, se permite.
    if signal["signal"] != "FLAT":
        _sig       = signal["signal"]             # "LONG" o "SHORT"
        _conf      = signal.get("confidence","LOW")
        _oil_reg   = signal.get("oil_regime","")
        _is_counter = (
            (_sig == "LONG"  and _oil_reg == "BEAR_TREND") or
            (_sig == "SHORT" and _oil_reg == "BULL_TREND")
        )
        # EXPERIMENTO 2026-06-17 (kill-switch): permitir MEDIUM counter-trend en demo.
        _allow_med = (_conf == "MEDIUM" and _experiment_active("brent_take_medium"))
        if _is_counter and _conf != "HIGH" and not _allow_med:
            print(f"[TREND FILTER] {_sig} contra {_oil_reg} con conf={_conf} -- veto (requiere HIGH)")
            _log("trend_filter_veto", {"signal":_sig,"oil_regime":_oil_reg,"confidence":_conf})
            signal["signal"]      = "FLAT"
            signal["veto_reason"] = f"Counter-trend {_sig} en {_oil_reg} requiere HIGH confidence (actual: {_conf})"
        elif _is_counter and (_conf == "HIGH" or _allow_med):
            _why = "HIGH" if _conf == "HIGH" else "MEDIUM (experimento)"
            print(f"[TREND FILTER] {_sig} contra {_oil_reg} PERMITIDO (conf={_why})")
            _log("trend_filter_allow", {"signal":_sig,"oil_regime":_oil_reg,"confidence":_conf,"experiment":_allow_med})
    # -------------------------------------------------------------------------

    # -- G. Cluster cap direction-aware (energy) --------------------------------
    # Solo bloquear si la posicion del cluster va en la MISMA direccion.
    # BRENT LONG + WTI SHORT = spread legitimo -> permitir.
    # BRENT LONG + WTI LONG  = doble exposicion correlacionada -> bloquear.
    if signal["signal"] != "FLAT":
        try:
            import sys as _sys_g, os as _os_g
            _pg_root_g = _os_g.path.dirname(_os_g.path.dirname(_os_g.path.abspath(__file__)))
            if _pg_root_g not in _sys_g.path: _sys_g.path.insert(0, _pg_root_g)
            from portfolio_guard import cluster_cap_blocked as _ccb_g
            from mt5_bridge import get_positions as _gp_g
            _cc_g = _ccb_g(SYMBOL, _gp_g, proposed_direction=signal["signal"])
            if _cc_g:
                _dir_msg = f"{_cc_g['blocking_dir']} en {_cc_g['blocking_symbol']}"
                print(f"[CLUSTER CAP] Misma direccion ({_cc_g['proposed_dir']}) que {_dir_msg} -- veto")
                _log("cluster_cap", _cc_g)
                signal["signal"] = "FLAT"
                signal["veto_reason"] = f"Cluster energy: {_dir_msg} ya abierto (misma dir)"
            else:
                # Verificar si hay cluster-mate de direccion opuesta (spread permitido)
                from portfolio_guard import cluster_cap_blocked as _ccb_raw
                _cc_raw = _ccb_raw(SYMBOL, _gp_g)  # sin direction = check crudo
                if _cc_raw:
                    _opp_dir = "SHORT" if signal["signal"] == "LONG" else "LONG"
                    print(f"[CLUSTER CAP] Spread energy: {signal['signal']} WTI vs {_cc_raw['blocking_dir']} {_cc_raw['blocking_symbol']} -- PERMITIDO")
                    _log("cluster_spread", {"wti_signal": signal["signal"], **_cc_raw})
        except Exception as _cce:
            pass  # si falla el check, continuamos
    # -------------------------------------------------------------------------
    
    # -- H. Entry slippage gate ------------------------------------------------
    # Si el precio actual es adverso respecto al entry calculado por el LLM,
    # significa que el mercado ya se movio en nuestra contra antes de entrar.
    # SHORT: precio DEBAJO del entry (ya en el fondo, no en la resistencia)
    # LONG:  precio ENCIMA del entry (ya en el techo, no en el soporte)
    # Se veta la senal para no perseguir precios extendidos.
    if signal["signal"] != "FLAT":
        _entry_h = float(signal.get("entry") or price)
        if _entry_h > 0 and price > 0:
            if signal["signal"] == "SHORT":
                _adverse_h = (_entry_h - price) / _entry_h * 100
            else:  # LONG
                _adverse_h = (price - _entry_h) / _entry_h * 100
            if _adverse_h > MAX_ENTRY_SLIP_PCT:
                print(f"[ENTRY GATE] Slippage adverso {_adverse_h:.3f}% > {MAX_ENTRY_SLIP_PCT}% -- veto "
                      f"(entry_llm={_entry_h} precio={price:.4f})")
                _log("entry_gate_veto", {"signal": signal["signal"], "entry_llm": _entry_h,
                                         "price": round(price, 5), "adverse_pct": round(_adverse_h, 4),
                                         "threshold": MAX_ENTRY_SLIP_PCT})
                signal["signal"]      = "FLAT"
                signal["veto_reason"] = (f"Entry gate: slippage adverso {_adverse_h:.2f}% > "
                                          f"{MAX_ENTRY_SLIP_PCT}% (entry={_entry_h}, precio={price:.4f})")
    # -------------------------------------------------------------------------
    state["llm_calls"] += 2; state["cycles"] += 1; state["last_signal_time"] = signal["timestamp"]

    # set post-signal cooldown — usa COOLDOWN_BARS×CYCLE_MINUTES (era CYCLE_MINUTES×2=30min)
    if signal["signal"] != "FLAT":
        from datetime import datetime as _dt2, timedelta as _td2
        _cd_mins = CYCLE_MINUTES * COOLDOWN_BARS  # 15×8=120min (era 30min)
        state["signal_cooldown_until"] = (_dt2.now() + _td2(minutes=_cd_mins)).isoformat()
    else:
        state.pop("signal_cooldown_until", None)
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
    _save_state(state)
    _save_signal(signal)
    _log("signal",{k:signal.get(k) for k in ("signal","entry","sl","tp","lots","confidence","rr","risk_usd","oil_regime")})

    if signal["signal"] != "FLAT":
        _log_paper(signal, summary)
        print(f"\n>>> SIGNAL: {signal['signal']} @ ${signal.get('entry'):,.3f} | SL: ${signal.get('sl'):,.3f} | TP: ${signal.get('tp'):,.3f}")
        print(f"    Lots: {signal.get('lots')} | R:R: {signal.get('rr')} | Risk: ~${signal.get('risk_usd')}")
        print(f"    Regime: {signal.get('oil_regime')} | OPEC: {signal.get('opec_dynamic')} | Conf: {signal.get('confidence')}")
    else:
        print(f"\n>>> FLAT -- {str(signal.get('reasoning',''))[:150]}")

    if signal["signal"] != "FLAT" and not (signal.get("lots") or 0) > 0:
        print(f"[SIZING] lotes={signal.get('lots')} (riesgo > techo o margen insuficiente) -> no opera")
        _log("sizing_skip", {"signal": signal["signal"], "lots": signal.get("lots"),
                             "entry": signal.get("entry"), "sl": signal.get("sl")})
        signal["signal"] = "FLAT"; signal["veto_reason"] = "lote mínimo excede techo de riesgo / margen"

    if execute and signal["signal"] != "FLAT":
        mt5_dir = "BUY" if signal["signal"] == "LONG" else "SELL"
        _sl = float(signal.get("sl")) if signal.get("sl") is not None else None
        _tp = float(signal.get("tp")) if signal.get("tp") is not None else None
        # NOTA: place_order NO acepta 'deviation' (TypeError silencioso que bloqueó
        # ejecuciones, p.ej. señal LONG 2026-06-10 07:22 CT sin orden real). Ademas
        # ORDER_DEVIATION es no-op: broker usa MARKET execution.
        result = place_order(
            direction=mt5_dir, symbol=SYMBOL,
            volume=signal.get("lots", DEFAULT_VOLUME),
            sl=_sl, tp=_tp,
            comment="BRENT_QA", magic=OUR_MAGIC, dry_run=False,
        )
        _log("execution",{"result":result,"direction":mt5_dir,"volume":signal.get("lots"),
                          "fill_price":result.get("price"),"ok":result.get("ok")})
        if result.get("ok"):
            print(f"    ticket={result.get('order')} price={result.get('price')}")
        else:
            print(f"    FAILED | retcode={result.get('retcode')} {result.get('comment')}")

    return {"status":"ok","signal":signal}


def manage_position_fast(execute=False):
    """
    Gestión rápida de posición abierta — corre cada ~45s (no cada 15min).
    Ligero: solo lee posición + ATR, aplica trailing/breakeven. Sin LLM, sin gate.
    Corre INDEPENDIENTE del presupuesto LLM (clave: el trailing debe funcionar
    aunque se hayan agotado las calls del día).
    Devuelve True si hay posición abierta (seguir en modo fast), False si flat.
    """
    try:
        from mt5_bridge import get_positions as _gp, modify_position_sl as _msl
        from portfolio_guard import manage_open_position as _mop
        _pos = _gp(SYMBOL) or []
        _ours = [p for p in _pos if p.get("magic") == OUR_MAGIC]
        if not _ours:
            return False
        _p = _ours[0]
        _bars = fetch_mt5_bars(TIMEFRAME, 20, SYMBOL)
        _atr = float((_bars["high"] - _bars["low"]).tail(14).mean()) if _bars is not None and len(_bars) >= 14 else 0
        _trail = _mop(_p, _atr, close_position, modify_sl_fn=_msl,
                      breakeven_at_r=1.0, trail_at_r=999, trail_distance_r=0.5)
        if _trail:
            print(f"[FAST MGMT] {_trail['action']} #{_trail['ticket']} SL {_trail['old_sl']}->{_trail['new_sl']} ({_trail['profit_r']}R) ok={_trail['ok']}")
            _log("fast_mgmt", _trail)
        return True
    except Exception as _fe:
        print(f"[FAST MGMT] error (no critico): {_fe}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop",     action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--execute",  action="store_true")
    args = parser.parse_args()

    if args.diagnose:
        mt5_diagnose(); return

    execute = args.execute or _CFG_EXECUTE
    if execute:
        print("LIVE EXECUTION ENABLED (WTI demo)")

    if args.loop:
        print(f"WTI loop. Cycle: {CYCLE_MINUTES} min. Prime: {_PRIME_START[0]:02d}:{_PRIME_START[1]:02d}-{_PRIME_END[0]:02d}:{_PRIME_END[1]:02d} CT (Mon-Fri)")
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        with open(os.path.join(ARTIFACTS_DIR,"robot.pid"),"w",encoding="utf-8") as f:
            f.write(str(os.getpid()))
        print(f"PID {os.getpid()} saved")

        _MANAGE_POLL_SEC = 45  # gestion rapida de posicion abierta
        while True:
            try:
                if _is_trading_day():
                    run_cycle(execute=execute)
                else:
                    print("Weekend - WTI closed. Skip.")
            except KeyboardInterrupt:
                print("\nStopped."); break
            except Exception as e:
                print(f"[ERROR] {e}"); traceback.print_exc()
                try:    # stdout va a DEVNULL: si no se loguea, el error es invisible
                    _log("loop_error", {"error": str(e)[:200], "exc_type": type(e).__name__})
                except Exception:
                    pass
            # Sleep adaptativo: si hay posicion abierta, poll rapido (45s) para trailing;
            # si esta flat, espera el ciclo completo de 15 min.
            _cycle_sec = CYCLE_MINUTES * 60
            _elapsed = 0
            print(f"\nSleeping (poll gestion cada {_MANAGE_POLL_SEC}s hasta {CYCLE_MINUTES}min)...")
            while _elapsed < _cycle_sec:
                time.sleep(_MANAGE_POLL_SEC)
                _elapsed += _MANAGE_POLL_SEC
                try:
                    manage_position_fast(execute=execute)
                except KeyboardInterrupt:
                    raise
                except Exception as _me:
                    print(f"[FAST MGMT loop] {_me}")
    else:
        run_cycle(execute=execute)
    mt5_shutdown()


if __name__ == "__main__":
    main()
