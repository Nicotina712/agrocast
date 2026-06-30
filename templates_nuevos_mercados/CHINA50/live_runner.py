"""
US500 S&P 500 â€” Live Runner
Mon-Fri only, prime session 08:30-15:00 CT (09:30-16:00 ET).

Usage:
  python live_runner.py              # one cycle
  python live_runner.py --loop       # continuous
  python live_runner.py --diagnose   # MT5 test
  python live_runner.py --execute    # live orders (only after paper validation!)
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

from config import (
    SYMBOL, CT_OFFSET_HOURS, PRIME_OPEN_CT, PRIME_CLOSE_CT,
    TRADE_WEEKENDS, NO_NEW_SIGNALS_MINS,
    TIMEFRAME, N_BARS_LIVE, MIN_BARS_REQ,
    CYCLE_MINUTES, MAX_LLM_CALLS_PER_DAY,
    EXECUTE_TRADES as _CFG_EXECUTE,
    MIN_SL_PCT,
    MAX_ENTRY_SLIP_PCT,
    DEFAULT_VOLUME,
    SL_ATR_MULT, TP_ATR_MULT,
    WEEKEND_PROFIT_CLOSE, WEEKEND_FLAT_CT,
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

_PRIME_START = (PRIME_OPEN_CT.hour,  PRIME_OPEN_CT.minute)
_PRIME_END   = (PRIME_CLOSE_CT.hour, PRIME_CLOSE_CT.minute)

OUR_MAGIC = 20260612   # CHINA50   # HK50 robot magic number


def _now_ct():
    return datetime.now(timezone.utc) + timedelta(hours=CT_OFFSET_HOURS)

def _is_prime(ct=None):
    if ct is None: ct = _now_ct()
    t     = ct.hour*60 + ct.minute
    start = _PRIME_START[0]*60 + _PRIME_START[1]
    end   = _PRIME_END[0]*60   + _PRIME_END[1]
    if start > end:
        # Ventana cruza medianoche (ej: 22:00-14:00 CT)
        return t >= start or t <= end
    return start <= t <= end

def _mins_to_close(ct=None):
    if ct is None: ct = _now_ct()
    t   = ct.hour*60 + ct.minute
    end = _PRIME_END[0]*60 + _PRIME_END[1]
    start = _PRIME_START[0]*60 + _PRIME_START[1]
    if start > end:
        # Ventana cruza medianoche: si t >= start, faltan (1440-t+end) mins
        if t >= start:
            return max(0, 1440 - t + end)
        return max(0, end - t)
    return max(0, end - t)

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
        "sp500_regime":sig.get("sp500_regime"),"risk_on_off":sig.get("risk_on_off"),
        "price_at_signal":summary.get("current_state",{}).get("close"),
    }
    with open(PAPER_LOG_FILE,"a",encoding="utf-8") as f:
        f.write(json.dumps(entry,ensure_ascii=False,default=str)+"\n")


def _build_fundamental_context():
    ct = _now_ct()
    return {
        "china_macro":         "unknown â€” China PMI, GDP, retail sales, credit data",
        "pboc_policy":         "unknown â€” PBOC rate/RRR decisions, liquidity injections",
        "property_sector":     "unknown â€” China property developers stress (Evergrande-type risk)",
        "tech_megacaps":       "unknown â€” Tencent/Alibaba/Meituan/HSBC drive the index",
        "us_china_relations":  "unknown â€” tariffs, tech export controls, geopolitics",
        "usd_cny":             "unknown â€” CNH/HKD moves, capital flows",
        "session_note":        f"CHINA50 A50 (HKT) | local: {ct.strftime('%H:%M')} | weekday: {ct.weekday()}",
        "key_levels_note":     "Watch round levels: 16000, 17000, 18000, 19000, 20000, 22000",
    }


def run_cycle(execute=False):
    ct = _now_ct()
    is_prime_now = _is_prime(ct)
    mins_left    = _mins_to_close(ct)
    is_wknd      = ct.weekday() >= 5

    print(f"\n{'='*60}")
    print(f"CHINA50 A50 â€” Cycle at {ct.strftime('%Y-%m-%d %H:%M')} HKT")
    print(f"Prime: {is_prime_now} | Weekend: {is_wknd} | Mins left: {mins_left}")
    print("="*60)

    if not _is_trading_day():
        print("Weekend â€” HK50 market closed. Skip.")
        return {"status":"weekend"}

    if not mt5_connected():
        if not mt5_init():
            _log("error",{"msg":"MT5 connection failed"}); return {"status":"mt5_error"}

    acct = get_account_info()
    print(f"Account: ${acct.get('balance','?'):.2f} | Equity: ${acct.get('equity','?'):.2f}")

    # ── WEEKEND PROFIT-CLOSE (06-29): viernes cerca del cierre semanal, si la posición está en
    # GANANCIA, cerrarla para no exponerla al gap del lunes (que atraviesa el SL). Va ANTES de los
    # returns tempranos (vol_filter/storm_breaker) para que dispare siempre. Solo ganadores.
    if WEEKEND_PROFIT_CLOSE and ct.weekday() == 4 and (ct.hour*60+ct.minute) >= (WEEKEND_FLAT_CT.hour*60+WEEKEND_FLAT_CT.minute):
        try:
            from mt5_bridge import get_positions as _gp_wk
            for _wp in [p for p in (_gp_wk(SYMBOL) or []) if p.get("magic") == OUR_MAGIC]:
                if _wp.get("profit", 0) > 0:
                    _wt = _wp.get("ticket")
                    print(f"[WEEKEND CLOSE] Viernes {ct.strftime('%H:%M')} CT | #{_wt} en +{_wp['profit']:.2f} -> cierro antes del finde")
                    _wr = close_position(ticket=_wt, symbol=SYMBOL, comment="weekend_profit_close", magic=OUR_MAGIC, dry_run=not execute)
                    _log("weekend_profit_close", {"ticket":_wt,"profit":_wp.get("profit"),"result":_wr})
                    if _wr.get("ok"):
                        return {"status":"weekend_profit_close","ticket":_wt}
        except Exception as _wke:
            _log("weekend_close_error", {"error": str(_wke)[:120]})

    bars = fetch_mt5_bars(TIMEFRAME, N_BARS_LIVE, SYMBOL)
    if bars is None or len(bars) < MIN_BARS_REQ:
        _log("error",{"msg":f"Not enough bars: {len(bars) if bars is not None else 0}"}); return {"status":"no_data"}
    print(f"Bars: {len(bars)} x {TIMEFRAME} | Latest: {bars.index[-1]}")

    feats   = build_intraday_features(bars, TIMEFRAME)
    summary = summarize_bars(bars, feats)
    price   = summary["current_state"]["close"]

    _log("cycle",{"price":price,"is_prime":is_prime_now,"is_weekend":is_wknd,"mins_left":mins_left,"bars_count":len(bars)})

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
            # ── BE@1R (validado 2026-06-10): SL a entry al +1R (HK50 −0.01→+0.00 expR,
            # neutro en EV pero corta el tail de ganadores que revierten a perdedores).
            try:
                from portfolio_guard import manage_open_position as _mop
                from mt5_bridge import modify_position_sl as _msl
                _atr_pg = float((bars["high"] - bars["low"]).tail(14).mean())
                _mg = _mop(_p, _atr_pg, close_position, modify_sl_fn=_msl,
                           breakeven_at_r=1.0, trail_at_r=999, trail_distance_r=0.5)
                if _mg:
                    print(f"[MGMT] {_mg['action']} #{_mg['ticket']} SL {_mg['old_sl']}->{_mg['new_sl']} ({_mg['profit_r']}R) ok={_mg['ok']}")
                    _log("fast_mgmt", _mg)
            except Exception as _mge:
                _log("mgmt_error", {"error": str(_mge)[:120]})
            _log("position_guard", {"ticket":_ticket,"profit":_profit})
            return {"status":"position_open","ticket":_ticket}
    except Exception as _pg_err:
        print(f"[POSITION GUARD] check failed: {_pg_err}")
        _log("position_guard_error", {"error":str(_pg_err)})
        return {"status":"position_guard_error","error":str(_pg_err)}

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
        _sb = _sbc(window_hours=6.0)
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

    if mins_left < NO_NEW_SIGNALS_MINS:
        print(f"Last {NO_NEW_SIGNALS_MINS} min of prime. No new signals.")
        _log("too_late", {"mins_left": mins_left})
        return {"status":"too_late","mins_left":mins_left}

    state = _load_state()
    can_call, reason = _can_call_llm(state)
    # RESILIENCIA tope-diario: si se agotó la cuota LLM, NO ir dark — seguir con la
    # mecánica (vwap-reversion) sin veto. Solo se marca _llm_capped.
    _llm_capped = not can_call
    if _llm_capped:
        print(f"LLM gate: {reason} -> sigo con mecánica sin veto (resiliencia tope)")
        _log("llm_gate", {"reason": reason, "state_date": state.get("date"),
                          "llm_calls": state.get("llm_calls"), "fallback": "mecánico sin veto"})
    # -- B. position guard MOVIDO arriba (post log de ciclo) 2026-06-29:
    #    la gestion (news_exit + BE@1R/trailing) corre ANTES de los filtros de
    #    entrada (vol_filter/storm_breaker/dormancy) que hacian return temprano.

    # â”€â”€ B2. Cluster cap â€” mÃ¡x 1 posiciÃ³n abierta por cluster correlacionado â”€â”€
    try:
        import sys as _sys, os as _os_pg
        _pg_root = _os_pg.path.dirname(_os_pg.path.dirname(_os_pg.path.abspath(__file__)))
        if _pg_root not in _sys.path: _sys.path.insert(0, _pg_root)
        from portfolio_guard import cluster_cap_blocked as _ccb
        from mt5_bridge import get_positions as _gp_all
        _cc = _ccb(SYMBOL, _gp_all)
        if _cc:
            print(f"[CLUSTER CAP] {_cc['reason']}")
            _log("cluster_cap", _cc)
            return {"status": "cluster_cap", **_cc}
    except Exception:
        pass  # si falla el check, continuamos

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

    # LLM RESILIENTE: si cae -> trend/risk vacíos -> synthesize FLAT -> sin veto -> mecánica igual.
    from portfolio_guard import mechanical_signal as _mgate0
    from config import SL_ATR_MULT as _gsm, TP_ATR_MULT as _gtm
    try: _msig0 = _mgate0(bars, mode="vwap_reversion", sl_atr=_gsm, tp_atr=_gtm, vwap_th=0.8)
    except Exception: _msig0 = True
    if _msig0 is None: print("[COST] sin senal mecanica -> skip veto LLM")
    trend, risk = {}, {}
    if _llm_capped or _msig0 is None:
        # Cuota LLM agotada: sin veto -> trend/risk vacíos -> se toma la mecánica.
        print(f"[RESILIENCIA] cuota LLM agotada -> mecánica sin veto")
    else:
        print("Calling Trend+Risk Agent (CHINA50, veto)...")
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
    print(f"Risk: viable={viable} | {risk.get('volatility_regime')} | ATR: {risk.get('atr_5m')}pts | {risk.get('veto_reason') or 'No veto'}")

    # ── MECHANICAL-PRIMARY (VWAP mean-reversion) + LLM-VETO (2026-06-08) ──────
    # HK50 es MEAN-REVERTING, no trending. El EMA-cross perdia (OOS PF 0.69).
    # VWAP-reversion valido OOS PF 1.61 (112 trades). Ahora la señal de reversion
    # manda; el LLM solo VETA si propone dirección opuesta.
    _llm_sig = synthesize_signal(trend, risk)
    _llm_dir = _llm_sig.get("signal", "FLAT")
    _llm_conf = _llm_sig.get("confidence", "LOW")
    from portfolio_guard import mechanical_signal as _mech
    from config import calc_lots as _calc_lots
    _m = _mech(bars, mode="vwap_reversion", sl_atr=SL_ATR_MULT, tp_atr=TP_ATR_MULT, vwap_th=0.8)
    # ── FILTRO DE CONTEXTO REGIONAL (validado 2026-06-14, robusto multi-proxy + placebo) ──
    # CHINA50 (China A50) lo domina el sentimiento de riesgo regional. El fade VWAP rinde
    # MUCHO mas si se opera EN LA DIRECCION del momentum regional (rebote con viento de cola)
    # y falla fadeando CONTRA una tendencia regional fuerte (cuchillo cayendo).
    # WF: expR +0.36→+0.70 con HK50 mom20 (100% folds+); US500/USTEC confirman; oro NO (placebo).
    # Proxy: HK50 mom 20 barras 30m (mejor), fallback US500. Si no alineado -> skip el fade.
    if _m:
        try:
            _ctx_dir = 0
            for _px in ("HK50", "US500"):
                _pb = fetch_mt5_bars("30m", 25, _px)
                if _pb is not None and len(_pb) >= 21:
                    _pc = _pb["close"].values
                    _ctx_dir = 1 if _pc[-1] > _pc[-21] else -1
                    break
            _sig_dir = 1 if _m["signal"] == "LONG" else -1
            if _ctx_dir != 0 and _ctx_dir != _sig_dir:
                print(f"[CTX FILTER] fade {_m['signal']} contra momentum regional ({_px}) -> skip")
                _log("ctx_filter_skip", {"fade": _m["signal"], "proxy": _px, "ctx_dir": _ctx_dir})
                _m = None
        except Exception as _ce:
            _log("ctx_filter_err", {"error": str(_ce)[:100]})
    if not _m:
        signal = {"signal": "FLAT", "confidence": "LOW", "reasoning": "sin extension VWAP / contra-contexto regional"}
    elif _llm_dir in ("LONG", "SHORT") and _llm_dir != _m["signal"] and _llm_conf == "HIGH":
        # Mean-reversion es contra-tendencia por naturaleza; solo vetar si el LLM ve
        # tendencia opuesta FUERTE (HIGH) — entonces el fade es peligroso (trend day).
        print(f"[LLM VETO] LLM ve {_llm_dir} HIGH contra mean-rev {_m['signal']} -> FLAT (trend fuerte)")
        _log("llm_veto", {"mech": _m["signal"], "llm": _llm_dir, "conf": _llm_conf})
        signal = {"signal": "FLAT", "confidence": "LOW", "reasoning": f"LLM veto {_llm_dir} HIGH"}
    else:
        _stop_pts = abs(_m["entry"] - _m["sl"])
        _lots = _calc_lots(_stop_pts)
        signal = {
            "signal": _m["signal"], "entry": _m["entry"], "sl": _m["sl"], "tp": _m["tp"],
            "lots": _lots, "rr": _m["rr"], "risk_usd": round(_stop_pts * _lots, 2),
            "confidence": "MEDIUM", "reasoning": f"mean-rev {_m['basis']} | LLM={_llm_dir}",
        }
    signal.update({"timestamp":datetime.now().isoformat(),"symbol":SYMBOL,"price":price})
    # â”€â”€ E. Minimum SL distance enforcement (Fix #2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if signal["signal"] != "FLAT":
        _e   = float(signal.get("entry") or 0)
        _sl  = float(signal.get("sl")    or 0)
        _slp = abs(_e - _sl) / _e * 100 if _e else 0
        if _slp < 0.3:
            _min_d = _e * 0.3 / 100
            if signal["signal"] == "LONG":
                signal["sl"] = round(_e - _min_d, 5)
            else:
                signal["sl"] = round(_e + _min_d, 5)
            signal["sl_enforced"] = True
            signal["sl_original"] = _sl
            print(f"[SL GUARD] SL too tight ({_slp:.3f}% < 0.3%). "
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
    state["llm_calls"] += (0 if _llm_capped else 2)  # solo contar llamadas API reales
    state["cycles"] += 1; state["last_signal_time"] = signal["timestamp"]

    # set post-signal cooldown (Option C)
    if signal["signal"] != "FLAT":
        from datetime import datetime as _dt2, timedelta as _td2
        state["signal_cooldown_until"] = (_dt2.now() + _td2(minutes=CYCLE_MINUTES * 2)).isoformat()
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
    _log("signal",{k:signal.get(k) for k in ("signal","entry","sl","tp","lots","confidence","rr","risk_usd","sp500_regime")})

    if signal["signal"] != "FLAT":
        _log_paper(signal, summary)
        print(f"\n>>> SIGNAL: {signal['signal']} @ {signal.get('entry'):,.2f} | SL: {signal.get('sl'):,.2f} | TP: {signal.get('tp'):,.2f}")
        print(f"    Lots: {signal.get('lots')} | R:R: {signal.get('rr')} | Risk: ~${signal.get('risk_usd')}")
        print(f"    Regime: {signal.get('sp500_regime')} | Risk: {signal.get('risk_on_off')} | Conf: {signal.get('confidence')}")
    else:
        print(f"\n>>> FLAT -- {str(signal.get('reasoning',''))[:150]}")

    if execute and signal["signal"] != "FLAT":
        mt5_dir = "BUY" if signal["signal"] == "LONG" else "SELL"
        _sl = float(signal.get("sl")) if signal.get("sl") is not None else None
        _tp = float(signal.get("tp")) if signal.get("tp") is not None else None
        result = place_order(
            direction=mt5_dir, symbol=SYMBOL,
            volume=signal.get("lots", DEFAULT_VOLUME),
            sl=_sl, tp=_tp,
            comment="CHINA50_QA", magic=OUR_MAGIC, dry_run=False,
        )
        _log("execution",{"result":result,"direction":mt5_dir,"volume":signal.get("lots")})
        if result.get("ok"):
            print(f"    ticket={result.get('order')} price={result.get('price')}")
        else:
            print(f"    FAILED | retcode={result.get('retcode')} {result.get('comment')}")

    return {"status":"ok","signal":signal}


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
        print("LIVE EXECUTION ENABLED (HK50 demo)")

    if args.loop:
        print(f"HK50 loop. Cycle: {CYCLE_MINUTES} min. Prime: {_PRIME_START[0]:02d}:{_PRIME_START[1]:02d}-{_PRIME_END[0]:02d}:{_PRIME_END[1]:02d} HKT (Mon-Fri)")
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        with open(os.path.join(ARTIFACTS_DIR,"robot.pid"),"w",encoding="utf-8") as f:
            f.write(str(os.getpid()))
        print(f"PID {os.getpid()} saved")

        while True:
            try:
                if _is_trading_day():
                    run_cycle(execute=execute)
                else:
                    print("Weekend â€” HK50 closed. Skip.")
            except KeyboardInterrupt:
                print("\nStopped."); break
            except Exception as e:
                print(f"[ERROR] {e}"); traceback.print_exc()
                try:    # stdout va a DEVNULL: si no se loguea, el error es invisible
                    _log("loop_error", {"error": str(e)[:200], "exc_type": type(e).__name__})
                except Exception:
                    pass
            print(f"\nSleeping {CYCLE_MINUTES} min..."); time.sleep(CYCLE_MINUTES*60)
    else:
        run_cycle(execute=execute)
    mt5_shutdown()


if __name__ == "__main__":
    main()
