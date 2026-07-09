"""
ETHUSD Ethereum — Live Runner — ⛔ RETIRADO 2026-06-12 (reemplazado por STOXX50).
GUARD: este robot fue retirado. Sale de inmediato sin operar, sin importar quién lo lance
(bat/watchdog/scheduler) — incidente zombies recurrentes 06-13/14. NO quitar este guard.
"""
import sys as _sys
print("ETHUSD RETIRADO — runner deshabilitado (guard anti-zombie 2026-06-14)")
_sys.exit(0)

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
    MIN_SL_PCT, MAX_SL_PCT,
    MAX_ENTRY_SLIP_PCT,
    MAX_HOLD_HOURS,
    DEFAULT_VOLUME,
    SL_ATR_MULT, TP_ATR_MULT,
    SIGNAL_FILE, PAPER_LOG_FILE, LIVE_STATE_FILE, LIVE_LOG_FILE,
    ARTIFACTS_DIR,
)
from mt5_bridge import (
    initialize as mt5_init, shutdown as mt5_shutdown, is_connected as mt5_connected,
    fetch_mt5_bars, get_account_info, get_positions,
    place_order, close_position, diagnose as mt5_diagnose,
)
from microstructure import build_intraday_features, summarize_bars
import importlib as _importlib
import agents as _agents_mod
from agents import call_trend_agent, call_risk_agent, synthesize_signal

_PRIME_START = (PRIME_OPEN_CT.hour,  PRIME_OPEN_CT.minute)
_PRIME_END   = (PRIME_CLOSE_CT.hour, PRIME_CLOSE_CT.minute)

OUR_MAGIC    = 20260603   # ETHUSD robot magic number


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
    return _now_ct().weekday() < 5


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
        "eth_regime":sig.get("eth_regime"),"eth_btc":sig.get("eth_btc"),
        "price_at_signal":summary.get("current_state",{}).get("close"),
    }
    with open(PAPER_LOG_FILE,"a",encoding="utf-8") as f:
        f.write(json.dumps(entry,ensure_ascii=False,default=str)+"\n")


def _build_fundamental_context():
    ct = _now_ct()
    return {
        "eth_ecosystem":    "Post-Merge proof-of-stake; staking yield ~3-5%",
        "eth_btc_ratio":    "unknown — check TradingView ETHBTC",
        "defi_tvl":         "unknown — check DeFiLlama",
        "eth_gas_gwei":     "unknown — check Etherscan Gas Tracker",
        "l2_activity":      "unknown — Arbitrum/Optimism/Base TVL",
        "etf_flows":        "unknown — ETHA/FETH daily flows",
        "btc_trend":        "unknown — check BTCUSD (ETH high-beta to BTC)",
        "dxy_trend":        "unknown — inverse correlation",
        "session_note":     f"ETH 24/7 | CT: {ct.strftime('%H:%M')} | weekend: {ct.weekday()>=5}",
        "key_levels_note":  "Watch: $1.8k, $2k, $2.2k, $2.5k, $3k psychological levels",
    }


def run_cycle(execute=False):
    # Hot-reload agents.py so code changes are picked up without restarting the loop
    global call_trend_agent, call_risk_agent, synthesize_signal
    try:
        _importlib.reload(_agents_mod)
        call_trend_agent  = _agents_mod.call_trend_agent
        call_risk_agent   = _agents_mod.call_risk_agent
        synthesize_signal = _agents_mod.synthesize_signal
    except Exception as _re:
        print(f"[WARN] agents.py reload failed: {_re}")

    ct = _now_ct()
    is_prime_now = _is_prime(ct)
    mins_left    = _mins_to_close(ct)
    is_wknd      = ct.weekday() >= 5

    print(f"\n{'='*60}")
    print(f"ETHUSD Ethereum — Cycle at {ct.strftime('%Y-%m-%d %H:%M')} CT")
    print(f"Prime: {is_prime_now} | Weekend: {is_wknd} | Mins left: {mins_left}")
    print("="*60)

    if not mt5_connected():
        if not mt5_init():
            _log("error",{"msg":"MT5 connection failed"}); return {"status":"mt5_error"}

    acct = get_account_info()
    print(f"Account: ${acct.get('balance','?'):.2f} | Equity: ${acct.get('equity','?'):.2f}")

    bars = fetch_mt5_bars(TIMEFRAME, N_BARS_LIVE, SYMBOL)
    if bars is None or len(bars) < MIN_BARS_REQ:
        _log("error",{"msg":f"Not enough bars: {len(bars) if bars is not None else 0}"}); return {"status":"no_data"}
    print(f"Bars: {len(bars)} × {TIMEFRAME} | Latest: {bars.index[-1]}")

    feats   = build_intraday_features(bars, TIMEFRAME)
    summary = summarize_bars(bars, feats)
    price   = summary["current_state"]["close"]

    _log("cycle",{"price":price,"is_prime":is_prime_now,"is_weekend":is_wknd,"mins_left":mins_left,"bars_count":len(bars)})

    # ── A2. ATR volatility filter ─────────────────────────────────────────────
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
            print(f"[STORM BREAKER] {_sb['reason']}")
            _log("storm_breaker", _sb)
            return {"status": "storm_breaker", **_sb}
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
        print(f"Outside prime ({_PRIME_START[0]:02d}:{_PRIME_START[1]:02d}–{_PRIME_END[0]:02d}:{_PRIME_END[1]:02d} CT). Monitor only.")
        return {"status":"outside_prime","price":price}

    if mins_left < NO_NEW_SIGNALS_MINS:
        print(f"Last {NO_NEW_SIGNALS_MINS} min of prime. No new signals.")
        return {"status":"too_late","mins_left":mins_left}

    state = _load_state()
    can_call, reason = _can_call_llm(state)
    if not can_call:
        print(f"LLM gate: {reason}"); return {"status":"llm_gate","reason":reason}
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
            _log("position_guard", {"ticket":_ticket,"profit":_profit})
            return {"status":"position_open","ticket":_ticket}
    except Exception as _pg_err:
        print(f"[POSITION GUARD] check failed: {_pg_err}")
        _log("position_guard_error", {"error":str(_pg_err)})
        return {"status":"position_guard_error","error":str(_pg_err)}

    # ── B2. Cluster cap — máx 1 posición abierta por cluster correlacionado ──
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


    fund_ctx = _build_fundamental_context()

    # LLM RESILIENTE: si cae -> trend/risk vacíos -> synthesize FLAT -> sin veto -> mecánica 1h igual.
    trend, risk = {}, {}
    print("Calling Trend+Risk Agent (ETH, veto)...")
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
    print(f"Risk: viable={viable} | {risk.get('volatility_regime')} | ATR30m: ${risk.get('atr_30m') or risk.get('atr_5m')} | {risk.get('veto_reason') or 'No veto'}")

    # ── MECHANICAL-PRIMARY (trend 1h) + LLM-VETO (2026-06-08) ────────────────
    # ETH funciona en 1h (no 15m): trend EMA20/100 meanPF 1.49, consist 83% (walk-forward).
    # La señal trend mecánica manda; el LLM solo VETA si ve reversión opuesta con conf HIGH
    # (captura regímenes malos donde el trend pierde).
    _llm_sig = synthesize_signal(trend, risk)
    _llm_dir = _llm_sig.get("signal", "FLAT")
    _llm_conf = _llm_sig.get("confidence", "LOW")
    from portfolio_guard import mechanical_signal as _mech
    from config import calc_lots as _calc_lots
    from config import EMA_FAST as _EF, EMA_SLOW as _ES
    _m = _mech(bars, mode="trend", sl_atr=SL_ATR_MULT, tp_atr=TP_ATR_MULT, ema_fast=_EF, ema_slow=_ES)
    # ── FILTRO DE AGOTAMIENTO (validado 2026-06-11): no perseguir movimientos viejos.
    # Si el precio ya se movió >3% en 48h EN LA DIRECCION de la señal, la entrada es
    # tardía: histórico ETH expR -0.07 (N=116) vs +0.40/+0.75 fresco/medio. El 06-11
    # los 2 shorts de ETH entraron así y la reversión en V los barrió.
    if _m and len(bars) > 49:
        try:
            _pre = (float(bars["close"].iloc[-1]) / float(bars["close"].iloc[-49]) - 1) * 100
            _pre_dir = _pre if _m["signal"] == "LONG" else -_pre
            if _pre_dir > 3.0:
                print(f"[EXHAUSTION] mov 48h {_pre_dir:+.1f}% en direccion {_m['signal']} > 3% -> entrada tardia, skip")
                _log("exhaustion_block", {"signal": _m["signal"], "pre_move_pct": round(_pre_dir, 2)})
                _m = None
        except Exception:
            pass
    if not _m:
        signal = {"signal": "FLAT", "confidence": "LOW", "reasoning": "trend 1h FLAT"}
    elif _llm_dir in ("LONG", "SHORT") and _llm_dir != _m["signal"] and _llm_conf == "HIGH":
        print(f"[LLM VETO] LLM ve {_llm_dir} HIGH contra trend {_m['signal']} -> FLAT")
        _log("llm_veto", {"mech": _m["signal"], "llm": _llm_dir, "conf": _llm_conf})
        signal = {"signal": "FLAT", "confidence": "LOW", "reasoning": f"LLM veto {_llm_dir} HIGH"}
    else:
        _stop_pts = abs(_m["entry"] - _m["sl"])
        _lots = _calc_lots(_stop_pts)
        signal = {
            "signal": _m["signal"], "entry": _m["entry"], "sl": _m["sl"], "tp": _m["tp"],
            "lots": _lots, "rr": _m["rr"], "risk_usd": round(_stop_pts * _lots, 2),
            "confidence": "MEDIUM", "eth_regime": trend.get("eth_regime", ""), "eth_btc": "",
            "reasoning": f"mecánico trend 1h {_m['basis']} | LLM={_llm_dir}",
        }
    signal.update({"timestamp":datetime.now().isoformat(),"symbol":SYMBOL,"price":price})
    # ── E. Minimum SL distance enforcement (Fix #2) ──────────────────────────
    if signal["signal"] != "FLAT":
        _e   = float(signal.get("entry") or 0)
        _sl  = float(signal.get("sl")    or 0)
        _slp = abs(_e - _sl) / _e * 100 if _e else 0
        if _slp < 0.8:
            _min_d = _e * 0.8 / 100
            if signal["signal"] == "LONG":
                signal["sl"] = round(_e - _min_d, 5)
            else:
                signal["sl"] = round(_e + _min_d, 5)
            signal["sl_enforced"] = True
            signal["sl_original"] = _sl
            print(f"[SL GUARD] SL too tight ({_slp:.3f}% < 0.8%). "
                  f"Adjusted: {_sl:.5f} -> {signal['sl']:.5f}")
            # Recompute R:R with expanded SL — veto if no longer viable
            _new_risk   = abs(_e - float(signal["sl"]))
            _new_reward = abs(float(signal.get("tp", 0)) - _e)
            _real_rr    = round(_new_reward / _new_risk, 2) if _new_risk > 0 else 0
            signal["rr"] = _real_rr
            if _real_rr < 2.0:
                print(f"[SL GUARD] Real R:R after SL expansion = {_real_rr:.2f} < 2.0 — VETO")
                signal["signal"] = "FLAT"
                signal["veto_reason"] = f"SL expansion destroyed R:R: {_real_rr:.2f}"

    # ── E2. Maximum SL distance enforcement (cap SL a MAX_SL_PCT) ────────────
    if signal["signal"] != "FLAT":
        _e   = float(signal.get("entry") or 0)
        _sl  = float(signal.get("sl")    or 0)
        _slp = abs(_e - _sl) / _e * 100 if _e else 0
        if _slp > MAX_SL_PCT:
            _max_d = _e * MAX_SL_PCT / 100
            _sl_orig = _sl
            if signal["signal"] == "LONG":
                signal["sl"] = round(_e - _max_d, 2)
            else:
                signal["sl"] = round(_e + _max_d, 2)
            signal["sl_capped"] = True
            signal["sl_original"] = _sl_orig
            print(f"[SL CAP] SL demasiado ancho ({_slp:.2f}% > {MAX_SL_PCT}%). "
                  f"Reducido: {_sl_orig:.2f} -> {signal['sl']:.2f}")
            _log("sl_cap", {"original_sl": _sl_orig, "capped_sl": signal["sl"],
                            "sl_pct": round(_slp, 3), "max_sl_pct": MAX_SL_PCT})
            # Verificar R:R sigue siendo valido
            _new_risk   = abs(_e - float(signal["sl"]))
            _new_reward = abs(float(signal.get("tp", 0)) - _e)
            _real_rr    = round(_new_reward / _new_risk, 2) if _new_risk > 0 else 0
            signal["rr"] = _real_rr
            if _real_rr < 1.5:
                print(f"[SL CAP] R:R tras cap = {_real_rr:.2f} < 1.5 — VETO")
                signal["signal"] = "FLAT"
                signal["veto_reason"] = f"SL cap destruyo R:R: {_real_rr:.2f}"

    
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

    # set post-signal cooldown (Option C)
    if signal["signal"] != "FLAT":
        from datetime import datetime as _dt2, timedelta as _td2
        state["signal_cooldown_until"] = (_dt2.now() + _td2(minutes=CYCLE_MINUTES * 2)).isoformat()
    else:
        state.pop("signal_cooldown_until", None)
    _save_state(state)
    _save_signal(signal)
    _log("signal",{k:signal.get(k) for k in ("signal","entry","sl","tp","lots","confidence","rr","risk_usd","eth_regime")})

    if signal["signal"] != "FLAT":
        _log_paper(signal, summary)
        print(f"\n>>> SIGNAL: {signal['signal']} @ ${signal.get('entry'):,.2f} | SL: ${signal.get('sl'):,.2f} | TP: ${signal.get('tp'):,.2f}")
        print(f"    Lots: {signal.get('lots')} ETH | R:R: {signal.get('rr')} | Risk: ~${signal.get('risk_usd')}")
        print(f"    Regime: {signal.get('eth_regime')} | ETH/BTC: {signal.get('eth_btc')} | Conf: {signal.get('confidence')}")
    else:
        print(f"\n>>> FLAT — {str(signal.get('reasoning',''))[:150]}")

    if execute and signal["signal"] != "FLAT":
        mt5_dir = "BUY" if signal["signal"] == "LONG" else "SELL"
        _sl = float(signal.get("sl")) if signal.get("sl") is not None else None
        _tp = float(signal.get("tp")) if signal.get("tp") is not None else None
        result = place_order(
            direction=mt5_dir, symbol=SYMBOL,
            volume=signal.get("lots", DEFAULT_VOLUME),
            sl=_sl, tp=_tp,
            comment="ETH_QA", magic=OUR_MAGIC, dry_run=False,
        )
        _log("execution",{"result":result,"direction":mt5_dir,"volume":signal.get("lots")})
        if result.get("ok"):
            print(f"    ✓ ticket={result.get('order')} price={result.get('price')}")
        else:
            print(f"    ✗ FAILED | retcode={result.get('retcode')} {result.get('comment')}")

    return {"status":"ok","signal":signal}


def manage_position_fast(execute=False):
    """Gestion rapida de posicion abierta cada ~45s (trailing/breakeven, sin LLM)."""
    try:
        from mt5_bridge import get_positions as _gp, modify_position_sl as _msl
        from portfolio_guard import manage_open_position as _mop
        _pos = _gp(SYMBOL) or []
        _ours = [p for p in _pos if p.get("magic") == OUR_MAGIC]
        if not _ours:
            return False
        _p = _ours[0]
        # ── MAX_HOLD: cierre por tiempo (libera slot crypto, desbloquea BTC) ──
        try:
            from datetime import datetime as _dtn, timezone as _tzn
            _ptime = _p.get("time")
            if _ptime is not None:
                _age_h = (_dtn.now(_tzn.utc) - _ptime).total_seconds() / 3600
                if _age_h >= MAX_HOLD_HOURS:
                    print(f"[MAX HOLD] Posicion ETH abierta {_age_h:.1f}h >= {MAX_HOLD_HOURS}h -> cierre por tiempo")
                    _cr = close_position(ticket=_p["ticket"], symbol=SYMBOL,
                                         comment="max_hold", magic=OUR_MAGIC, dry_run=not execute)
                    _log("max_hold_exit", {"ticket": _p["ticket"], "age_h": round(_age_h, 1),
                                           "pnl": _p.get("profit"), "ok": _cr.get("ok")})
                    return True
        except Exception as _mhe:
            print(f"[MAX HOLD] error (no critico): {_mhe}")
        _bars = fetch_mt5_bars(TIMEFRAME, 20, SYMBOL)
        _atr = float((_bars["high"] - _bars["low"]).tail(14).mean()) if _bars is not None and len(_bars) >= 14 else 0
        _trail = _mop(_p, _atr, close_position, modify_sl_fn=_msl,
                      breakeven_at_r=999, trail_at_r=999, trail_distance_r=0.5)
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
        print("⚠ LIVE EXECUTION ENABLED (ETH demo)")

    if args.loop:
        print(f"ETH loop. Cycle: {CYCLE_MINUTES} min. Prime: {_PRIME_START[0]:02d}:{_PRIME_START[1]:02d}–{_PRIME_END[0]:02d}:{_PRIME_END[1]:02d} CT (24/7)")
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        with open(os.path.join(ARTIFACTS_DIR,"robot.pid"),"w",encoding="utf-8") as f:
            f.write(str(os.getpid()))
        print(f"PID {os.getpid()} saved")

        _MANAGE_POLL_SEC = 45
        while True:
            try:
                if _is_trading_day():
                    run_cycle(execute=execute)
                else:
                    print("Weekend disabled - skip.")
            except KeyboardInterrupt:
                print("\nStopped."); break
            except Exception as e:
                print(f"[ERROR] {e}"); traceback.print_exc()
                try:    # stdout va a DEVNULL: si no se loguea, el error es invisible
                    _log("loop_error", {"error": str(e)[:200], "exc_type": type(e).__name__})
                except Exception:
                    pass
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
