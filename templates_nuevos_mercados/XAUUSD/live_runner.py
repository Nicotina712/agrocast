"""
XAUUSD Gold — Live Runner
Main loop: polls MT5 every cycle during London-NY session.

Usage:
  cd templates_nuevos_mercados/XAUUSD
  python live_runner.py              # one cycle
  python live_runner.py --loop       # continuous loop
  python live_runner.py --diagnose   # MT5 connection test
  python live_runner.py --execute    # live execution (only after paper trading!)
"""

import os
import sys
import io
import json
import time
import argparse
import traceback
from datetime import datetime, date, timedelta, timezone

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from config import (
    SYMBOL, RTH_OPEN_CT, RTH_CLOSE_CT, CT_OFFSET_HOURS,
    TIMEFRAME, N_BARS_LIVE, MIN_BARS_REQ,
    CYCLE_MINUTES, MAX_LLM_CALLS_PER_DAY,
    EXECUTE_TRADES as _CFG_EXECUTE,
    MIN_SL_PCT,
    MAX_ENTRY_SLIP_PCT,
    DEFAULT_VOLUME,
    SIGNAL_FILE, PAPER_LOG_FILE, LIVE_STATE_FILE, LIVE_LOG_FILE,
    ARTIFACTS_DIR,
)
from mt5_bridge import (
    initialize as mt5_init,
    shutdown as mt5_shutdown,
    is_connected as mt5_connected,
    fetch_mt5_bars,
    get_live_tick,
    get_account_info,
    get_positions,
    place_order, close_position,
    diagnose as mt5_diagnose,
)
from microstructure import build_intraday_features, summarize_bars
from agents import call_trend_agent, call_risk_agent, synthesize_signal


# ─── Constants ───────────────────────────────────────────────────────────────

OUR_MAGIC  = 20260602   # XAUUSD robot magic number

_RTH_START = (RTH_OPEN_CT.hour,  RTH_OPEN_CT.minute)   # (7, 0)
_RTH_END   = (RTH_CLOSE_CT.hour, RTH_CLOSE_CT.minute)  # (11, 30)


# ─── Time helpers ─────────────────────────────────────────────────────────────

def _now_ct() -> datetime:
    utc_now = datetime.now(timezone.utc)
    return utc_now + timedelta(hours=CT_OFFSET_HOURS)


def _is_rth(ct: datetime = None) -> bool:
    if ct is None:
        ct = _now_ct()
    t     = ct.hour * 60 + ct.minute
    start = _RTH_START[0] * 60 + _RTH_START[1]
    end   = _RTH_END[0]   * 60 + _RTH_END[1]
    return start <= t <= end


def _mins_to_close(ct: datetime = None) -> int:
    if ct is None:
        ct = _now_ct()
    t   = ct.hour * 60 + ct.minute
    end = _RTH_END[0] * 60 + _RTH_END[1]
    return max(0, end - t)


def _is_weekday() -> bool:
    wd = _now_ct().weekday()
    if wd < 5: return True
    if wd == 5: return False
    return _now_ct().hour >= 17  # Sunday: ICMarkets opens ~17:00 CT


# ─── Live state ───────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if os.path.exists(LIVE_STATE_FILE):
        try:
            with open(LIVE_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"date": "", "llm_calls": 0, "cycles": 0, "last_signal_time": None}


def _save_state(state: dict):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(LIVE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def _can_call_llm(state: dict) -> tuple[bool, str]:
    today = date.today().isoformat()
    if state.get("date") != today:
        state["date"]      = today
        state["llm_calls"] = 0
        state["cycles"]    = 0
    if state["llm_calls"] >= MAX_LLM_CALLS_PER_DAY:
        return False, f"max {MAX_LLM_CALLS_PER_DAY} LLM calls/day reached"
    return True, "ok"


# ─── Logging ──────────────────────────────────────────────────────────────────

def _log(event_type: str, data: dict):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "ct_time":   _now_ct().strftime("%H:%M"),
        "type":      event_type,
        **data,
    }
    try:
        with open(LIVE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
    print(f"[{entry['ct_time']} CT] [{event_type}] {json.dumps({k: v for k, v in data.items() if k not in ('trend_analysis', 'risk_analysis')}, default=str)}")


def _save_signal(signal: dict):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(SIGNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(signal, f, indent=2, ensure_ascii=False, default=str)


def _log_paper_trade(signal: dict, bars_summary: dict):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "symbol":    SYMBOL,
        "signal":    signal.get("signal"),
        "entry":     signal.get("entry"),
        "sl":        signal.get("sl"),
        "tp":        signal.get("tp"),
        "lots":      signal.get("lots"),
        "confidence": signal.get("confidence"),
        "rr":        signal.get("rr"),
        "risk_usd":  signal.get("risk_usd"),
        "reasoning": signal.get("reasoning"),
        "price_at_signal": bars_summary.get("current_state", {}).get("close"),
    }
    with open(PAPER_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ─── Fundamental context ──────────────────────────────────────────────────────

def _build_fundamental_context() -> dict:
    """
    Build Gold fundamental context.
    Currently returns static structure — enhance with live DXY/yield data later.

    TODO: fetch DXY tick from MT5 (USDX or DXY symbol) and add to context.
    TODO: integrate economic calendar API for FOMC/NFP dates.
    """
    ct = _now_ct()
    return {
        "daily_trend":       "unknown — check daily chart",
        "dxy_trend":         "unknown — check DXY/USDX",
        "real_yields":       "unknown — check TIP/TNX",
        "vix_level":         "unknown",
        "next_macro_event":  "Check economic calendar for FOMC/NFP/CPI",
        "overall_bias":      "neutral",
        "session_note":      f"London-NY overlap, CT: {ct.strftime('%H:%M')}",
    }


# ─── Main cycle ───────────────────────────────────────────────────────────────

def run_cycle(execute: bool = False) -> dict:
    """
    One complete analysis cycle:
    1. Fetch bars from MT5
    2. Build features
    3. Call Trend + Risk agents
    4. Synthesize signal
    5. Log paper trade (+ execute if enabled)
    """
    ct = _now_ct()
    print(f"\n{'='*60}")
    print(f"XAUUSD Gold — Cycle at {ct.strftime('%Y-%m-%d %H:%M')} CT")
    print(f"Is RTH (London-NY): {_is_rth(ct)} | Mins to close: {_mins_to_close(ct)}")
    print(f"{'='*60}")

    # ── 1. MT5 connection ────────────────────────────────────────────────────
    if not mt5_connected():
        ok = mt5_init()
        if not ok:
            _log("error", {"msg": "MT5 connection failed"})
            return {"status": "mt5_error"}

    # ── 2. Account info ──────────────────────────────────────────────────────
    acct = get_account_info()
    print(f"Account: ${acct.get('balance', '?'):.2f} | Equity: ${acct.get('equity', '?'):.2f}")

    # ── 3. Fetch bars ────────────────────────────────────────────────────────
    bars = fetch_mt5_bars(TIMEFRAME, N_BARS_LIVE, SYMBOL)
    if bars is None or len(bars) < MIN_BARS_REQ:
        _log("error", {"msg": f"Not enough bars: {len(bars) if bars is not None else 0}"})
        return {"status": "no_data"}

    print(f"Bars fetched: {len(bars)} × {TIMEFRAME} | Latest: {bars.index[-1]}")

    # ── 4. Features ──────────────────────────────────────────────────────────
    feats = build_intraday_features(bars, TIMEFRAME)
    summary = summarize_bars(bars, feats)

    current_price = summary.get("current_state", {}).get("close", 0)
    is_rth_now    = summary.get("current_state", {}).get("is_rth", False)
    mins_left     = _mins_to_close(ct)   # real-time clock, not bar data

    _log("cycle", {
        "price": current_price,
        "is_rth": _is_rth(ct),            # real-time clock for log accuracy
        "mins_to_close": mins_left,
        "bars_count": len(bars),
    })

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

    # ── 5. RTH guard: no new signals outside session or last 30 min ──────────
    # Use real-time clock as primary check; is_rth_now from bars as secondary.
    rth_real = _is_rth(ct)
    if not rth_real:
        print(f"Outside Gold RTH window ({_RTH_START[0]:02d}:{_RTH_START[1]:02d}–{_RTH_END[0]:02d}:{_RTH_END[1]:02d} CT). Monitoring only.")
        return {"status": "outside_rth", "price": current_price}

    if mins_left < 30:
        print(f"Last 30 min of session ({mins_left} min left). No new signals.")
        return {"status": "too_late", "mins_left": mins_left}

    # ── 6. LLM cost gate ─────────────────────────────────────────────────────
    state = _load_state()
    can_call, reason = _can_call_llm(state)
    # RESILIENCIA tope-diario: si se agotó la cuota LLM, NO ir dark — seguir con la
    # mecánica (trend 4h LONG-only) sin veto. Solo se marca _llm_capped.
    _llm_capped = not can_call
    if _llm_capped:
        print(f"LLM gate: {reason} -> sigo con mecánica sin veto (resiliencia tope)")
        _log("llm_gate", {"reason": reason, "llm_calls": state.get("llm_calls"),
                          "fallback": "mecánico sin veto"})
    # -- B. position guard MOVIDO arriba (post log de ciclo) 2026-06-29:
    #    la gestion (news_exit + BE@1R/trailing) corre ANTES de los filtros de
    #    entrada (vol_filter/storm_breaker/dormancy) que hacian return temprano.

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


    # ── 7. Fundamental context ────────────────────────────────────────────────

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

    fund_ctx = _build_fundamental_context()

    # ── 8-10. MECHANICAL-PRIMARY (trend 4h 50/200 LONG-ONLY) + LLM-VETO (2026-06-09)
    # Walk-forward 8 folds 2022-2026: meanPF 3.13, consist 88%, expR +1.38 (N=207).
    # SOLO LONG: el short mecánico pierde (PF 0.74). Si EMA50<200 → FLAT (no shortea).
    # El LLM solo VETA un LONG si ve bajista con conf HIGH (riesgo macro/desplome).
    # Resiliencia: LLM caído o cuota agotada → se toma la mecánica sin veto.
    from portfolio_guard import mechanical_signal as _mgate0
    from config import EMA_FAST as _gef, EMA_SLOW as _ges, SL_ATR_MULT as _gsm, TP_ATR_MULT as _gtm
    try: _msig0 = _mgate0(bars, mode="trend", sl_atr=_gsm, tp_atr=_gtm, ema_fast=_gef, ema_slow=_ges)
    except Exception: _msig0 = True
    if _msig0 is None: print("[COST] sin senal mecanica -> skip veto LLM")
    trend, risk = {}, {}
    if _llm_capped or _msig0 is None:
        print(f"[RESILIENCIA] cuota LLM agotada -> mecánica sin veto")
    else:
        try:
            print("Calling Trend+Risk Agent (XAU, veto)...")
            trend = call_trend_agent(summary, fund_ctx)
            if "error" in trend: raise RuntimeError(f"trend {trend['error']} | RAW={str(trend.get('raw'))[:300]}")
            risk = call_risk_agent(summary, trend, fund_ctx)
            if "error" in risk: raise RuntimeError(f"risk {risk['error']} | RAW={str(risk.get('raw'))[:300]}")
        except Exception as _exc:
            trend, risk = {}, {}
            _log("agent_error", {"error": str(_exc)[:400], "fallback": "mecánico sin veto"})
            print(f"[RESILIENCIA] LLM caído ({str(_exc)[:50]}) -> mecánica sin veto")

    _llm_sig = synthesize_signal(trend, risk)
    _llm_dir = _llm_sig.get("signal", "FLAT")
    _llm_conf = _llm_sig.get("confidence", "LOW")
    from portfolio_guard import mechanical_signal as _mech
    from config import calc_lots as _calc_lots, EMA_FAST as _EF, EMA_SLOW as _ES, \
        SL_ATR_MULT as _SLM, TP_ATR_MULT as _TPM, LONG_ONLY as _LONG_ONLY
    _m = _mech(bars, mode="trend", sl_atr=_SLM, tp_atr=_TPM, ema_fast=_EF, ema_slow=_ES)
    if _m and _LONG_ONLY and _m["signal"] == "SHORT":
        print("[LONG-ONLY] señal mecánica SHORT descartada (edge solo LONG en oro)")
        _log("long_only_skip", {"mech": "SHORT"})
        _m = None
    # ── FILTRO DE CONTEXTO USD (validado 2026-06-14 multi-lookback + placebo, Δ+0.78 100% folds):
    # el trend del oro rinde mucho mas si su direccion se alinea con EURUSD (proxy inverso del USD).
    if _m:
        try:
            from portfolio_guard import context_aligned as _ctxa
            _al = _ctxa("EURUSD", _m["signal"])
            if _al is False:
                print(f"[CTX FILTER] {_m['signal']} contra momentum USD (EURUSD) -> skip")
                _log("ctx_filter_skip", {"mech": _m["signal"], "proxy": "EURUSD"}); _m = None
        except Exception as _ce:
            _log("ctx_filter_err", {"error": str(_ce)[:100]})
    if not _m:
        signal = {"signal": "FLAT", "confidence": "LOW", "reasoning": "trend 4h sin señal LONG / contra-USD"}
    elif _llm_dir == "SHORT" and _llm_conf == "HIGH":
        print(f"[LLM VETO] LLM ve SHORT HIGH contra LONG mecánico -> FLAT (riesgo macro)")
        _log("llm_veto", {"mech": "LONG", "llm": _llm_dir, "conf": _llm_conf})
        signal = {"signal": "FLAT", "confidence": "LOW", "reasoning": "LLM veto SHORT HIGH"}
    else:
        _stop_pts = abs(_m["entry"] - _m["sl"])
        _lots = _calc_lots(_stop_pts)
        signal = {
            "signal": _m["signal"], "entry": _m["entry"], "sl": _m["sl"], "tp": _m["tp"],
            "lots": _lots, "rr": _m["rr"], "risk_usd": round(_stop_pts * _lots * 100, 2),
            "confidence": "MEDIUM", "reasoning": f"trend 4h {_m['basis']} | LLM={_llm_dir}",
        }
    signal["timestamp"] = datetime.now().isoformat()
    signal["symbol"]    = SYMBOL
    signal["price"]     = current_price
    # ── E. Minimum SL distance enforcement (Fix #2) ──────────────────────────
    if signal["signal"] != "FLAT":
        _e   = float(signal.get("entry") or 0)
        _sl  = float(signal.get("sl")    or 0)
        _slp = abs(_e - _sl) / _e * 100 if _e else 0
        if _slp < 0.15:
            _min_d = _e * 0.15 / 100
            if signal["signal"] == "LONG":
                signal["sl"] = round(_e - _min_d, 5)
            else:
                signal["sl"] = round(_e + _min_d, 5)
            signal["sl_enforced"] = True
            signal["sl_original"] = _sl
            print(f"[SL GUARD] SL too tight ({_slp:.3f}% < 0.15%). "
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

    
    # -- H. Entry slippage gate ------------------------------------------------
    # Si el precio actual es adverso respecto al entry calculado por el LLM,
    # significa que el mercado ya se movio en nuestra contra antes de entrar.
    # SHORT: precio DEBAJO del entry (ya en el fondo, no en la resistencia)
    # LONG:  precio ENCIMA del entry (ya en el techo, no en el soporte)
    # Se veta la senal para no perseguir precios extendidos.
    if signal["signal"] != "FLAT":
        _entry_h = float(signal.get("entry") or current_price)
        if _entry_h > 0 and current_price > 0:
            if signal["signal"] == "SHORT":
                _adverse_h = (_entry_h - current_price) / _entry_h * 100
            else:  # LONG
                _adverse_h = (current_price - _entry_h) / _entry_h * 100
            if _adverse_h > MAX_ENTRY_SLIP_PCT:
                print(f"[ENTRY GATE] Slippage adverso {_adverse_h:.3f}% > {MAX_ENTRY_SLIP_PCT}% -- veto "
                      f"(entry_llm={_entry_h} precio={current_price:.4f})")
                _log("entry_gate_veto", {"signal": signal["signal"], "entry_llm": _entry_h,
                                         "price": round(current_price, 5), "adverse_pct": round(_adverse_h, 4),
                                         "threshold": MAX_ENTRY_SLIP_PCT})
                signal["signal"]      = "FLAT"
                signal["veto_reason"] = (f"Entry gate: slippage adverso {_adverse_h:.2f}% > "
                                          f"{MAX_ENTRY_SLIP_PCT}% (entry={_entry_h}, precio={current_price:.4f})")
    # -------------------------------------------------------------------------
    state["llm_calls"] += (0 if _llm_capped else 2)  # solo contar llamadas API reales  # trend + risk
    state["cycles"]    += 1
    state["last_signal_time"] = signal["timestamp"]

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
    _log("signal", {
        "signal":     signal["signal"],
        "entry":      signal.get("entry"),
        "sl":         signal.get("sl"),
        "tp":         signal.get("tp"),
        "lots":       signal.get("lots"),
        "confidence": signal.get("confidence"),
        "rr":         signal.get("rr"),
        "risk_usd":   signal.get("risk_usd"),
    })

    # ── 11. Paper log ─────────────────────────────────────────────────────────
    if signal["signal"] != "FLAT":
        _log_paper_trade(signal, summary)
        print(f"\n>>> SIGNAL: {signal['signal']} @ ${signal.get('entry')} | SL: {signal.get('sl')} | TP: {signal.get('tp')} | {signal.get('lots')} lots | R:R {signal.get('rr')}")
        print(f"    Risk: ~${signal.get('risk_usd')} | Confidence: {signal.get('confidence')}")
        print(f"    Reasoning: {signal.get('reasoning')[:120]}...")
    else:
        print(f"\n>>> FLAT — {signal.get('reasoning', 'No setup.')[:120]}")

    # ── 12. Execute (only if enabled) ─────────────────────────────────────────
    if execute and signal["signal"] != "FLAT":
        # Map LONG/SHORT → BUY/SELL (MT5 convention)
        mt5_direction = "BUY" if signal["signal"] == "LONG" else "SELL"
        volume    = signal.get("lots", DEFAULT_VOLUME)
        sl_price  = float(signal.get("sl")) if signal.get("sl") is not None else None
        tp_price  = float(signal.get("tp")) if signal.get("tp") is not None else None

        print(f"\n>>> EXECUTING LIVE ORDER: {mt5_direction} {volume} lots {SYMBOL}")
        result = place_order(
            direction=mt5_direction,
            symbol=SYMBOL,
            volume=volume,
            sl=sl_price,
            tp=tp_price,
            comment="GOLD_QA",
            magic=OUR_MAGIC,
            dry_run=False,          # ← ejecución real
        )
        _log("execution", {"result": result, "direction": mt5_direction, "volume": volume, "sl": sl_price, "tp": tp_price})
        if result.get("ok"):
            print(f"    ✓ Order placed | ticket={result.get('order')} price={result.get('price')}")
        else:
            print(f"    ✗ Order FAILED | retcode={result.get('retcode')} {result.get('comment')}")

    return {"status": "ok", "signal": signal}


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XAUUSD Gold Live Runner")
    parser.add_argument("--loop",     action="store_true", help="Run continuously every CYCLE_MINUTES")
    parser.add_argument("--diagnose", action="store_true", help="MT5 connection diagnostic")
    parser.add_argument("--execute",  action="store_true", help="Enable live trade execution")
    args = parser.parse_args()

    if args.diagnose:
        mt5_diagnose()
        return

    execute = args.execute or _CFG_EXECUTE
    if execute:
        print("⚠ LIVE EXECUTION ENABLED — trades will be placed in MT5")

    if args.loop:
        print(f"Starting Gold live loop. Cycle: {CYCLE_MINUTES} min. RTH: {_RTH_START[0]:02d}:{_RTH_START[1]:02d}–{_RTH_END[0]:02d}:{_RTH_END[1]:02d} CT")
        # Write PID so watchdog can monitor this process
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        pid_file = os.path.join(ARTIFACTS_DIR, "robot.pid")
        with open(pid_file, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        print(f"PID {os.getpid()} saved to {pid_file}")
        while True:
            try:
                if _is_weekday():
                    run_cycle(execute=execute)
                else:
                    print("Weekend — skipping cycle.")
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except Exception as e:
                print(f"[ERROR] Cycle failed: {e}")
                traceback.print_exc()
                try:    # stdout va a DEVNULL: si no se loguea, el error es invisible
                    _log("loop_error", {"error": str(e)[:200], "exc_type": type(e).__name__})
                except Exception:
                    pass
            print(f"\nSleeping {CYCLE_MINUTES} min...")
            time.sleep(CYCLE_MINUTES * 60)
    else:
        run_cycle(execute=execute)

    mt5_shutdown()


if __name__ == "__main__":
    main()
