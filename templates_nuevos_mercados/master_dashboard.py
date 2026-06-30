"""
Master Portfolio Dashboard — All 12 Instruments on port 8090
Aggregates signals, performance and simulated P&L for the full robot portfolio.
Auto-refreshes every 30 s.

Usage:
  python master_dashboard.py          # starts on port 8090
  python master_dashboard.py --port 9000
"""

import os, sys, io, json, argparse, importlib, threading, socketserver
from datetime import datetime, date, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _MVP_ROOT not in sys.path:
    sys.path.insert(0, _MVP_ROOT)

PORT = 8090

# CT = UTC-5  (no DST correction; adjust to -6 if CDT/CST matters)
CT_OFFSET = -5

# Real artifacts root: MVP/artifacts/[sym_lower]/
_MVP_ROOT_DIR = os.path.dirname(_HERE)   # C:\...\MVP

INSTRUMENTS = [
    {"sym": "XAUUSD",   "name": "Gold",      "color": "#FFD700", "emoji": "🥇", "ind_port": 8080},
    {"sym": "BTCUSD",   "name": "Bitcoin",   "color": "#f7931a", "emoji": "₿",  "ind_port": 8081},
    {"sym": "US500",    "name": "S&P 500",   "color": "#00c853", "emoji": "📈", "ind_port": 8083},
    {"sym": "USTEC",    "name": "Nasdaq",    "color": "#00b0ff", "emoji": "💻", "ind_port": 8084},
    {"sym": "US30",     "name": "Dow Jones", "color": "#ff6f00", "emoji": "🏭", "ind_port": 8085},
    {"sym": "WTI_N6",   "name": "WTI Oil",  "color": "#795548", "emoji": "🛢", "ind_port": 8086},
    {"sym": "BRENT_N6", "name": "Brent Oil","color": "#546e7a", "emoji": "🛢", "ind_port": 8087, "mt5_sym": "BRENT_Q6", "display_sym": "BRENT_Q6"},
    {"sym": "UK100",    "name": "FTSE 100", "color": "#e91e63", "emoji": "🇬🇧","ind_port": 8088},
    {"sym": "Corn_N6",  "name": "Corn",     "color": "#cddc39", "emoji": "🌽", "ind_port": 8089},
    {"sym": "CHINA50",  "name": "China A50","color": "#ec407a", "emoji": "🇨🇳","ind_port": 8091},
    {"sym": "STOXX50",  "name": "EuroStoxx","color": "#3f51b5", "emoji": "🇪🇺","ind_port": 8082},
]


# ── time helpers ──────────────────────────────────────────────────────────────

def _now_ct():
    return datetime.now(timezone.utc) + timedelta(hours=CT_OFFSET)


# ── MT5 market-open cache ─────────────────────────────────────────────────────
# Refresh every 60 s so the HTTP response stays fast.

import time as _time
_MT5_STATUS_CACHE: dict = {}      # {sym: bool}
_MT5_STATUS_TS:    float = 0.0
_MT5_STATUS_TTL:   float = 60.0


_MT5_REFRESH_LOCK = threading.Lock()
_MT5_REFRESHING   = False


def _refresh_mt5_status():
    """
    Ask MT5 for the most-recent tick of each symbol (runs in background thread).
    If tick is within 5 min → open. Falls back silently if MT5 not available.
    """
    global _MT5_STATUS_CACHE, _MT5_STATUS_TS, _MT5_REFRESHING
    with _MT5_REFRESH_LOCK:
        if _MT5_REFRESHING:
            return
        _MT5_REFRESHING = True
    try:
        import MetaTrader5 as mt5
        if not mt5.terminal_info():
            if not mt5.initialize():
                return
        result = {}
        now_ts = _time.time()
        for inst in INSTRUMENTS:
            sym     = inst["sym"]
            mt5_sym = inst.get("mt5_sym", sym)
            try:
                tick = mt5.symbol_info_tick(mt5_sym)
                if tick and tick.time > 0:
                    result[sym] = (now_ts - tick.time) < 300
                else:
                    result[sym] = False
            except Exception:
                result[sym] = False
        _MT5_STATUS_CACHE = result
        _MT5_STATUS_TS    = now_ts
    except Exception:
        pass
    finally:
        _MT5_REFRESHING = False


def _mt5_open(sym):
    """
    Return cached MT5 open status. Triggers a background refresh if stale.
    Returns None when MT5 has never responded (use time-based fallback).
    """
    if _time.time() - _MT5_STATUS_TS > _MT5_STATUS_TTL:
        t = threading.Thread(target=_refresh_mt5_status, daemon=True)
        t.start()
    if sym in _MT5_STATUS_CACHE:
        return _MT5_STATUS_CACHE[sym]
    return None   # not yet populated → fallback


# ── market-open fallback (ICMarkets CFD hours, CT = UTC-5) ──────────────────
#
# Used only when MT5 is not connected.
# ICMarkets opens essentially every CFD on Sunday evening (when US futures open).
# XAUUSD  : Sun 16:00 CT  / Fri close ~22:00 CT
# Indices  : Sun 17:00 CT  / Fri close ~22:00 CT
# WTI/BRENT: Sun 17:00 CT  / Fri close ~22:00 CT
# UK100   : Sun 19:00 CT  / Fri close ~22:00 CT
# BTC/ETH : 24/7

_SUN_OPEN_CT = {           # minute-of-day (Sun) when ICMarkets opens
    "XAUUSD":   16 * 60,
    "BTCUSD":   0,
    "ETHUSD":   0,
    "US500":    17 * 60,
    "USTEC":    17 * 60,
    "US30":     17 * 60,
    "WTI_N6":   17 * 60,
    "BRENT_N6": 17 * 60,
    "UK100":    19 * 60,
}

def _is_market_open(sym):
    """
    Returns (is_open: bool, status_label: str).
    Primary source: MT5 symbol_info_tick (live).
    Fallback: ICMarkets CFD schedule expressed in CT.
    """
    # ── MT5 live check ────────────────────────────────────────────────────────
    mt5_status = _mt5_open(sym)
    if mt5_status is not None:
        label = "Abierto (MT5)" if mt5_status else "Cerrado (MT5)"
        return mt5_status, label

    # ── time-based fallback ───────────────────────────────────────────────────
    ct  = _now_ct()
    wd  = ct.weekday()          # 0=Mon … 4=Fri, 5=Sat, 6=Sun
    hm  = ct.hour * 60 + ct.minute

    if sym in ("BTCUSD", "ETHUSD"):
        return True, "24/7"

    if wd == 5:                             # Saturday: all closed
        return False, "Fin de semana"

    FRI_CLOSE = 22 * 60                     # ~22:00 CT Friday close

    if wd == 6:                             # Sunday
        opens_at = _SUN_OPEN_CT.get(sym, 17 * 60)
        if hm >= opens_at:
            return True, "Abierto"
        h, m = divmod(opens_at, 60)
        return False, f"Abre dom {h:02d}:{m:02d} CT"
    # ── Monday-Friday ─────────────────────────────────────────────────────────
    # All ICMarkets CFDs close briefly ~22:00 CT Friday for weekly maintenance.
    # Brief daily maintenance break ~21:00-22:00 CT is handled by MT5 live check above.
    if wd == 4 and hm >= FRI_CLOSE:        # Friday after 22:00 CT
        return False, "Cierra ~22:00 CT"
    # Everything else: Mon-Fri is open (ICMarkets CFD extended hours)
    return True, "Abierto"

    return True, "Abierto"


# ── prime-session logic (robot's preferred trading window) ───────────────────

def _load_inst_cfg(sym):
    """Import config.py from the instrument folder; returns module or None."""
    folder = os.path.join(_HERE, sym)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    try:
        import config as _cfg
        importlib.reload(_cfg)
        return _cfg
    except Exception:
        return None
    finally:
        if folder in sys.path:
            sys.path.remove(folder)


def _is_prime_now(sym):
    """
    Returns (in_prime: bool, label: str).
    Reads prime open/close from the instrument's config.py.
    """
    ct  = _now_ct()
    wd  = ct.weekday()
    hm  = ct.hour * 60 + ct.minute
    cfg = _load_inst_cfg(sym)

    if cfg is None:
        return False, "—"

    trade_wknd = getattr(cfg, "TRADE_WEEKENDS", False)
    if not trade_wknd and wd >= 5:
        return False, "Fin de semana"

    # XAUUSD uses RTH_OPEN_CT / RTH_CLOSE_CT
    open_ct  = getattr(cfg, "PRIME_OPEN_CT",  None) or getattr(cfg, "RTH_OPEN_CT",  None)
    close_ct = getattr(cfg, "PRIME_CLOSE_CT", None) or getattr(cfg, "RTH_CLOSE_CT", None)
    if open_ct is None or close_ct is None:
        return False, "—"

    o_hm = open_ct.hour  * 60 + open_ct.minute
    c_hm = close_ct.hour * 60 + close_ct.minute
    if o_hm <= hm <= c_hm:
        return True, f"{open_ct.strftime('%H:%M')}-{close_ct.strftime('%H:%M')} CT"
    return False, f"{open_ct.strftime('%H:%M')}-{close_ct.strftime('%H:%M')} CT"


# ── data helpers ─────────────────────────────────────────────────────────────

def _artifacts(sym):
    """Real artifacts path: MVP/artifacts/[sym_lower]/ — with override support."""
    inst = next((i for i in INSTRUMENTS if i["sym"] == sym), {})
    folder = inst.get("artifacts_override", sym.lower())
    return os.path.join(_MVP_ROOT_DIR, "artifacts", folder)

def _read_jsonl(path, n=500):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows[-n:]

def _read_signal(sym):
    art  = _artifacts(sym)
    inst = next((i for i in INSTRUMENTS if i["sym"] == sym), {})
    # Use per-instrument override or standard fallback names
    candidates = []
    if "signal_file_override" in inst:
        candidates.append(inst["signal_file_override"])
    candidates += ["live_signal.json", "signal_latest.json", "latest_signal.json"]
    for name in candidates:
        p = os.path.join(art, name)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    raw = json.load(f)
                # Sbean wraps signal inside {"signal": {...}}
                if isinstance(raw.get("signal"), dict):
                    inner = raw["signal"]
                    # Normalise field names to match portfolio convention
                    return {
                        "signal":     inner.get("signal", "FLAT"),
                        "entry":      inner.get("entry"),
                        "sl":         inner.get("stop_loss"),
                        "tp":         inner.get("take_profit"),
                        "confidence": inner.get("confidence","—"),
                        "rr":         inner.get("rr","—"),
                        "timestamp":  raw.get("timestamp",""),
                        "price":      raw.get("current_price"),
                    }
                return raw
            except Exception:
                pass
    return {}

def _read_paper(sym):
    inst = next((i for i in INSTRUMENTS if i["sym"] == sym), {})
    fname = inst.get("paper_file_override", "paper_trades.jsonl")
    p = os.path.join(_artifacts(sym), fname)
    return _read_jsonl(p, 500)

def _read_live_log(sym, n=30):
    p = os.path.join(_artifacts(sym), "live_log.jsonl")
    return _read_jsonl(p, n)


# ── Live execution instruments ────────────────────────────────────────────────
# EXECUTE_TRADES=True: generate real MT5 orders → use real deal history.
LIVE_EXEC_SYMS = {"ETHUSD", "BRENT_N6", "UK100"}


# ── Shared timestamp parser ───────────────────────────────────────────────────
def _parse_dt(s):
    """Parse ISO timestamp string → UTC-aware datetime, or None."""
    try:
        dt = datetime.fromisoformat((s or "").replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


# ── OHLC SL/TP check (paper & live fallback) ─────────────────────────────────
_OHLC_RESULT_CACHE: dict = {}   # {(sym, sig_ts, nxt_ts): (status, pnl, wall_ts)}
_OHLC_TTL_DONE = 86400          # 24 h — WIN/LOSS never change once confirmed
_OHLC_TTL_OPEN = 30             # 30 s — inconclusive windows may update


def _check_sl_tp_via_ohlc(sym, sig_time, nxt_time, signal, sl, tp, risk, rr):
    """
    Download M5 bars for [sig_time, nxt_time] and scan bar-by-bar for SL/TP hit.
    Returns (status, pnl):
      'WIN'  / +pnl  → TP was hit first
      'LOSS' / -pnl  → SL was hit first
      None   / 0.0   → neither hit in window (use fallback)

    For LONG: checks HIGH >= TP before LOW <= SL within each bar.
    For SHORT: checks LOW  <= TP before HIGH >= SL within each bar.
    Within the same M5 candle the order is ambiguous — we optimistically assume
    the favourable level is hit first (same convention as most backtesting engines).
    Results cached: 24 h for conclusive, 30 s for inconclusive.
    """
    cache_key = (sym, sig_time.isoformat(), nxt_time.isoformat())
    now = _time.time()
    cached = _OHLC_RESULT_CACHE.get(cache_key)
    if cached:
        status, pnl, wall_ts = cached
        ttl = _OHLC_TTL_DONE if status in ("WIN", "LOSS") else _OHLC_TTL_OPEN
        if now - wall_ts < ttl:
            return status, pnl

    try:
        import MetaTrader5 as mt5
        if not mt5.terminal_info():
            mt5.initialize()
        bars = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M5, sig_time, nxt_time)
        if bars is None or len(bars) == 0:
            _OHLC_RESULT_CACHE[cache_key] = (None, 0.0, now)
            return None, 0.0

        sl = float(sl); tp = float(tp)
        for bar in bars:
            high = bar["high"]; low = bar["low"]
            if signal == "LONG":
                if high >= tp:
                    r = ("WIN",  round(risk * rr, 2))
                    _OHLC_RESULT_CACHE[cache_key] = (*r, now); return r
                if low <= sl:
                    r = ("LOSS", round(-risk, 2))
                    _OHLC_RESULT_CACHE[cache_key] = (*r, now); return r
            else:  # SHORT
                if low <= tp:
                    r = ("WIN",  round(risk * rr, 2))
                    _OHLC_RESULT_CACHE[cache_key] = (*r, now); return r
                if high >= sl:
                    r = ("LOSS", round(-risk, 2))
                    _OHLC_RESULT_CACHE[cache_key] = (*r, now); return r

        # Neither level hit in this window
        _OHLC_RESULT_CACHE[cache_key] = (None, 0.0, now)
        return None, 0.0
    except Exception:
        return None, 0.0

# ── Live price cache ──────────────────────────────────────────────────────────
_LIVE_PRICE_CACHE: dict = {}   # {sym: (price, ts)}
_LIVE_PRICE_TTL = 30.0


def _get_live_price(sym):
    """Return cached MT5 mid-price (bid+ask)/2 for sym, or None."""
    now = _time.time()
    cached = _LIVE_PRICE_CACHE.get(sym)
    if cached and now - cached[1] < _LIVE_PRICE_TTL:
        return cached[0]
    try:
        import MetaTrader5 as mt5
        if not mt5.terminal_info():
            mt5.initialize()
        # Resolve mt5_sym for instruments with contract rollover (e.g. BRENT_Q6)
        mt5_sym = next((i.get("mt5_sym", sym) for i in INSTRUMENTS if i["sym"] == sym), sym)
        tick = mt5.symbol_info_tick(mt5_sym)
        if tick and tick.bid > 0:
            price = (tick.bid + tick.ask) / 2
            _LIVE_PRICE_CACHE[sym] = (price, now)
            return price
    except Exception:
        pass
    return None


# ── Real deals cache (LIVE_EXEC_SYMS only) ───────────────────────────────────
_REAL_DEALS_CACHE: dict = {}   # {sym: (deals_list, ts)}
_REAL_DEALS_TTL = 60.0


def _get_real_deals(sym):
    """Return list of real closing deals for a live-executed instrument."""
    now = _time.time()
    cached = _REAL_DEALS_CACHE.get(sym)
    if cached and now - cached[1] < _REAL_DEALS_TTL:
        return cached[0]
    try:
        import MetaTrader5 as mt5
        if not mt5.terminal_info():
            mt5.initialize()
        from_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        to_dt   = datetime.now(timezone.utc)
        raw = mt5.history_deals_get(from_dt, to_dt, group=f"*{sym}*")
        if raw is None:
            raw = []
        deals = []
        for d in raw:
            if d.entry == 1:   # 1 = OUT (position close)
                deals.append({
                    "time":   datetime.fromtimestamp(d.time, tz=timezone.utc),
                    "profit": round(d.profit + d.commission + d.swap, 2),
                })
        _REAL_DEALS_CACHE[sym] = (deals, now)
        return deals
    except Exception:
        return []


# ── Open position P&L cache ───────────────────────────────────────────────────
_OPEN_PNL_CACHE: dict = {}   # {sym: (pnl, ts)}
_OPEN_PNL_TTL = 15.0          # refresh cada 15 s (posiciones abiertas cambian rápido)


def _get_open_position_pnl(sym):
    """
    Return the REAL floating P&L of any currently open MT5 position for sym.
    Uses positions_get() — captures unrealized P&L + swap.
    Returns None if no open position or MT5 unavailable.
    """
    now = _time.time()
    cached = _OPEN_PNL_CACHE.get(sym)
    if cached and now - cached[1] < _OPEN_PNL_TTL:
        return cached[0]
    try:
        import MetaTrader5 as mt5
        if not mt5.terminal_info():
            mt5.initialize()
        positions = mt5.positions_get(symbol=sym)
        if positions:
            pnl = round(sum(p.profit + p.swap for p in positions), 2)
            _OPEN_PNL_CACHE[sym] = (pnl, now)
            return pnl
        # No open position
        _OPEN_PNL_CACHE[sym] = (None, now)
        return None
    except Exception:
        return None


def _pnl_vs_live(t, live_price):
    """
    Compare an open trade's SL/TP against current live price.
    Returns (status, pnl).  status: 'WIN' | 'LOSS' | 'OPEN'
    """
    sig   = t.get("signal")
    entry = t.get("entry")
    sl    = t.get("sl")
    tp    = t.get("tp")
    risk  = float(t.get("risk_usd") or 20)
    rr    = float(t.get("rr")       or 1.5)

    if not all([entry, sl, tp, live_price]):
        return "OPEN", 0.0

    entry = float(entry); sl = float(sl); tp = float(tp)

    if sig == "LONG":
        sl_d = abs(entry - sl) or 1e-9
        tp_d = abs(tp - entry) or 1e-9
        if live_price >= tp:
            return "WIN",  round(risk * rr, 2)
        elif live_price <= sl:
            return "LOSS", round(-risk, 2)
        else:
            move = live_price - entry
            pnl  = (move / tp_d) * risk * rr if move >= 0 else (move / sl_d) * risk
            return "OPEN", round(pnl, 2)
    else:  # SHORT
        sl_d = abs(sl - entry) or 1e-9
        tp_d = abs(entry - tp) or 1e-9
        if live_price <= tp:
            return "WIN",  round(risk * rr, 2)
        elif live_price >= sl:
            return "LOSS", round(-risk, 2)
        else:
            move = entry - live_price
            pnl  = (move / tp_d) * risk * rr if move >= 0 else (move / sl_d) * risk
            return "OPEN", round(pnl, 2)


# ── P&L simulation ───────────────────────────────────────────────────────────

def _simulate_pnl(trades, sym=None, live_price=None):
    """
    Compute P&L for paper-traded instruments.

    For each historical trade (not the last):
      1. If sym is provided → download M5 OHLC bars and scan bar-by-bar for SL/TP hit.
      2. Fallback → use next signal's price (proportional, as before).

    For the last (open) trade:
      If live_price is available → compare vs SL/TP for provisional status + floating P&L.
      Otherwise → OPEN / $0.
    """
    active = [t for t in trades if t.get("signal") in ("LONG", "SHORT")]
    if not active:
        return 0.0, 0, 0, []

    detail = []
    total  = 0.0
    wins   = 0
    losses = 0

    for i, t in enumerate(active):
        sig   = t["signal"]
        entry = t.get("entry")
        sl    = t.get("sl")
        tp    = t.get("tp")
        risk  = float(t.get("risk_usd") or 20)
        rr    = float(t.get("rr")       or 1.5)

        if not all([entry, sl, tp]):
            continue
        entry = float(entry); sl = float(sl); tp = float(tp)

        if i == len(active) - 1:
            if live_price is not None:
                status, pnl = _pnl_vs_live(t, live_price)
                pnl = round(pnl, 2)
                total += pnl
                if status == "WIN":    wins   += 1
                elif status == "LOSS": losses += 1
                detail.append({**t, "status": status, "pnl": pnl})
            else:
                detail.append({**t, "status": "OPEN", "pnl": 0.0})
            continue

        nxt = active[i + 1]

        # ── Step 1: OHLC bar-by-bar check (accurate) ──────────────────────────
        used_ohlc = False
        if sym is not None:
            sig_time = _parse_dt(t.get("timestamp", ""))
            nxt_time = _parse_dt(nxt.get("timestamp", ""))
            if sig_time and nxt_time:
                ohlc_status, ohlc_pnl = _check_sl_tp_via_ohlc(
                    sym, sig_time, nxt_time, sig, sl, tp, risk, rr
                )
                if ohlc_status in ("WIN", "LOSS"):
                    total += ohlc_pnl
                    if ohlc_status == "WIN":  wins   += 1
                    else:                     losses += 1
                    detail.append({**t, "status": ohlc_status, "pnl": ohlc_pnl})
                    used_ohlc = True

        if used_ohlc:
            continue

        # ── Step 2: Fallback — next-signal price (proportional) ───────────────
        nxt_price = nxt.get("price_at_signal") or nxt.get("entry") or nxt.get("price")
        if nxt_price is None:
            detail.append({**t, "status": "?", "pnl": 0.0})
            continue
        nxt_price = float(nxt_price)

        if sig == "LONG":
            sl_d = abs(entry - sl); tp_d = abs(tp - entry)
            if sl_d == 0:
                detail.append({**t, "status": "?", "pnl": 0.0}); continue
            if nxt_price >= tp:
                pnl = risk * rr; status = "WIN"; wins += 1
            elif nxt_price <= sl:
                pnl = -risk;     status = "LOSS"; losses += 1
            else:
                move = nxt_price - entry
                pnl  = (move / tp_d) * risk * rr if move >= 0 else (move / sl_d) * risk
                status = "WIN" if pnl >= 0 else "LOSS"
                if pnl >= 0: wins += 1
                else: losses += 1
        else:  # SHORT
            sl_d = abs(sl - entry); tp_d = abs(entry - tp)
            if sl_d == 0:
                detail.append({**t, "status": "?", "pnl": 0.0}); continue
            if nxt_price <= tp:
                pnl = risk * rr; status = "WIN"; wins += 1
            elif nxt_price >= sl:
                pnl = -risk;     status = "LOSS"; losses += 1
            else:
                move = entry - nxt_price
                pnl  = (move / tp_d) * risk * rr if move >= 0 else (move / sl_d) * risk
                status = "WIN" if pnl >= 0 else "LOSS"
                if pnl >= 0: wins += 1
                else: losses += 1

        pnl = round(pnl, 2)
        total += pnl
        detail.append({**t, "status": status, "pnl": pnl})

    return round(total, 2), wins, losses, detail


def _real_pnl_from_deals(trades, sym):
    """
    For LIVE_EXEC_SYMS: compute P&L using real MT5 deal history.
    Matches closing deals to each signal by timestamp window.
    Falls back to next-signal simulation when no deal is found.
    Uses live price for the last (still open) signal.
    """
    active = [t for t in trades if t.get("signal") in ("LONG", "SHORT")]
    if not active:
        return 0.0, 0, 0, []

    deals      = _get_real_deals(sym)
    live_price = _get_live_price(sym)
    detail = []
    total  = 0.0
    wins   = 0
    losses = 0

    for i, t in enumerate(active):
        sig_time = _parse_dt(t.get("timestamp", ""))
        if sig_time is None:
            detail.append({**t, "status": "?", "pnl": 0.0})
            continue

        is_last  = (i == len(active) - 1)
        nxt_time = _parse_dt(active[i + 1].get("timestamp", "")) if not is_last else None
        window_end = min(
            (nxt_time or datetime.now(timezone.utc)) + timedelta(hours=48),
            datetime.now(timezone.utc)
        )

        matching = [d for d in deals if sig_time <= d["time"] <= window_end]

        if matching:
            real_profit = sum(d["profit"] for d in matching)
            pnl    = round(real_profit, 2)
            status = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BE")
            if pnl > 0:   wins   += 1
            elif pnl < 0: losses += 1
            total += pnl
            detail.append({**t, "status": status, "pnl": pnl})

        elif is_last:
            # Last signal: check for real open MT5 position first
            open_pnl = _get_open_position_pnl(sym)
            if open_pnl is not None:
                # Real floating P&L from positions_get()
                pnl    = open_pnl
                status = "OPEN"
            else:
                # No open position (may have closed without a recorded deal):
                # fall back to live price vs SL/TP
                status, pnl = _pnl_vs_live(t, live_price)
                pnl = round(pnl, 2)
            total += pnl
            if status == "WIN":    wins   += 1
            elif status == "LOSS": losses += 1
            detail.append({**t, "status": status, "pnl": pnl})

        else:
            # No deal found for this signal
            # ── Step 1: OHLC bar-by-bar check ─────────────────────────────────
            sig_dir = t["signal"]
            ev = t.get("entry"); sv = t.get("sl"); tv = t.get("tp")
            rv = float(t.get("risk_usd") or 20); rrv = float(t.get("rr") or 1.5)

            used_ohlc = False
            if sig_time and nxt_time and all([ev, sv, tv]):
                ohlc_status, ohlc_pnl = _check_sl_tp_via_ohlc(
                    sym, sig_time, window_end, sig_dir, sv, tv, rv, rrv
                )
                if ohlc_status in ("WIN", "LOSS"):
                    total += ohlc_pnl
                    if ohlc_pnl > 0:  wins   += 1
                    else:             losses += 1
                    detail.append({**t, "status": ohlc_status, "pnl": ohlc_pnl})
                    used_ohlc = True

            if used_ohlc:
                continue

            # ── Step 2: Fallback — next-signal price (proportional) ───────────
            nxt = active[i + 1]
            nxt_price = nxt.get("price_at_signal") or nxt.get("entry") or nxt.get("price")
            if nxt_price is None or not all([ev, sv, tv]):
                detail.append({**t, "status": "?", "pnl": 0.0})
                continue
            nxt_price = float(nxt_price)
            ev = float(ev); sv = float(sv); tv = float(tv)
            if sig_dir == "LONG":
                sl_d = abs(ev - sv) or 1e-9; tp_d = abs(tv - ev) or 1e-9
                if nxt_price >= tv:   pnl = rv * rrv; status = "WIN";  wins   += 1
                elif nxt_price <= sv: pnl = -rv;       status = "LOSS"; losses += 1
                else:
                    move = nxt_price - ev
                    pnl  = (move / tp_d) * rv * rrv if move >= 0 else (move / sl_d) * rv
                    status = "WIN" if pnl >= 0 else "LOSS"
                    if pnl >= 0: wins += 1
                    else: losses += 1
            else:
                sl_d = abs(sv - ev) or 1e-9; tp_d = abs(ev - tv) or 1e-9
                if nxt_price <= tv:   pnl = rv * rrv; status = "WIN";  wins   += 1
                elif nxt_price >= sv: pnl = -rv;       status = "LOSS"; losses += 1
                else:
                    move = ev - nxt_price
                    pnl  = (move / tp_d) * rv * rrv if move >= 0 else (move / sl_d) * rv
                    status = "WIN" if pnl >= 0 else "LOSS"
                    if pnl >= 0: wins += 1
                    else: losses += 1
            pnl = round(pnl, 2)
            total += pnl
            detail.append({**t, "status": status, "pnl": pnl})

    return round(total, 2), wins, losses, detail


# ── MT5 account ───────────────────────────────────────────────────────────────

_MT5_ACCOUNT_CACHE: dict = {}
_MT5_ACCOUNT_TS:   float = 0.0
_MT5_ACCOUNT_TTL:  float = 60.0
_MT5_ACCOUNT_LOCK  = threading.Lock()
_MT5_ACCOUNT_BUSY  = False


def _refresh_mt5_account():
    global _MT5_ACCOUNT_CACHE, _MT5_ACCOUNT_TS, _MT5_ACCOUNT_BUSY
    with _MT5_ACCOUNT_LOCK:
        if _MT5_ACCOUNT_BUSY:
            return
        _MT5_ACCOUNT_BUSY = True
    try:
        import MetaTrader5 as mt5
        if not mt5.terminal_info():
            if not mt5.initialize():
                return
        info = mt5.account_info()
        if info:
            _MT5_ACCOUNT_CACHE = {
                "balance": info.balance,
                "equity":  info.equity,
                "profit":  info.profit,
                "currency":info.currency,
            }
            # ── P&L REAL desde historial de deals MT5 (fuente de verdad) ──────
            # Deals cerrados son INMUTABLES → el total no fluctúa con el mercado.
            # (El P&L simulado desde señales re-evaluaba la última señal contra el
            #  precio vivo aunque no hubiera posición real → total no confiable.)
            try:
                from datetime import datetime as _dt, timedelta as _td
                _now = _dt.now()
                _deals = mt5.history_deals_get(_dt(2025, 1, 1), _now + _td(hours=12)) or []
                _tot = _today = _7d = 0.0
                _bysym = {}
                _today_d = _now.date(); _7d_cut = _now - _td(days=7)
                for _d in _deals:
                    if _d.type not in (0, 1):  # solo BUY/SELL; type 2 = balance ops (depósitos)
                        continue
                    _p = _d.profit + _d.swap + _d.commission
                    if _p == 0:
                        continue
                    _tot += _p
                    _bysym[_d.symbol] = _bysym.get(_d.symbol, 0.0) + _p
                    _dt_deal = _dt.fromtimestamp(_d.time)
                    if _dt_deal.date() == _today_d: _today += _p
                    if _dt_deal >= _7d_cut:         _7d    += _p
                _pos = mt5.positions_get() or []
                _float = sum(p.profit + p.swap for p in _pos)
                _MT5_ACCOUNT_CACHE.update({
                    "realized_total": round(_tot, 2),
                    "realized_today": round(_today, 2),
                    "realized_7d":    round(_7d, 2),
                    "realized_bysym": {k: round(v, 2) for k, v in _bysym.items()},
                    "floating":       round(_float, 2),
                    "open_count":     len(_pos),
                })
            except Exception:
                pass
            _MT5_ACCOUNT_TS = _time.time()
    except Exception:
        pass
    finally:
        _MT5_ACCOUNT_BUSY = False


def _mt5_account():
    """Return cached MT5 account info; refresh in background if stale."""
    if _time.time() - _MT5_ACCOUNT_TS > _MT5_ACCOUNT_TTL:
        t = threading.Thread(target=_refresh_mt5_account, daemon=True)
        t.start()
    return _MT5_ACCOUNT_CACHE


# ── aggregate ─────────────────────────────────────────────────────────────────

def _load_all():
    rows = []
    for inst in INSTRUMENTS:
        sym    = inst["sym"]
        paper  = _read_paper(sym)
        sig    = _read_signal(sym)

        trades  = [t for t in paper if t.get("signal") in ("LONG","SHORT")]
        flat_ct = sum(1 for t in paper if t.get("signal") == "FLAT")
        longs   = sum(1 for t in trades if t.get("signal") == "LONG")
        shorts  = sum(1 for t in trades if t.get("signal") == "SHORT")
        today_s = date.today().isoformat()
        today_ct= sum(1 for t in trades if (t.get("timestamp","") or "")[:10] == today_s)

        rr_vals = [float(t["rr"]) for t in trades if t.get("rr")]
        avg_rr  = round(sum(rr_vals)/len(rr_vals), 2) if rr_vals else None

        if sym in LIVE_EXEC_SYMS:
            pnl, wins, losses, detail = _real_pnl_from_deals(trades, sym)
            pnl_mode = "Real"
        else:
            live_price = _get_live_price(sym)
            pnl, wins, losses, detail = _simulate_pnl(trades, sym=sym, live_price=live_price)
            pnl_mode = "Sim"
        mkt_open, mkt_label  = _is_market_open(sym)
        prime_now, prime_hrs = _is_prime_now(sym)

        # Last live log entry (for robot health status)
        live_logs  = _read_live_log(sym, 5)
        last_log   = live_logs[-1] if live_logs else {}
        last_log_ts   = (last_log.get("timestamp","") or "")[:16].replace("T"," ")
        last_log_type = last_log.get("type","")

        rows.append({
            **inst,
            "sig":          sig,
            "trades":       trades,
            "flat_ct":      flat_ct,
            "longs":        longs,
            "shorts":       shorts,
            "today_ct":     today_ct,
            "avg_rr":       avg_rr,
            "pnl":          pnl,
            "wins":         wins,
            "losses":       losses,
            "detail":       detail,
            "pnl_mode":     pnl_mode,
            "mkt_open":     mkt_open,
            "mkt_label":    mkt_label,
            "prime_now":    prime_now,
            "prime_hrs":    prime_hrs,
            "last_log_ts":  last_log_ts,
            "last_log_type":last_log_type,
        })
    return rows


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _fmt(v, dec=2):
    if isinstance(v, (int, float)):
        return f"${v:,.{dec}f}"
    return str(v) if v else "—"

def _pnl_color(v):
    if isinstance(v, (int, float)):
        return "#00c853" if v >= 0 else "#f44336"
    return "#888"

def _sig_color(s):
    return {"LONG": "#00c853", "SHORT": "#f44336"}.get(s, "#888")


# ── Prop Firm readiness card ────────────────────────────────────────────────

def _prop_card():
    """Lee data/prop_readiness.json (lo escribe prop_qualification.py, diario 21:30) y lo renderiza."""
    import json as _json
    p = os.path.join(_MVP_ROOT_DIR, "data", "prop_readiness.json")
    if not os.path.exists(p):
        return ('<div class="card" style="border-top:3px solid #888"><div style="padding:14px">'
                '<b>🎯 Prop Firm Readiness</b><br><span style="color:#888">Sin datos aún. '
                'Corre <code>prop_qualification.py</code> o esperá la corrida agendada (21:30).</span></div></div>')
    try:
        s = _json.load(open(p, encoding="utf-8"))
    except Exception as e:
        return f'<div class="card"><div style="padding:14px">🎯 Prop Readiness: error leyendo JSON ({e})</div></div>'

    def dot(ok):
        return '🟢' if ok is True else ('🔴' if ok is False else '⚪')
    ts = (s.get("ts", "") or "")[:16].replace("T", " ")
    ready = s.get("ready")
    head_c = "#00c853" if ready else "#ffa000"
    head_txt = "🎯 PROP-READY" if ready else ("🎯 Prop Firm Readiness — en progreso" if s.get("sample_ok") else "🎯 Prop Firm Readiness")

    if not s.get("sample_ok"):
        body = (f'<div style="color:#ffa000;margin-top:8px">⏳ {s.get("motivo","muestra insuficiente")} '
                f'— {s.get("trades",0)} trade(s). El edge y el DD necesitan acumular operaciones.</div>')
    else:
        crit = ""
        for c in s.get("criterios", []):
            crit += (f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #ffffff10">'
                     f'<span>{dot(c["ok"])} {c["nombre"]}</span>'
                     f'<span style="color:#aaa">{c["valor"]} <span style="color:#666">(lim {c["limite"]})</span></span></div>')
        rows = ""
        for r in s.get("proyeccion", []):
            sem = r["semanas"] if r["semanas"] is not None else "—"
            rows += (f'<tr><td>{r["riesgo"]}%</td><td style="text-align:right">{r["maxdd"]:+.2f}%</td>'
                     f'<td style="text-align:right">{r["peor_dia"]:+.2f}%</td>'
                     f'<td style="text-align:right">{r["ret_mes"]:+.2f}%</td>'
                     f'<td style="text-align:right">{sem}</td><td style="text-align:center">{dot(r["ok"])}</td></tr>')
        body = (f'<div style="margin-top:8px">{crit}</div>'
                f'<div style="color:#888;font-size:12px;margin:10px 0 4px">Proyección a riesgo prop '
                f'(riesgo actual ~{s.get("cur_risk_pct","?")}%, Sharpe {s.get("sharpe","—")}):</div>'
                f'<table style="width:100%;font-size:13px"><thead><tr style="color:#888">'
                f'<th style="text-align:left">riesgo/trade</th><th style="text-align:right">MaxDD</th>'
                f'<th style="text-align:right">peor día</th><th style="text-align:right">ret/mes</th>'
                f'<th style="text-align:right">sem→8%</th><th style="text-align:center">prop?</th></tr></thead>'
                f'<tbody>{rows}</tbody></table>')

    return (f'<div class="card" style="border-top:3px solid {head_c};grid-column:1/-1">'
            f'<div style="padding:14px">'
            f'<div style="display:flex;justify-content:space-between"><b style="color:{head_c}">{head_txt}</b>'
            f'<span style="color:#666;font-size:12px">actualizado {ts or "—"}</span></div>'
            f'{body}</div></div>')


_MAGIC_NAME = {20260617: "EVENTBREAK", 20260605: "USTEC", 20260608: "BRENT_N6", 20260604: "US500",
               20260612: "CHINA50", 20260614: "STOXX50", 20260615: "GAPENGINE"}

def _open_positions_html():
    """Tabla de TODAS las posiciones abiertas en MT5 (ground truth, por magic/robot).
    Asi se ven EVENTBREAK/GAPENGINE/OILFADE y cualquier trade que no tenga tarjeta de instrumento."""
    try:
        import MetaTrader5 as mt5
        if not mt5.terminal_info():
            mt5.initialize()
        pos = mt5.positions_get() or []
    except Exception:
        pos = []
    if not pos:
        return ('<div class="card" style="grid-column:1/-1"><div style="padding:12px;color:#888">'
                'Sin posiciones abiertas en MT5.</div></div>')
    rows = ""
    total = 0.0
    for p in sorted(pos, key=lambda x: x.magic):
        pl = p.profit + p.swap; total += pl
        col = "#00c853" if pl >= 0 else "#f44336"
        rob = _MAGIC_NAME.get(p.magic, f"magic {p.magic}")
        rows += (f'<tr><td>{rob}</td><td>{p.symbol}</td><td>{"BUY" if p.type==0 else "SELL"}</td>'
                 f'<td style="text-align:right">{p.volume}</td><td style="text-align:right">{p.price_open}</td>'
                 f'<td style="text-align:right;color:{col}">${pl:+.2f}</td></tr>')
    tcol = "#00c853" if total >= 0 else "#f44336"
    return (f'<div class="card" style="grid-column:1/-1;border-top:3px solid #00b0ff"><div style="padding:12px">'
            f'<div style="display:flex;justify-content:space-between"><b>📌 Posiciones abiertas (MT5)</b>'
            f'<span style="color:{tcol}">flotante total ${total:+.2f}</span></div>'
            f'<table style="width:100%;font-size:13px;margin-top:8px"><thead><tr style="color:#888">'
            f'<th style="text-align:left">Robot</th><th style="text-align:left">Símbolo</th>'
            f'<th style="text-align:left">Dir</th><th style="text-align:right">Lots</th>'
            f'<th style="text-align:right">Entrada</th><th style="text-align:right">P&L</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div></div>')


def _closed_by_magic_html():
    """Historial de deals CERRADOS agrupado por magic/robot (ground truth MT5). Asi EVENTBREAK
    y demas robots multi-simbolo tienen su P&L cerrado VISIBLE y separado (el desglose por
    instrumento de abajo atribuye por simbolo y los mezclaria)."""
    try:
        import MetaTrader5 as mt5
        from datetime import datetime as _dt, timezone as _tz
        if not mt5.terminal_info():
            mt5.initialize()
        deals = mt5.history_deals_get(_dt(2026, 6, 1, tzinfo=_tz.utc), _dt.now(_tz.utc) + timedelta(hours=12)) or []
    except Exception:
        deals = []
    agg = {}
    for d in deals:
        if d.entry != 1 or d.magic == 0:
            continue
        a = agg.setdefault(d.magic, {"n": 0, "pnl": 0.0, "w": 0})
        pl = d.profit + d.commission + d.swap
        a["n"] += 1; a["pnl"] += pl; a["w"] += (pl > 0)
    if not agg:
        return ('<div class="card" style="grid-column:1/-1"><div style="padding:12px;color:#888">'
                'Sin trades cerrados aún (forward test).</div></div>')
    rows = ""
    for mg, a in sorted(agg.items(), key=lambda x: x[1]["pnl"]):
        col = "#00c853" if a["pnl"] >= 0 else "#f44336"
        wr = a["w"] / a["n"] * 100 if a["n"] else 0
        rob = _MAGIC_NAME.get(mg, f"magic {mg}")
        rows += (f'<tr><td>{rob}</td><td style="text-align:right">{a["n"]}</td>'
                 f'<td style="text-align:right">{wr:.0f}%</td>'
                 f'<td style="text-align:right;color:{col}">${a["pnl"]:+.2f}</td></tr>')
    tot = sum(a["pnl"] for a in agg.values())
    tcol = "#00c853" if tot >= 0 else "#f44336"
    return (f'<div class="card" style="grid-column:1/-1;border-top:3px solid #9c27b0"><div style="padding:12px">'
            f'<div style="display:flex;justify-content:space-between"><b>📒 Historial cerrado por robot (magic)</b>'
            f'<span style="color:{tcol}">total ${tot:+.2f}</span></div>'
            f'<table style="width:100%;font-size:13px;margin-top:8px"><thead><tr style="color:#888">'
            f'<th style="text-align:left">Robot</th><th style="text-align:right">Trades</th>'
            f'<th style="text-align:right">WR</th><th style="text-align:right">P&L</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div></div>')


# ── HTML builder ──────────────────────────────────────────────────────────────

def _build_html():
    all_data = _load_all()
    acct     = _mt5_account()
    ct       = _now_ct()

    # Portfolio totals
    total_trades = sum(d["longs"] + d["shorts"] for d in all_data)
    total_longs  = sum(d["longs"]  for d in all_data)
    total_shorts = sum(d["shorts"] for d in all_data)
    total_flat   = sum(d["flat_ct"] for d in all_data)
    total_pnl    = round(sum(d["pnl"] for d in all_data), 2)
    total_wins   = sum(d["wins"]   for d in all_data)
    total_losses = sum(d["losses"] for d in all_data)
    win_rate     = round(total_wins / (total_wins + total_losses) * 100, 1) if (total_wins + total_losses) else 0
    markets_open = sum(1 for d in all_data if d["mkt_open"])

    balance = acct.get("balance", "—")
    equity  = acct.get("equity",  "—")
    bal_s   = f"${balance:,.2f}" if isinstance(balance, (int,float)) else "—"
    eq_s    = f"${equity:,.2f}"  if isinstance(equity,  (int,float)) else "—"
    live_diff = round(equity - balance, 2) if isinstance(equity,(int,float)) and isinstance(balance,(int,float)) else None
    live_diff_s = (f"+${live_diff:,.2f}" if live_diff >= 0 else f"-${abs(live_diff):,.2f}") if live_diff is not None else "—"

    pnl_color = _pnl_color(total_pnl)
    pnl_s     = (f"+${total_pnl:,.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.2f}")

    # ── P&L REAL MT5 (deals cerrados — inmutable, no fluctúa con el mercado) ──
    real_total = acct.get("realized_total")
    real_today = acct.get("realized_today")
    real_7d    = acct.get("realized_7d")
    def _fmt_pnl(v):
        if not isinstance(v, (int, float)): return "—"
        return f"+${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}"
    real_total_s = _fmt_pnl(real_total)
    real_today_s = _fmt_pnl(real_today)
    real_7d_s    = _fmt_pnl(real_7d)

    # ── instrument cards ──────────────────────────────────────────────────────
    cards_html = ""
    for d in all_data:
        sig      = d["sig"]
        sl_dir   = sig.get("signal", "—")
        sl_c     = _sig_color(sl_dir)
        last_ts  = (sig.get("timestamp","") or "")[:16].replace("T"," ")
        price    = sig.get("price","—")
        price_s  = f"${price:,.2f}" if isinstance(price,(int,float)) else str(price) if price else "—"
        entry_s  = _fmt(sig.get("entry"))
        conf     = sig.get("confidence","—")
        rr_s     = str(sig.get("rr","—"))

        p_color  = _pnl_color(d["pnl"])
        n_trades = d["longs"] + d["shorts"]
        p_s      = (f"+${d['pnl']:,.2f}" if d["pnl"] >= 0 else f"-${abs(d['pnl']):,.2f}") if n_trades else "—"

        wr_pct   = round(d["wins"]/(d["wins"]+d["losses"])*100, 0) if (d["wins"]+d["losses"]) > 0 else 0

        # Market status badge
        mkt_bg   = "#00c85320" if d["mkt_open"] else "#f4433620"
        mkt_bc   = "#00c85366" if d["mkt_open"] else "#f4433666"
        mkt_tc   = "#00c853"   if d["mkt_open"] else "#f44336"
        mkt_dot  = "●" if d["mkt_open"] else "●"
        mkt_word = "Mercado abierto" if d["mkt_open"] else "Mercado cerrado"

        # Prime session badge
        pr_bg    = "#ffa00020" if d["prime_now"] else "#ffffff10"
        pr_bc    = "#ffa00066" if d["prime_now"] else "#ffffff20"
        pr_tc    = "#ffa000"   if d["prime_now"] else "#666"
        pr_word  = "Prime activo" if d["prime_now"] else "Fuera de prime"

        # Win-rate progress bar
        wr_bar   = f'''<div style="background:#1c2128;border-radius:4px;height:5px;margin-top:6px">
            <div style="background:{p_color};height:5px;border-radius:4px;width:{min(wr_pct,100):.0f}%"></div>
          </div>'''

        cards_html += f"""
        <div class="card" style="border-top:3px solid {d['color']}">

          <!-- Header: emoji + name + link -->
          <div class="card-header">
            <span class="card-emoji">{d['emoji']}</span>
            <div class="card-title">
              <div class="card-sym">{d.get('display_sym', d['sym'])}</div>
              <div class="card-name">{d['name']}</div>
            </div>
            <a class="dash-link" href="http://localhost:{d['ind_port']}" target="_blank" title="Abrir dashboard individual">&#8599;</a>
          </div>

          <!-- Status badges -->
          <div class="badge-row">
            <span class="badge" style="background:{mkt_bg};color:{mkt_tc};border:1px solid {mkt_bc}">
              <span style="font-size:9px">{mkt_dot}</span> {mkt_word}
            </span>
            <span class="badge" style="background:{pr_bg};color:{pr_tc};border:1px solid {pr_bc}">
              ⏱ {pr_word}
            </span>
          </div>
          <div class="prime-hrs">{d['prime_hrs']}</div>

          <!-- Current signal -->
          <div class="sig-section">
            <span class="sig-badge" style="background:{sl_c}22;color:{sl_c};border:1px solid {sl_c}44">{sl_dir}</span>
            <span class="sig-meta">@ {entry_s} &nbsp;|&nbsp; Conf: {conf} &nbsp;|&nbsp; R:R {rr_s}</span>
          </div>
          <div class="price-row">Precio: {price_s} &nbsp;·&nbsp; {last_ts or '—'}</div>

          <!-- Stats -->
          <div class="stats-row">
            <div class="stat-box">
              <div class="stat-val">{n_trades}</div>
              <div class="stat-lbl">Señales</div>
            </div>
            <div class="stat-box">
              <div class="stat-val" style="color:#00c853">{d['wins']}</div>
              <div class="stat-lbl">Wins</div>
            </div>
            <div class="stat-box">
              <div class="stat-val" style="color:#f44336">{d['losses']}</div>
              <div class="stat-lbl">Losses</div>
            </div>
            <div class="stat-box">
              <div class="stat-val" style="color:{p_color}">{p_s}</div>
              <div class="stat-lbl">{d['pnl_mode']} P&L</div>
            </div>
          </div>

          <!-- Win-rate bar -->
          <div class="wr-label">{wr_pct:.0f}% win rate &nbsp;·&nbsp; Hoy: {d['today_ct']} señales</div>
          {wr_bar}
          <div class="ts">Último ciclo: {d['last_log_ts'] or '—'} [{d['last_log_type'] or '—'}]</div>
        </div>"""

    # ── combined signals table ────────────────────────────────────────────────
    all_trades = []
    for d in all_data:
        for t in d["detail"]:
            all_trades.append({**t, "_sym": d.get("display_sym", d["sym"]), "_color": d["color"]})
    all_trades.sort(key=lambda x: x.get("timestamp",""), reverse=True)

    table_rows = ""
    for t in all_trades[:50]:
        ts  = (t.get("timestamp","") or "")[:16].replace("T"," ")
        sym = t["_sym"]; col = t["_color"]
        s   = t.get("signal","?"); sc = _sig_color(s)
        ent = _fmt(t.get("entry")); sl_ = _fmt(t.get("sl")); tp_ = _fmt(t.get("tp"))
        rr_ = t.get("rr","?"); cf_ = t.get("confidence","?")
        st  = t.get("status","?")
        st_c = {"WIN":"#00c853","LOSS":"#f44336","OPEN":"#ffa000","?":"#888"}.get(st,"#888")
        p_  = t.get("pnl", 0)
        p_s_= (f"+${p_:,.2f}" if p_ >= 0 else f"-${abs(p_):,.2f}") if isinstance(p_,(int,float)) and p_ != 0 else "—"
        p_c_= _pnl_color(p_)
        table_rows += f"""<tr>
          <td>{ts}</td>
          <td><span style="color:{col};font-weight:700">{sym}</span></td>
          <td style="color:{sc};font-weight:700">{s}</td>
          <td>{ent}</td><td>{sl_}</td><td>{tp_}</td>
          <td>{rr_}</td><td>{cf_}</td>
          <td><span style="color:{st_c};font-weight:600">{st}</span></td>
          <td style="color:{p_c_};font-weight:600">{p_s_}</td>
        </tr>"""

    # ── P&L breakdown table ───────────────────────────────────────────────────
    pnl_bars = ""
    max_abs  = max((abs(d["pnl"]) for d in all_data if d["pnl"] != 0), default=1) or 1
    for d in all_data:
        p     = d["pnl"]
        p_c   = _pnl_color(p)
        n_t   = d["longs"] + d["shorts"]
        p_s2  = (f"+${p:,.2f}" if p >= 0 else f"-${abs(p):,.2f}") if n_t else "—"
        bar_w = min(abs(p) / max_abs * 100, 100) if p != 0 else 0
        wr_p  = round(d["wins"]/(d["wins"]+d["losses"])*100,1) if (d["wins"]+d["losses"]) > 0 else 0
        mkt_dot_c = "#00c853" if d["mkt_open"] else "#f44336"
        pnl_bars += f"""<tr>
          <td>
            <span style="color:{mkt_dot_c};font-size:10px">●</span>
            <span style="color:{d['color']};font-weight:600"> {d['emoji']} {d.get('display_sym', d['sym'])}</span>
          </td>
          <td style="text-align:right">{n_t}</td>
          <td style="text-align:right;color:#00c853">{d['wins']}</td>
          <td style="text-align:right;color:#f44336">{d['losses']}</td>
          <td style="text-align:right">{wr_p}%</td>
          <td style="min-width:140px;padding:8px 14px">
            <div style="background:#21262d;border-radius:4px;height:10px">
              <div style="background:{p_c};height:10px;border-radius:4px;width:{bar_w:.1f}%"></div>
            </div>
          </td>
          <td style="color:{p_c};font-weight:700;text-align:right">{p_s2}</td>
          <td style="text-align:right">{d['avg_rr'] or '—'}</td>
          <td style="text-align:right">{d['prime_hrs']}</td>
        </tr>"""

    now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ct_s  = ct.strftime("%H:%M CT")
    prop_html = _prop_card()
    positions_html = _open_positions_html()
    closed_magic_html = _closed_by_magic_html()

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Master Dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0 }}
  body {{
    background: #0d1117;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    padding: 24px;
    font-size: 14px;
  }}
  a {{ color: inherit; text-decoration: none }}

  /* TOP BAR */
  .topbar {{
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 28px; flex-wrap: wrap; gap: 12px;
  }}
  .topbar h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 4px }}
  .topbar .sub {{ color: #666; font-size: 12px }}
  .refresh-badge {{
    background: #21262d; border: 1px solid #30363d; border-radius: 20px;
    padding: 5px 14px; font-size: 12px; color: #888;
  }}

  /* SUMMARY STRIP */
  .summary {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 32px }}
  .scard {{
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px 20px; flex: 1; min-width: 120px;
  }}
  .scard .sv {{ font-size: 28px; font-weight: 700; line-height: 1 }}
  .scard .sl {{ font-size: 11px; color: #888; margin-top: 6px;
                 text-transform: uppercase; letter-spacing: .5px }}

  /* SECTION TITLE */
  h2 {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; color: #666; margin: 0 0 16px;
  }}

  /* INSTRUMENT CARDS GRID */
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 18px;
    margin-bottom: 36px;
  }}
  .card {{
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 22px 22px 18px;
  }}

  /* Card header */
  .card-header {{ display: flex; align-items: center; gap: 14px; margin-bottom: 16px }}
  .card-emoji  {{ font-size: 28px }}
  .card-sym    {{ font-size: 17px; font-weight: 700 }}
  .card-name   {{ font-size: 12px; color: #888; margin-top: 2px }}
  .dash-link   {{
    margin-left: auto; background: #21262d; border: 1px solid #30363d;
    border-radius: 8px; padding: 5px 11px; font-size: 15px; color: #888;
    transition: color .15s; cursor: pointer;
  }}
  .dash-link:hover {{ color: #e6edf3; border-color: #888 }}

  /* Status badges */
  .badge-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 6px }}
  .badge {{
    font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 20px;
    letter-spacing: .2px;
  }}
  .prime-hrs {{ font-size: 11px; color: #555; margin-bottom: 14px }}

  /* Signal section */
  .sig-section {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; flex-wrap: wrap }}
  .sig-badge {{
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 13px; font-weight: 700; letter-spacing: .5px;
  }}
  .sig-meta {{ font-size: 12px; color: #888 }}
  .price-row {{ font-size: 12px; color: #555; margin-bottom: 16px }}

  /* Stats row */
  .stats-row  {{ display: flex; gap: 8px; margin-bottom: 10px }}
  .stat-box   {{ flex: 1; background: #0d1117; border-radius: 8px; padding: 10px 8px; text-align: center }}
  .stat-val   {{ font-size: 17px; font-weight: 700 }}
  .stat-lbl   {{ font-size: 10px; color: #888; margin-top: 3px; text-transform: uppercase }}

  .wr-label   {{ font-size: 11px; color: #555; margin-bottom: 4px }}

  /* TABLES */
  .tbl-wrap {{
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    overflow: hidden; margin-bottom: 32px;
  }}
  table   {{ width: 100%; border-collapse: collapse }}
  th      {{
    background: #21262d; color: #888; font-size: 11px; text-transform: uppercase;
    letter-spacing: .6px; padding: 11px 14px; text-align: left; white-space: nowrap;
  }}
  td      {{ padding: 10px 14px; border-bottom: 1px solid #21262d; font-size: 13px }}
  tr:last-child td {{ border-bottom: none }}
  tr:hover td {{ background: #1c2128 }}

  /* Disclaimer */
  .disc {{
    font-size: 11px; color: #555; border: 1px solid #21262d; border-radius: 8px;
    padding: 10px 16px; margin-bottom: 32px; line-height: 1.6;
  }}
</style>
</head>
<body>

<!-- TOP BAR -->
<div class="topbar">
  <div>
    <h1>📊 Master Portfolio Dashboard</h1>
    <div class="sub">Actualizado: {now_s} &nbsp;·&nbsp; {ct_s} &nbsp;·&nbsp; {markets_open}/12 mercados abiertos</div>
  </div>
  <span class="refresh-badge">&#8635; auto-refresh 30 s</span>
</div>

<!-- SUMMARY STRIP -->
<div class="summary">
  <div class="scard"><div class="sv">{markets_open}<span style="font-size:14px;color:#888">/12</span></div><div class="sl">Mercados abiertos</div></div>
  <div class="scard"><div class="sv">{total_trades}</div><div class="sl">Total señales</div></div>
  <div class="scard"><div class="sv" style="color:#00c853">{total_longs}</div><div class="sl">LONG</div></div>
  <div class="scard"><div class="sv" style="color:#f44336">{total_shorts}</div><div class="sl">SHORT</div></div>
  <div class="scard"><div class="sv">{total_flat}</div><div class="sl">FLAT</div></div>
  <div class="scard"><div class="sv" style="color:#00c853">{total_wins}</div><div class="sl">Wins</div></div>
  <div class="scard"><div class="sv" style="color:#f44336">{total_losses}</div><div class="sl">Losses</div></div>
  <div class="scard"><div class="sv">{win_rate}%</div><div class="sl">Win rate</div></div>
  <div class="scard"><div class="sv" style="color:{_pnl_color(real_total)}">{real_total_s}</div><div class="sl">P&amp;L Total (real MT5)</div></div>
  <div class="scard"><div class="sv" style="color:{_pnl_color(real_today)}">{real_today_s}</div><div class="sl">P&amp;L hoy (real)</div></div>
  <div class="scard"><div class="sv" style="color:{_pnl_color(real_7d)}">{real_7d_s}</div><div class="sl">P&amp;L 7 días (real)</div></div>
  <div class="scard"><div class="sv">{bal_s}</div><div class="sl">MT5 Balance</div></div>
  <div class="scard"><div class="sv">{eq_s}</div><div class="sl">MT5 Equity</div></div>
  <div class="scard"><div class="sv" style="color:{_pnl_color(live_diff)}">{live_diff_s}</div><div class="sl">P&amp;L abierto MT5</div></div>
</div>

<!-- POSICIONES ABIERTAS (MT5 ground truth) -->
<h2>📌 Posiciones abiertas</h2>
<div class="grid">
{positions_html}
</div>

<!-- HISTORIAL CERRADO POR MAGIC -->
<h2>📒 Historial por robot (real MT5)</h2>
<div class="grid">
{closed_magic_html}
</div>

<!-- PROP FIRM READINESS -->
<h2>🎯 Calificación Prop Firm</h2>
<div class="grid">
{prop_html}
</div>

<!-- INSTRUMENT CARDS -->
<h2>Instrumentos</h2>
<div class="grid">
{cards_html}
</div>

<!-- PORTFOLIO BREAKDOWN -->
<h2>Desglose del Portfolio — P&amp;L Real MT5 por instrumento</h2>
<div class="tbl-wrap" style="margin-bottom:14px">
  <table>
    <thead><tr>
      <th>Instrumento</th>
      <th style="text-align:right">Señales</th>
      <th style="text-align:right">Wins</th>
      <th style="text-align:right">Losses</th>
      <th style="text-align:right">Win %</th>
      <th>Barra</th>
      <th style="text-align:right">P&amp;L</th>
      <th style="text-align:right">Avg R:R</th>
      <th style="text-align:right">Prime (CT)</th>
    </tr></thead>
    <tbody>{pnl_bars}</tbody>
  </table>
</div>

<div class="disc">
  &#9888; <b>P&L Real MT5</b>: se obtiene de <code>history_deals_get()</code>; cada señal se cruza con los deals reales cerrados en su ventana de tiempo. La señal activa muestra precio live vs SL/TP como P&L flotante.<br>
  <b>Resto de instrumentos — P&L Simulado</b>: cada señal histórica descarga barras M5 de MT5 y escanea candle por candle hasta que HIGH≥TP o LOW≤SL; si ningún nivel es tocado en la ventana, fallback al precio de la siguiente señal. La última señal usa precio live vs SL/TP. <b>No ejecutan órdenes reales en MT5.</b> El balance/equity y P&L abierto de MT5 son datos live de la cuenta demo.
</div>

<!-- ALL SIGNALS TABLE -->
<h2>Todas las Señales — Más Recientes Primero</h2>
<div class="tbl-wrap">
  <table>
    <thead><tr>
      <th>Hora (UTC)</th><th>Símbolo</th><th>Señal</th>
      <th>Entry</th><th>SL</th><th>TP</th>
      <th>R:R</th><th>Conf</th><th>Estado</th><th>P&amp;L</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

</body>
</html>"""
    return html


# ── HTTP server ───────────────────────────────────────────────────────────────

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle each HTTP request in its own thread."""
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = _build_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type",   "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def start(port=PORT):
    srv = ThreadedHTTPServer(("0.0.0.0", port), Handler)
    print(f"Master dashboard -> http://localhost:{port}")
    print("Ctrl+C para detener.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nDetenido.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Portfolio Dashboard")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    start(args.port)
