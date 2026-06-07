"""
src/intraday/data/mt5_bridge.py
MT5 Data Bridge — Real-time tick and bar data via MetaTrader 5 local connection.

Drop-in replacement for tick_feed.py. Same interface, zero delay.
Falls back to yfinance if MT5 is not available.

Requirements:
  - MetaTrader 5 terminal running on this machine
  - pip install MetaTrader5
  - Demo or live account connected

API (same as tick_feed.py):
  fetch_mt5_bars(interval, n_bars, symbol) -> pd.DataFrame
  fetch_intraday_bars_mt5(interval, ...) -> pd.DataFrame  (drop-in for tick_feed)
  get_live_tick(symbol) -> dict
  stream_ticks(symbol, callback, duration_sec) -> None
  get_account_info() -> dict
  place_order(...) -> dict   (Phase 4: execution)
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable

import pandas as pd
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# MT5 timeframe mapping
_TF_MAP = {}
_MT5_AVAILABLE = False

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
    _TF_MAP = {
        "1m": mt5.TIMEFRAME_M1,
        "2m": mt5.TIMEFRAME_M2,
        "3m": mt5.TIMEFRAME_M3,
        "5m": mt5.TIMEFRAME_M5,
        "10m": mt5.TIMEFRAME_M10,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "60m": mt5.TIMEFRAME_H1,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }
except ImportError:
    mt5 = None

# Default symbol — must match your MT5 broker's naming
DEFAULT_SYMBOL = os.environ.get("MT5_SYMBOL", "Sbean_N6")
# Fallback symbols to try if default fails
FALLBACK_SYMBOLS = ["Sbean_N6", "SOYBEAN", "ZS", "SOYBEANS", "Soybean"]

# Connection state
_initialized = False


# =========================================================================
#  CONNECTION MANAGEMENT
# =========================================================================

def initialize() -> bool:
    """Initialize MT5 connection. Returns True if successful."""
    global _initialized
    if not _MT5_AVAILABLE:
        print("[MT5] MetaTrader5 package not installed")
        return False
    if _initialized:
        return True
    try:
        ok = mt5.initialize()
        if ok:
            _initialized = True
            info = mt5.terminal_info()
            print(f"[MT5] Connected to {info.name} build {info.build}")
            return True
        else:
            err = mt5.last_error()
            print(f"[MT5] Initialize failed: {err}")
            return False
    except Exception as e:
        print(f"[MT5] Initialize error: {e}")
        return False


def shutdown():
    """Close MT5 connection."""
    global _initialized
    if _MT5_AVAILABLE and _initialized:
        mt5.shutdown()
        _initialized = False


def is_connected() -> bool:
    """Check if MT5 is connected and terminal is active."""
    if not _initialized:
        return False
    try:
        info = mt5.terminal_info()
        return info is not None and info.connected
    except Exception:
        return False


def _resolve_symbol(symbol: Optional[str] = None) -> Optional[str]:
    """Find a valid soybean symbol in MT5."""
    candidates = [symbol] if symbol else []
    candidates.extend([DEFAULT_SYMBOL] + FALLBACK_SYMBOLS)

    for sym in candidates:
        if not sym:
            continue
        info = mt5.symbol_info(sym)
        if info is not None:
            mt5.symbol_select(sym, True)
            return sym

    # Last resort: search all symbols
    all_syms = mt5.symbols_get()
    if all_syms:
        for s in all_syms:
            name_lower = s.name.lower()
            if "soy" in name_lower or "sbean" in name_lower:
                mt5.symbol_select(s.name, True)
                return s.name
    return None


# =========================================================================
#  DATA RETRIEVAL
# =========================================================================

def get_live_tick(symbol: Optional[str] = None) -> Optional[dict]:
    """Get the latest tick for the symbol. Zero latency."""
    if not initialize():
        return None

    sym = _resolve_symbol(symbol)
    if not sym:
        print("[MT5] No soybean symbol found")
        return None

    tick = mt5.symbol_info_tick(sym)
    if tick is None:
        return None

    return {
        "symbol": sym,
        "bid": tick.bid,
        "ask": tick.ask,
        "last": tick.last if tick.last > 0 else (tick.bid + tick.ask) / 2,
        "volume": tick.volume,
        "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
        "spread": round(tick.ask - tick.bid, 2),
        "flags": tick.flags,
    }


def fetch_mt5_bars(
    interval: str = "60m",
    n_bars: int = 500,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from MT5. Real-time, zero delay.

    Args:
        interval: "1m", "5m", "15m", "60m", "4h", "1d"
        n_bars: number of bars to fetch (max ~100k depending on broker)
        symbol: MT5 symbol name (auto-detected if None)

    Returns:
        DataFrame with columns: time, open, high, low, close, volume, spread
        Index is DatetimeIndex (UTC)
    """
    if not initialize():
        return pd.DataFrame()

    tf = _TF_MAP.get(interval)
    if tf is None:
        print(f"[MT5] Unknown interval: {interval}")
        return pd.DataFrame()

    sym = _resolve_symbol(symbol)
    if not sym:
        print("[MT5] No soybean symbol found")
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(sym, tf, 0, n_bars)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        print(f"[MT5] No bars for {sym} {interval}: {err}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "tick_volume": "volume",
        "real_volume": "real_volume",
    })

    # Keep standard columns
    cols = ["time", "open", "high", "low", "close", "volume", "spread"]
    available = [c for c in cols if c in df.columns]
    df = df[available].copy()
    df = df.set_index("time")
    df.index.name = None

    # Rename to match tick_feed.py output format
    df.columns = [c.capitalize() if c != "volume" else "Volume" for c in df.columns]
    df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})

    # Lowercase to match tick_feed convention
    df.columns = [c.lower() for c in df.columns]

    print(f"[MT5] {sym} {interval}: {len(df)} bars [{df.index[0]} -> {df.index[-1]}]")
    return df


def fetch_intraday_bars_mt5(
    interval: str = "60m",
    symbol: Optional[str] = None,
    n_bars: int = 2000,
    **kwargs,
) -> pd.DataFrame:
    """
    Drop-in replacement for tick_feed.fetch_intraday_bars().
    Same output format, but with real-time data from MT5.
    Falls back to yfinance if MT5 is unavailable.
    """
    df = fetch_mt5_bars(interval=interval, n_bars=n_bars, symbol=symbol)

    if df.empty:
        print("[MT5] Falling back to yfinance...")
        from src.intraday.data.tick_feed import fetch_intraday_bars
        return fetch_intraday_bars(interval=interval, **kwargs)

    return df


def stream_ticks(
    symbol: Optional[str] = None,
    callback: Optional[Callable] = None,
    duration_sec: int = 60,
    interval_ms: int = 500,
) -> list:
    """
    Stream live ticks for a duration. Calls callback(tick_dict) on each new tick.
    Returns list of all ticks collected.

    Args:
        symbol: MT5 symbol
        callback: function(tick_dict) called on each tick
        duration_sec: how long to stream
        interval_ms: polling interval in ms (MT5 doesn't have push — we poll)
    """
    if not initialize():
        return []

    sym = _resolve_symbol(symbol)
    if not sym:
        return []

    ticks = []
    last_time = 0
    end_time = time.time() + duration_sec

    print(f"[MT5] Streaming {sym} for {duration_sec}s...")

    while time.time() < end_time:
        tick = mt5.symbol_info_tick(sym)
        if tick and tick.time != last_time:
            last_time = tick.time
            tick_dict = {
                "symbol": sym,
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last if tick.last > 0 else (tick.bid + tick.ask) / 2,
                "volume": tick.volume,
                "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
            }
            ticks.append(tick_dict)
            if callback:
                callback(tick_dict)
        time.sleep(interval_ms / 1000)

    print(f"[MT5] Collected {len(ticks)} ticks")
    return ticks


# =========================================================================
#  ACCOUNT & EXECUTION (Phase 4)
# =========================================================================

def get_account_info() -> Optional[dict]:
    """Get account balance, equity, margin info."""
    if not initialize():
        return None
    info = mt5.account_info()
    if not info:
        return None
    return {
        "login": info.login,
        "server": info.server,
        "balance": info.balance,
        "equity": info.equity,
        "margin": info.margin,
        "free_margin": info.margin_free,
        "leverage": info.leverage,
        "profit": info.profit,
        "currency": info.currency,
        "trade_mode": "demo" if info.trade_mode == 0 else "live",
    }


def get_positions(symbol: Optional[str] = None) -> list:
    """Get open positions, optionally filtered by symbol."""
    if not initialize():
        return []
    sym = _resolve_symbol(symbol) if symbol else None
    if sym:
        positions = mt5.positions_get(symbol=sym)
    else:
        positions = mt5.positions_get()
    if positions is None:
        return []
    return [
        {
            "ticket": p.ticket,
            "symbol": p.symbol,
            "type": "BUY" if p.type == 0 else "SELL",
            "volume": p.volume,
            "price_open": p.price_open,
            "price_current": p.price_current,
            "sl": p.sl,
            "tp": p.tp,
            "profit": p.profit,
            "swap": p.swap,
            "time": datetime.fromtimestamp(p.time, tz=timezone.utc),
            "magic": p.magic,
            "comment": p.comment,
        }
        for p in positions
    ]


def get_trade_history(days: int = 7) -> list:
    """Get closed trade history for the last N days."""
    if not initialize():
        return []
    from_date = datetime.now() - timedelta(days=days)
    to_date = datetime.now() + timedelta(days=1)
    deals = mt5.history_deals_get(from_date, to_date)
    if deals is None:
        return []
    return [
        {
            "ticket": d.ticket,
            "order": d.order,
            "symbol": d.symbol,
            "type": d.type,
            "volume": d.volume,
            "price": d.price,
            "profit": d.profit,
            "swap": d.swap,
            "commission": d.commission,
            "time": datetime.fromtimestamp(d.time, tz=timezone.utc),
            "magic": d.magic,
            "comment": d.comment,
            "entry": d.entry,  # 0=in, 1=out, 2=reverse
        }
        for d in deals
    ]


def place_order(
    direction: str,  # "BUY" or "SELL"
    symbol: Optional[str] = None,
    volume: float = 0.01,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    comment: str = "QA_V3",
    magic: int = 20260522,
    dry_run: bool = True,
) -> dict:
    """
    Place a market order via MT5.

    Args:
        direction: "BUY" or "SELL"
        symbol: MT5 symbol (auto-detected)
        volume: lot size
        sl: stop loss price
        tp: take profit price
        comment: order comment
        magic: magic number for tracking
        dry_run: if True, only validate — don't execute

    Returns dict with order result or error.
    """
    if not initialize():
        return {"ok": False, "error": "MT5 not connected"}

    sym = _resolve_symbol(symbol)
    if not sym:
        return {"ok": False, "error": "No soybean symbol found"}

    tick = mt5.symbol_info_tick(sym)
    if not tick:
        return {"ok": False, "error": f"No tick data for {sym}"}

    order_type = mt5.ORDER_TYPE_BUY if direction.upper() == "BUY" else mt5.ORDER_TYPE_SELL
    price = tick.ask if direction.upper() == "BUY" else tick.bid

    # Auto-detect filling mode supported by the symbol
    # filling_mode is a bitmask: bit0=FOK, bit1=IOC
    # ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1, ORDER_FILLING_RETURN=2
    sym_info = mt5.symbol_info(sym)
    filling = mt5.ORDER_FILLING_IOC  # default
    if sym_info:
        fm = sym_info.filling_mode
        if fm & 2:    # bit 1 → IOC supported
            filling = mt5.ORDER_FILLING_IOC
        elif fm & 1:  # bit 0 → FOK supported
            filling = mt5.ORDER_FILLING_FOK
        else:
            filling = mt5.ORDER_FILLING_RETURN

    # Respect symbol volume constraints
    if sym_info:
        vol_min = sym_info.volume_min
        vol_step = sym_info.volume_step
        if volume < vol_min:
            volume = vol_min
        # Round to nearest step
        volume = round(round(volume / vol_step) * vol_step, 8)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": sym,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl or 0.0,
        "tp": tp or 0.0,
        "deviation": 5,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    if dry_run:
        # Validate only
        check = mt5.order_check(request)
        if check is None:
            return {"ok": False, "error": str(mt5.last_error()), "dry_run": True}
        return {
            "ok": check.retcode == mt5.TRADE_RETCODE_DONE or check.retcode == 0,
            "retcode": check.retcode,
            "comment": check.comment,
            "margin": check.margin,
            "balance": check.balance,
            "dry_run": True,
            "request": {k: v for k, v in request.items() if k != "action"},
        }

    # Execute
    result = mt5.order_send(request)
    if result is None:
        return {"ok": False, "error": str(mt5.last_error())}

    return {
        "ok": result.retcode == mt5.TRADE_RETCODE_DONE,
        "retcode": result.retcode,
        "order": result.order,
        "deal": result.deal,
        "volume": result.volume,
        "price": result.price,
        "comment": result.comment,
        "request_id": result.request_id,
    }


# =========================================================================
#  CLOSE POSITION
# =========================================================================

def close_position(
    ticket:  int,
    symbol:  Optional[str] = None,
    comment: str = "close",
    magic:   int = 20260522,
    dry_run: bool = False,
) -> dict:
    """Close an open MT5 position by ticket number (opposite market order)."""
    if not initialize():
        return {"ok": False, "error": "MT5 not connected"}
    sym = _resolve_symbol(symbol)
    if not sym:
        return {"ok": False, "error": "Symbol not resolved"}
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return {"ok": False, "error": f"Position {ticket} not found"}
    pos = positions[0]
    close_type  = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    tick        = mt5.symbol_info_tick(sym)
    close_price = tick.bid if pos.type == 0 else tick.ask
    sym_info    = mt5.symbol_info(sym)
    filling     = mt5.ORDER_FILLING_IOC
    if sym_info:
        fm = sym_info.filling_mode
        if fm & 2:   filling = mt5.ORDER_FILLING_IOC
        elif fm & 1: filling = mt5.ORDER_FILLING_FOK
        else:        filling = mt5.ORDER_FILLING_RETURN
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "position":     ticket,
        "symbol":       sym,
        "volume":       pos.volume,
        "type":         close_type,
        "price":        close_price,
        "deviation":    20,
        "magic":        magic,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }
    if dry_run:
        check = mt5.order_check(request)
        if check is None:
            return {"ok": False, "error": str(mt5.last_error()), "dry_run": True}
        return {"ok": check.retcode in (mt5.TRADE_RETCODE_DONE, 0),
                "retcode": check.retcode, "dry_run": True, "ticket": ticket}
    result = mt5.order_send(request)
    if result is None:
        return {"ok": False, "error": str(mt5.last_error())}
    return {"ok": result.retcode == mt5.TRADE_RETCODE_DONE,
            "retcode": result.retcode, "order": result.order,
            "deal": result.deal, "price": result.price, "ticket": ticket}


def modify_position_sl(ticket: int, new_sl: float, new_tp: float = None) -> bool:
    """Modifica el SL (y opcionalmente TP) de una posición abierta. Para trailing stop."""
    if not initialize():
        return False
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return False
    pos = positions[0]
    sym_info = mt5.symbol_info(pos.symbol)
    digits = sym_info.digits if sym_info else 5
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol":   pos.symbol,
        "sl":       round(float(new_sl), digits),
        "tp":       round(float(new_tp), digits) if new_tp else pos.tp,
    }
    try:
        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
    except Exception:
        return False


# =========================================================================
#  DIAGNOSTICS
# =========================================================================

def diagnose() -> dict:
    """Full diagnostic of MT5 connection and data availability."""
    result = {
        "mt5_package": _MT5_AVAILABLE,
        "connected": False,
        "symbol": None,
        "has_bars": False,
        "has_ticks": False,
        "account": None,
        "latency_ms": None,
    }

    if not _MT5_AVAILABLE:
        result["error"] = "MetaTrader5 package not installed"
        return result

    if not initialize():
        result["error"] = "Cannot connect to MT5 terminal"
        return result

    result["connected"] = True

    sym = _resolve_symbol()
    result["symbol"] = sym

    if sym:
        # Test bar retrieval speed
        t0 = time.time()
        bars = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 10)
        latency = (time.time() - t0) * 1000
        result["latency_ms"] = round(latency, 1)
        result["has_bars"] = bars is not None and len(bars) > 0

        tick = mt5.symbol_info_tick(sym)
        result["has_ticks"] = tick is not None and tick.bid > 0

        if tick:
            result["current_price"] = {
                "bid": tick.bid,
                "ask": tick.ask,
                "time": datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat(),
            }

    result["account"] = get_account_info()
    return result


# =========================================================================
#  CLI
# =========================================================================

if __name__ == "__main__":
    import json as _json
    print("=== MT5 Bridge Diagnostic ===\n")
    diag = diagnose()
    print(_json.dumps(diag, indent=2, default=str))

    if diag["connected"] and diag["symbol"]:
        print(f"\n=== Last 10 bars (5m) ===")
        df = fetch_mt5_bars("5m", n_bars=10)
        if not df.empty:
            print(df.to_string())

        print(f"\n=== Account ===")
        acc = get_account_info()
        if acc:
            for k, v in acc.items():
                print(f"  {k}: {v}")

    shutdown()
