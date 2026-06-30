"""
XAUUSD Gold — MT5 Bridge
Thin adapter over the soja MT5 bridge, pre-configured for XAUUSD.

Patches DEFAULT_SYMBOL and FALLBACK_SYMBOLS from config, then re-exports
all functions from the original bridge so this module is a drop-in replacement.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _MVP_ROOT not in sys.path:
    sys.path.insert(0, _MVP_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import SYMBOL, FALLBACK_SYMBOLS

# Override env var before importing the bridge so it picks up our symbol
os.environ.setdefault("MT5_SYMBOL", SYMBOL)

# Import original bridge
import src.intraday.data.mt5_bridge as _bridge

# Patch symbol in-place so all bridge functions use XAUUSD
_bridge.DEFAULT_SYMBOL  = SYMBOL
_bridge.FALLBACK_SYMBOLS = FALLBACK_SYMBOLS

# Re-export everything
from src.intraday.data.mt5_bridge import (
    initialize,
    shutdown,
    is_connected,
    fetch_mt5_bars,
    get_live_tick,
    stream_ticks,
    get_account_info,
    get_positions,
    place_order,
    close_position,
    modify_position_sl,
    diagnose,
)

__all__ = [
    "initialize", "shutdown", "is_connected",
    "fetch_mt5_bars", "get_live_tick", "stream_ticks",
    "get_account_info", "get_positions", "place_order",
    "close_position", "diagnose", "SYMBOL", "FALLBACK_SYMBOLS",
]
