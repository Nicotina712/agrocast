"""
BTCUSD Bitcoin Robot — MT5 Bridge
Thin wrapper over the shared MT5 bridge, patching symbol to BTCUSD.
"""

import os
import sys

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

# Ensure MVP root is on path
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import SYMBOL, FALLBACK_SYMBOLS

# Set env var so base bridge picks up our symbol
os.environ.setdefault("MT5_SYMBOL", SYMBOL)

# Import and patch the shared bridge
from src.intraday.data import mt5_bridge as _bridge

_bridge.DEFAULT_SYMBOL   = SYMBOL
_bridge.FALLBACK_SYMBOLS = FALLBACK_SYMBOLS

# Re-export all public functions
from src.intraday.data.mt5_bridge import (
    initialize,
    shutdown,
    is_connected,
    fetch_mt5_bars,
    get_live_tick,
    get_account_info,
    get_positions,
    place_order,
    close_position,
    modify_position_sl,
    diagnose,
)

__all__ = [
    "initialize", "shutdown", "is_connected",
    "fetch_mt5_bars", "get_live_tick", "get_account_info",
    "get_positions", "place_order", "close_position", "modify_position_sl", "diagnose",
]
