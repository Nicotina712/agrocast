"""US500 S&P 500 Robot — MT5 Bridge"""
import os, sys

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import SYMBOL, FALLBACK_SYMBOLS
os.environ.setdefault("MT5_SYMBOL", SYMBOL)

from src.intraday.data import mt5_bridge as _bridge
_bridge.DEFAULT_SYMBOL   = SYMBOL
_bridge.FALLBACK_SYMBOLS = FALLBACK_SYMBOLS

from src.intraday.data.mt5_bridge import (
    initialize, shutdown, is_connected,
    fetch_mt5_bars, get_live_tick, get_account_info,
    get_positions, place_order, close_position, diagnose,
    modify_position_sl,
)
__all__ = [
    "initialize","shutdown","is_connected","fetch_mt5_bars","get_live_tick",
    "get_account_info","get_positions","place_order","close_position","modify_position_sl","diagnose",
]
