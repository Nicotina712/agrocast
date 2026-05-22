"""
src/intraday/data/bar_builder.py
Tick -> Bar aggregation using MT5 live tick data.

Converts raw tick stream from MT5 into OHLCV bars at any interval.
Used for sub-minute analysis or custom bar sizes.

For standard intervals (1m, 5m, 15m, 60m), prefer mt5_bridge.fetch_mt5_bars()
which uses MT5's native bar API. Use this module only when you need:
  - Custom intervals (e.g., 3m, 7m)
  - Tick-level aggregation with volume profiling
  - Real-time bar building from streaming ticks
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Callable

import numpy as np
import pandas as pd


def aggregate_ticks_to_bars(
    ticks: list[dict],
    interval_seconds: int = 300,
) -> pd.DataFrame:
    """
    Aggregate tick dicts into OHLCV bars.

    Args:
        ticks: list of dicts with keys: time (datetime), last (float), volume (int)
        interval_seconds: bar interval in seconds (default 300 = 5m)

    Returns:
        DataFrame with columns: open, high, low, close, volume, tick_count
        Indexed by bar start time (UTC)
    """
    if not ticks:
        return pd.DataFrame()

    df = pd.DataFrame(ticks)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time")

    # Floor to interval
    freq = f"{interval_seconds}s"
    df["bar_time"] = df["time"].dt.floor(freq)

    bars = df.groupby("bar_time").agg(
        open=("last", "first"),
        high=("last", "max"),
        low=("last", "min"),
        close=("last", "last"),
        volume=("volume", "sum"),
        tick_count=("last", "count"),
    )

    bars.index.name = None
    return bars


def build_bars_from_mt5_ticks(
    symbol: Optional[str] = None,
    duration_sec: int = 3600,
    interval_seconds: int = 300,
    callback: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Stream ticks from MT5 and build bars in real-time.

    Args:
        symbol: MT5 symbol (auto-detected if None)
        duration_sec: how long to collect ticks
        interval_seconds: bar size in seconds
        callback: called with (completed_bar_dict) when a bar closes

    Returns:
        DataFrame of completed bars
    """
    from src.intraday.data.mt5_bridge import stream_ticks

    all_ticks = []
    last_bar_time = None

    def _on_tick(tick):
        nonlocal last_bar_time
        all_ticks.append(tick)

        # Check if a bar just closed
        t = pd.to_datetime(tick["time"], utc=True)
        bar_time = t.floor(f"{interval_seconds}s")

        if last_bar_time is not None and bar_time > last_bar_time:
            # Previous bar is complete — aggregate and callback
            bar_ticks = [tk for tk in all_ticks
                         if pd.to_datetime(tk["time"], utc=True).floor(f"{interval_seconds}s") == last_bar_time]
            if bar_ticks and callback:
                bar_df = aggregate_ticks_to_bars(bar_ticks, interval_seconds)
                if not bar_df.empty:
                    bar_dict = bar_df.iloc[0].to_dict()
                    bar_dict["time"] = last_bar_time
                    callback(bar_dict)

        last_bar_time = bar_time

    ticks = stream_ticks(
        symbol=symbol,
        callback=_on_tick,
        duration_sec=duration_sec,
        interval_ms=250,  # faster polling for tick-level data
    )

    # Build final bars from all collected ticks
    return aggregate_ticks_to_bars(ticks, interval_seconds)
