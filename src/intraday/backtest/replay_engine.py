"""
src/intraday/backtest/replay_engine.py
Event-driven bar-by-bar replay con fills realistas, slippage, y SL/TP.

Simula:
  - Una posición abierta a la vez (no hay piramidación en Fase 0)
  - Entry al close de la barra de señal (LMT/MKT) con slippage
  - SL/TP intra-barra usando high/low de barras siguientes
  - Time stop: cierre forzado a HORIZON barras si no se ejecutó SL/TP
  - Latency: la señal generada en barra t se ejecuta en barra t+1 (no peek)
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from datetime import timedelta

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from src.intraday.execution.slippage_model import (
    ContractSpec, MZS, fill_price, pnl_usd,
)
from src.intraday.execution.risk_intraday import (
    RiskConfig, RiskState, compute_position_size, compute_sl_tp,
    can_open_trade, register_trade_outcome, reset_daily_if_needed,
)


def _exit_intra_bar(row, side: str, sl: float, tp: float):
    """
    Detecta si SL o TP se tocan dentro de la barra. Pesimismo: si ambos en
    misma barra, asumimos que SL pega primero.
    """
    h, l = float(row["high"]), float(row["low"])
    if side == "BUY":
        sl_hit = l <= sl
        tp_hit = h >= tp
        if sl_hit and tp_hit: return ("SL", sl)
        if sl_hit:            return ("SL", sl)
        if tp_hit:            return ("TP", tp)
    else:  # SELL
        sl_hit = h >= sl
        tp_hit = l <= tp
        if sl_hit and tp_hit: return ("SL", sl)
        if sl_hit:            return ("SL", sl)
        if tp_hit:            return ("TP", tp)
    return (None, None)


def replay(
    signals_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    initial_capital: float = 10_000,
    contract: ContractSpec = MZS,
    horizon_bars: int = 12,
    cfg: RiskConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Args:
      signals_df: salida de predict_intraday (datetime, close, atr_14, side_router, ...)
      bars_df:    barras OHLCV originales (necesita high/low para SL/TP intra-barra)
      initial_capital: USD
      contract: MZS o ZS
      horizon_bars: time stop si no ejecuta SL/TP
    Returns:
      (trades_df, metrics_summary)
    """
    cfg = cfg or RiskConfig()
    # Merge bars on datetime
    sig = signals_df.copy()
    sig["datetime"] = pd.to_datetime(sig["datetime"], utc=True)
    bars = bars_df.copy()
    bars["datetime"] = pd.to_datetime(bars["datetime"], utc=True)
    df = pd.merge(sig, bars[["datetime", "open", "high", "low"]], on="datetime", how="left")
    df = df.sort_values("datetime").reset_index(drop=True)

    state = RiskState(capital=initial_capital,
                      starting_capital_today=initial_capital,
                      last_session_date=df["datetime"].iloc[0].date().isoformat())
    trades = []
    open_pos = None  # dict con keys: side, entry, sl, tp, n, entry_idx, entry_dt

    for i, row in df.iterrows():
        now = row["datetime"].to_pydatetime()
        reset_daily_if_needed(state, now)

        # ── Si hay posición abierta, ver si exit intra-barra ───────────
        if open_pos is not None:
            tag, px = _exit_intra_bar(row, open_pos["side"], open_pos["sl"], open_pos["tp"])
            time_stop = (i - open_pos["entry_idx"]) >= horizon_bars
            if tag is not None or time_stop:
                if tag is None:  # time stop usa close
                    tag = "TIME"; px = float(row["close"])
                # Slippage en exit
                exit_side = "SELL" if open_pos["side"] == "BUY" else "BUY"
                exit_px = fill_price(exit_side, px, contract, extra_slippage_ticks=1.0)
                gross = pnl_usd(open_pos["entry"], exit_px, open_pos["side"],
                                contract, open_pos["n"])
                net = gross - contract.commission_round_trip_usd * open_pos["n"]
                trades.append({
                    "entry_dt": open_pos["entry_dt"],
                    "exit_dt":  row["datetime"],
                    "side":     open_pos["side"],
                    "entry":    open_pos["entry"],
                    "exit":     exit_px,
                    "n":        open_pos["n"],
                    "exit_tag": tag,
                    "gross_pnl": gross,
                    "net_pnl":  net,
                    "bars_held": i - open_pos["entry_idx"],
                })
                register_trade_outcome(state, net, now)
                open_pos = None

        # ── Si no hay posición, ver si abrir ───────────────────────────
        if open_pos is None and i + 1 < len(df):
            side = row.get("side_router", "HOLD")
            if side in ("BUY", "SELL"):
                ok, reason = can_open_trade(state, now, cfg)
                if not ok:
                    continue
                # Ejecutamos en la barra siguiente al close (latency = 1 bar)
                next_row = df.iloc[i + 1]
                entry_quote = float(next_row["open"])
                import math
                atr_raw = row.get("atr_14", 0)
                atr = float(atr_raw) if atr_raw is not None else 0
                if math.isnan(atr) or atr <= 0:
                    continue
                if math.isnan(entry_quote):
                    continue
                exp_vol = float(row.get("swing_expected_vol", 0)) if "swing_expected_vol" in row.index else 0
                sl, tp = compute_sl_tp(entry_quote, atr, side, cfg, exp_vol)
                sl_dist = abs(entry_quote - sl)
                n = compute_position_size(state.capital, sl_dist, contract, cfg)
                if n <= 0:
                    continue
                fill = fill_price(side, entry_quote, contract, extra_slippage_ticks=1.0)
                open_pos = {
                    "side": side, "entry": fill, "sl": sl, "tp": tp,
                    "n": n, "entry_idx": i + 1,
                    "entry_dt": next_row["datetime"],
                }

    # Cerrar posición pendiente al final
    if open_pos is not None:
        last = df.iloc[-1]
        exit_side = "SELL" if open_pos["side"] == "BUY" else "BUY"
        exit_px = fill_price(exit_side, float(last["close"]), contract, 1.0)
        gross = pnl_usd(open_pos["entry"], exit_px, open_pos["side"], contract, open_pos["n"])
        net = gross - contract.commission_round_trip_usd * open_pos["n"]
        trades.append({
            "entry_dt": open_pos["entry_dt"], "exit_dt": last["datetime"],
            "side": open_pos["side"], "entry": open_pos["entry"], "exit": exit_px,
            "n": open_pos["n"], "exit_tag": "EOD",
            "gross_pnl": gross, "net_pnl": net,
            "bars_held": len(df) - 1 - open_pos["entry_idx"],
        })

    trades_df = pd.DataFrame(trades)
    summary = {
        "n_trades":       int(len(trades_df)),
        "final_capital":  round(float(state.capital), 2),
        "total_pnl":      round(float(state.capital - initial_capital), 2),
        "return_pct":     round(float((state.capital / initial_capital - 1) * 100), 2),
    }
    return trades_df, summary
