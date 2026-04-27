"""
src/intraday/backtest/metrics.py
Métricas de performance intradía.

Sharpe (anualizado, asumiendo 252 trading days).
Profit Factor = sum(wins) / abs(sum(losses)).
Win rate, expectancy, max drawdown, average win/loss ratio.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_metrics(
    trades_df: pd.DataFrame,
    initial_capital: float = 10_000,
    sessions_per_year: int = 252,
) -> dict:
    if trades_df is None or trades_df.empty:
        return {"n_trades": 0, "verdict": "NO_TRADES"}

    df = trades_df.copy()
    df["exit_dt"] = pd.to_datetime(df["exit_dt"], utc=True)
    df = df.sort_values("exit_dt").reset_index(drop=True)

    n = len(df)
    pnl = df["net_pnl"].astype(float)
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    total_pnl = float(pnl.sum())
    win_rate  = float((pnl > 0).mean())
    avg_win   = float(wins.mean())   if len(wins)   else 0.0
    avg_loss  = float(losses.mean()) if len(losses) else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    expectancy = float(pnl.mean())
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss else float("inf")

    # Equity curve y drawdown
    equity = initial_capital + pnl.cumsum()
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    max_dd = float(dd.min())

    # Sharpe diario → anualizado
    df["session"] = df["exit_dt"].dt.tz_convert("America/Chicago").dt.date
    daily = df.groupby("session")["net_pnl"].sum()
    daily_ret = daily / initial_capital
    if daily_ret.std() > 0:
        sharpe_d = daily_ret.mean() / daily_ret.std()
        sharpe_ann = sharpe_d * np.sqrt(sessions_per_year)
    else:
        sharpe_ann = 0.0

    # Win/Loss breakdown por exit_tag
    by_tag = df.groupby("exit_tag")["net_pnl"].agg(["count", "sum", "mean"]).round(2).to_dict("index") \
        if "exit_tag" in df.columns else {}

    # Side breakdown
    by_side = df.groupby("side")["net_pnl"].agg(["count", "sum", "mean"]).round(2).to_dict("index") \
        if "side" in df.columns else {}

    metrics = {
        "n_trades":      int(n),
        "n_sessions":    int(daily.shape[0]),
        "total_pnl":     round(total_pnl, 2),
        "return_pct":    round(total_pnl / initial_capital * 100, 2),
        "win_rate":      round(win_rate, 4),
        "n_wins":        int(len(wins)),
        "n_losses":      int(len(losses)),
        "avg_win":       round(avg_win, 2),
        "avg_loss":      round(avg_loss, 2),
        "win_loss_ratio": round(win_loss_ratio, 2) if win_loss_ratio != float("inf") else None,
        "profit_factor": round(pf, 2) if pf != float("inf") else None,
        "expectancy_per_trade": round(expectancy, 2),
        "sharpe_annualized":    round(float(sharpe_ann), 2),
        "max_drawdown_pct":     round(max_dd * 100, 2),
        "by_exit_tag":  by_tag,
        "by_side":      by_side,
    }

    # Gate Fase 0 → Fase 1
    gate_pass = (
        metrics["sharpe_annualized"] >= 0.8
        and (metrics["profit_factor"] or 0) >= 1.4
        and metrics["win_rate"] >= 0.48
        and metrics["max_drawdown_pct"] >= -15
    )
    metrics["fase0_gate"] = "PASS" if gate_pass else "FAIL"
    return metrics


def print_metrics_report(m: dict) -> None:
    print("=" * 60)
    print("  BACKTEST INTRADAY — RESUMEN")
    print("=" * 60)
    if m.get("n_trades", 0) == 0:
        print("  (no hubo trades en el período)")
        return
    rows = [
        ("Trades",            m["n_trades"]),
        ("Sesiones",          m["n_sessions"]),
        ("PnL total (USD)",   m["total_pnl"]),
        ("Return %",          f'{m["return_pct"]}%'),
        ("Win rate",          f'{m["win_rate"]:.1%}'),
        ("Wins / Losses",     f'{m["n_wins"]} / {m["n_losses"]}'),
        ("Avg win",           f'${m["avg_win"]}'),
        ("Avg loss",          f'${m["avg_loss"]}'),
        ("Win/Loss ratio",    m.get("win_loss_ratio")),
        ("Profit factor",     m.get("profit_factor")),
        ("Expectancy/trade",  f'${m["expectancy_per_trade"]}'),
        ("Sharpe anualizado", m["sharpe_annualized"]),
        ("Max drawdown",      f'{m["max_drawdown_pct"]}%'),
        ("Gate Fase 0",       m.get("fase0_gate")),
    ]
    for label, val in rows:
        print(f"  {label:<22}{val}")
    print()
    if m.get("by_exit_tag"):
        print("  Por exit tag:")
        for tag, stats in m["by_exit_tag"].items():
            print(f"    {tag:<6} n={int(stats['count']):>3} "
                  f"sum=${stats['sum']:>9.2f} mean=${stats['mean']:>7.2f}")
    if m.get("by_side"):
        print("  Por side:")
        for side, stats in m["by_side"].items():
            print(f"    {side:<6} n={int(stats['count']):>3} "
                  f"sum=${stats['sum']:>9.2f} mean=${stats['mean']:>7.2f}")
    print("=" * 60)
