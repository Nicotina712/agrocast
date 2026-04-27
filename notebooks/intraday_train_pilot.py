"""
notebooks/intraday_train_pilot.py
Orquestador end-to-end Fase 0:

  1. Fetch barras 5m (yfinance ZS=F)
  2. Build features + swing context + regime
  3. Train walk-forward XGBoost
  4. Predict + route signals
  5. Backtest replay con MZS, slippage, comisiones
  6. Métricas + verdict gate Fase 0

Output: artifacts/intraday/intraday_metrics.json + intraday_backtest.csv

Uso: python notebooks/intraday_train_pilot.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

# UTF-8 stdout para Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from src.intraday.data.tick_feed import fetch_intraday_bars
from src.intraday.model.train_intraday import main as train_main
from src.intraday.model.predict_intraday import predict_all
from src.intraday.backtest.replay_engine import replay
from src.intraday.backtest.metrics import compute_metrics, print_metrics_report
from src.intraday.execution.slippage_model import MZS, compute_round_trip_cost
from src.intraday.execution.risk_intraday import RiskConfig


_OUT_DIR = os.path.join(_ROOT, "artifacts", "intraday")
os.makedirs(_OUT_DIR, exist_ok=True)
INITIAL_CAPITAL = 10_000


def main():
    print("=" * 72)
    print("  AGROCAST INTRADAY — Pilot Run (Fase 0)")
    print(f"  Run: {datetime.now().isoformat()}")
    print(f"  Instrumento: MZS  |  Capital: ${INITIAL_CAPITAL:,}")
    cost = compute_round_trip_cost(MZS, 1)
    print(f"  Costo round-trip MZS 1c: ${cost['total_usd']} ({cost['total_ticks']} ticks)")
    print("=" * 72)

    # ── 1. Train ────────────────────────────────────────────────────────
    print("\n[1/3] Entrenando modelo intradía ...")
    train_main()

    # ── 2. Predict ──────────────────────────────────────────────────────
    print("\n[2/3] Generando señales ...")
    signals = predict_all(interval="5m")

    # ── 3. Backtest ─────────────────────────────────────────────────────
    print("\n[3/3] Backtest replay ...")
    bars = fetch_intraday_bars(interval="5m")
    trades, summary = replay(
        signals_df=signals,
        bars_df=bars,
        initial_capital=INITIAL_CAPITAL,
        contract=MZS,
        horizon_bars=12,
        cfg=RiskConfig(),
    )
    print(f"\n  Trades ejecutados: {summary['n_trades']}")
    print(f"  Capital final: ${summary['final_capital']:,.2f} "
          f"({summary['return_pct']:+.2f}%)")

    # Guardar trades
    trades_path = os.path.join(_OUT_DIR, "intraday_backtest.csv")
    trades.to_csv(trades_path, index=False)
    print(f"  Trades log: {trades_path}")

    # ── 4. Métricas + Gate ──────────────────────────────────────────────
    print()
    metrics = compute_metrics(trades, initial_capital=INITIAL_CAPITAL)
    print_metrics_report(metrics)

    metrics_path = os.path.join(_OUT_DIR, "intraday_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "run_at": datetime.now().isoformat(),
            "initial_capital": INITIAL_CAPITAL,
            "contract": "MZS",
            "summary": summary,
            "metrics": metrics,
        }, f, indent=2, default=str)
    print(f"\n  Metrics: {metrics_path}")
    print(f"\n  GATE Fase 0: {metrics.get('fase0_gate', '?')}")
    if metrics.get("fase0_gate") == "PASS":
        print("    → Pasar a Fase 1 (CME DataMine + paper trading 30d)")
    else:
        print("    → Iterar features/hyperparámetros antes de Fase 1")


if __name__ == "__main__":
    main()
