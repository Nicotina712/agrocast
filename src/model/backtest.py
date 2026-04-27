"""
src/model/backtest.py
Backtesting walk-forward para las señales de AgroCast.

Soporta tanto el clasificador (XGBClassifier, model_type="classifier")
como el regresor legacy. El clasificador usa predict_proba y umbrales
fijos (buy_thresh/sell_thresh). El regresor usa percentiles.

Simula: BUY → long, SELL → short, HOLD → cash.
Compara contra buy-and-hold como benchmark.
"""

import os

import joblib
import numpy as np
import pandas as pd

from src.model.predict_returns import generate_signal

NON_FEATURE_COLS = {"Date", "Soybeans",
                    "ret_1d_fwd", "ret_14d_fwd", "ret_14d_fwd", "ret_30d_fwd"}
TRADE_STEP      = 14
HORIZON_COL     = "ret_14d_fwd"
INITIAL_CAPITAL = 10_000


def _to_python(obj):
    """Convierte recursivamente tipos numpy a Python nativo para JSON."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_to_python(v) for v in obj.tolist()]
    return obj


def _predict_window(df_window: pd.DataFrame, model, feature_cols: list,
                    model_type: str, buy_thresh: float, sell_thresh: float) -> pd.DataFrame:
    """
    Devuelve df_window con columnas 'pred_score' y 'signal' añadidas.
    pred_score: P(sube) para clasificador, retorno predicho para regresor.
    """
    X = df_window.reindex(columns=feature_cols, fill_value=0).fillna(0).replace([np.inf, -np.inf], 0)

    window = df_window.copy()

    if model_type == "classifier":
        probs = model.predict_proba(X)[:, 1].astype(float)
        window["pred_score"] = probs
        window["signal"]     = "HOLD"
        window.loc[probs > buy_thresh,  "signal"] = "BUY"
        window.loc[probs < sell_thresh, "signal"] = "SELL"

    else:
        # Regresor legacy: percentiles por ventana
        preds = model.predict(X).astype(float)
        window["pred_score"] = preds
        low_t  = float(pd.Series(preds).quantile(0.33))
        high_t = float(pd.Series(preds).quantile(0.67))

        if abs(high_t - low_t) < 1e-6:
            n     = len(window)
            ranks = pd.Series(preds).rank(method="first")
            window["signal"] = "HOLD"
            window.loc[ranks <= n // 3,      "signal"] = "SELL"
            window.loc[ranks > n - (n // 3), "signal"] = "BUY"
        else:
            window["signal"] = window["pred_score"].apply(
                lambda r: generate_signal(r, low_t, high_t)
            )

    return window


def run_backtest(features_path: str, model_dir: str) -> dict:
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"features.csv no encontrado: {features_path}")

    df = pd.read_csv(features_path)
    df = df.dropna(subset=["ret_14d_fwd", "Soybeans"]).reset_index(drop=True)

    if len(df) < 100:
        raise ValueError("Datos insuficientes para backtest (mínimo 100 filas)")

    model_path = os.path.join(model_dir, "returns_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    saved        = joblib.load(model_path)
    model        = saved["model"]
    feature_cols = saved["feature_cols"]
    model_type   = saved.get("model_type", "regressor")
    buy_thresh   = saved.get("buy_thresh",  0.58)
    sell_thresh  = saved.get("sell_thresh", 0.42)

    print(f"📊 Backtest usando modelo tipo: {model_type}")

    # ── Walk-forward: último 30% en 3 ventanas, con embargo (Fix #3) ──
    # 18 días = horizonte (14) + buffer evento (4) entre train y test
    EMBARGO_DAYS = 18
    train_end   = int(len(df) * 0.70)
    if "Date" in df.columns:
        df_dates = pd.to_datetime(df["Date"])
        cut_date = df_dates.iloc[train_end - 1]
        embargo_end = cut_date + pd.Timedelta(days=EMBARGO_DAYS)
        test = df[df_dates > embargo_end].copy().reset_index(drop=True)
        print(f"   [Embargo] {EMBARGO_DAYS}d entre train ({cut_date.date()}) "
              f"y test (start={test['Date'].iloc[0] if len(test) else 'N/A'})")
    else:
        test = df.iloc[train_end:].copy().reset_index(drop=True)
    window_size = len(test) // 3

    capital    = float(INITIAL_CAPITAL)
    bh_capital = float(INITIAL_CAPITAL)
    capital_hist: list[dict] = []
    bh_hist:      list[dict] = []
    trades:       list[dict] = []

    for w in range(3):
        w_start = w * window_size
        w_end   = w_start + window_size if w < 2 else len(test)
        raw_win = test.iloc[w_start:w_end].copy()

        window = _predict_window(raw_win, model, feature_cols,
                                 model_type, buy_thresh, sell_thresh)

        # Simular cada TRADE_STEP días
        try:
            from src.trader.costs import estimate_round_trip_cost_pct
        except Exception:
            estimate_round_trip_cost_pct = lambda m=1.0: 0.0005

        try:
            from src.trader.risk_manager import compute_atr, ATR_MULT_SL
            atr_series = compute_atr(window) if "Soybeans" in window.columns else None
        except Exception:
            atr_series = None
            ATR_MULT_SL = 1.5

        for i in range(0, len(window) - TRADE_STEP, TRADE_STEP):
            row        = window.iloc[i]
            signal     = row["signal"]
            actual_ret = float(row["ret_14d_fwd"])
            date       = str(row["Date"])[:10]

            # ── No-trade window ±2d WASDE/Acreage/Stocks ─────────────
            # El backtest histórico mostraba WR=0% (n=3) en ventanas de
            # eventos USDA: el modelo no anticipa la sorpresa y el spread
            # se ensancha. Forzamos HOLD dentro de la ventana.
            if "in_event_window" in row.index and int(row.get("in_event_window", 0)) == 1:
                if signal in ("BUY", "SELL"):
                    signal = "HOLD"  # localmente, para esta iteración

            # Cost model variable por evento
            cost_mult = float(row.get("event_cost_mult", 1.0)) if "event_cost_mult" in row else 1.0
            cost_pct  = estimate_round_trip_cost_pct(cost_mult)

            # ── Fix #15: simulación de ATR-stop intra-trade ────────
            # Si dentro de la ventana 14d el extremo intradiario sobrepasa
            # el stop ATR, el trade exits temprano con pérdida acotada.
            stop_hit = False
            if signal in ("BUY", "SELL") and atr_series is not None and i + TRADE_STEP < len(window):
                entry_px = float(row["Soybeans"]) if "Soybeans" in row else None
                atr_now  = float(atr_series.iloc[i]) if i < len(atr_series) else None
                if entry_px and atr_now and atr_now > 0:
                    win = window.iloc[i + 1: i + 1 + TRADE_STEP]
                    if "Soybeans_Low" in win.columns and "Soybeans_High" in win.columns:
                        if signal == "BUY":
                            stop_px = entry_px - ATR_MULT_SL * atr_now
                            if (win["Soybeans_Low"] <= stop_px).any():
                                stop_hit = True
                                actual_ret = (stop_px - entry_px) / entry_px
                        else:  # SELL
                            stop_px = entry_px + ATR_MULT_SL * atr_now
                            if (win["Soybeans_High"] >= stop_px).any():
                                stop_hit = True
                                actual_ret = (entry_px - stop_px) / entry_px

            gross = actual_ret if signal == "BUY" else (-actual_ret if signal == "SELL" else 0.0)
            # Si el stop ya cerró en la dirección correcta (fue stop_hit), gross = -|stop_dist|/entry
            if stop_hit:
                gross = actual_ret  # ya está en signo negativo (stop loss)
            pnl   = gross - cost_pct if signal != "HOLD" else 0.0

            capital    *= (1 + pnl)
            bh_capital *= (1 + actual_ret)

            capital_hist.append({"Date": date, "value": round(capital,    2)})
            bh_hist.append(     {"Date": date, "value": round(bh_capital, 2)})

            if signal != "HOLD":
                pred_display = round((float(row["pred_score"]) - 0.5) * 100, 2) \
                    if model_type == "classifier" \
                    else round(float(row["pred_score"]) * 100, 2)
                trades.append({
                    "Date":          date,
                    "signal":        signal,
                    "pred_return":   pred_display,
                    "actual_return": round(actual_ret * 100, 2),
                    "cost_pct":      round(cost_pct * 100, 3),
                    "net_pnl_pct":   round(pnl * 100, 2),
                    "cost_mult":     round(cost_mult, 2),
                    "hit":           bool(pnl > 0),
                    "stop_hit":      bool(stop_hit),
                })

    if not capital_hist:
        raise ValueError("Backtest no produjo ninguna operación")

    # ── Métricas ──────────────────────────────────────────────────
    total_return = (capital    - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    bh_return    = (bh_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    alpha        = total_return - bh_return

    n_active = len(trades)
    win_rate = (sum(1 for t in trades if t["hit"]) / n_active * 100) if n_active else 0

    vals   = [INITIAL_CAPITAL] + [c["value"] for c in capital_hist]
    rets   = np.diff(vals) / np.array(vals[:-1])
    sharpe = float(rets.mean() / rets.std() * np.sqrt(26)) if rets.std() > 0 else 0.0

    # Sortino (solo downside deviation)
    neg_rets = rets[rets < 0]
    downside_std = float(neg_rets.std()) if len(neg_rets) > 1 else 1e-9
    sortino = float(rets.mean() / downside_std * np.sqrt(26)) if downside_std > 0 else 0.0

    peak = vals[0]; max_dd = 0.0
    for v in vals:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd: max_dd = dd

    # Calmar = retorno anualizado / max drawdown
    n_years  = len(rets) / 26 if len(rets) > 0 else 1
    ann_ret  = (1 + total_return / 100) ** (1 / max(n_years, 0.1)) - 1
    calmar   = float(ann_ret / max_dd) if max_dd > 0 else 0.0

    # Racha máxima de pérdidas consecutivas
    hits = [t["hit"] for t in trades]
    max_consec_loss = 0
    cur = 0
    for h in hits:
        cur = cur + 1 if not h else 0
        max_consec_loss = max(max_consec_loss, cur)

    print(f"📊 Backtest: ret={total_return:.1f}% | B&H={bh_return:.1f}% | "
          f"α={alpha:.1f}% | Sharpe={sharpe:.2f} | Sortino={sortino:.2f} | "
          f"WinRate={win_rate:.0f}% | Trades={n_active}")

    # ── Split event-days vs normal-days (Fix #9) ─────────────────
    event_days = set()
    if "in_event_window" in test.columns and "Date" in test.columns:
        event_days = set(
            test.loc[test["in_event_window"] == 1, "Date"].astype(str).str[:10].tolist()
        )

    def _bucket(bucket_trades):
        if not bucket_trades:
            return {"n": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0, "median_pnl_pct": 0.0}
        pnls = [t["net_pnl_pct"] for t in bucket_trades]
        wins = sum(1 for t in bucket_trades if t["hit"])
        return {
            "n":              len(bucket_trades),
            "win_rate":       round(wins / len(bucket_trades) * 100, 1),
            "avg_pnl_pct":    round(sum(pnls) / len(pnls), 2),
            "median_pnl_pct": round(float(np.median(pnls)), 2),
        }

    event_trades  = [t for t in trades if t["Date"] in event_days]
    normal_trades = [t for t in trades if t["Date"] not in event_days]
    event_split = {"event": _bucket(event_trades), "normal": _bucket(normal_trades)}
    print(f"📊 Event vs Normal: "
          f"event n={event_split['event']['n']} WR={event_split['event']['win_rate']}% "
          f"pnl={event_split['event']['avg_pnl_pct']}% | "
          f"normal n={event_split['normal']['n']} WR={event_split['normal']['win_rate']}% "
          f"pnl={event_split['normal']['avg_pnl_pct']}%")

    return _to_python({
        "total_return":        round(total_return, 2),
        "bh_return":           round(bh_return,    2),
        "alpha":               round(alpha,         2),
        "win_rate":            round(win_rate,      1),
        "sharpe":              round(sharpe,        2),
        "sortino":             round(sortino,       2),
        "calmar":              round(calmar,        2),
        "max_drawdown":        round(max_dd * 100,  2),
        "max_consec_losses":   max_consec_loss,
        "n_trades":            n_active,
        "model_type":          model_type,
        "test_period": {
            "start": str(test["Date"].iloc[0])[:10],
            "end":   str(test["Date"].iloc[-1])[:10],
        },
        "capital_history":     capital_hist,
        "bh_history":          bh_hist,
        "recent_trades":       trades[-20:],
        "event_split":         event_split,
    })
