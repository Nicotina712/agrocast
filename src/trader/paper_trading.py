"""
src/trader/paper_trading.py
Sistema de paper trading para el Módulo Trader de AgroCast PRO.

Flujo:
  1. Cuando el pipeline detecta una señal BUY o SELL nueva, llama a log_trade_entry()
  2. En cada ciclo del pipeline, check_and_close_trades() evalúa si:
       - El precio tocó el Stop Loss  → cierra con pérdida
       - El precio tocó el Take Profit → cierra con ganancia
       - Pasaron 7 días              → cierra al precio de mercado (timeout)
  3. get_paper_portfolio() devuelve el estado completo para el dashboard

Almacenamiento: data/paper_trades.csv
Cada fila = 1 operación (entrada + cierre cuando corresponde)

Columnas:
  id, signal, entry_date, entry_price, stop_loss, take_profit,
  atr, n_contracts, capital_at_entry, risk_usd,
  status (open/closed_sl/closed_tp/closed_timeout),
  exit_date, exit_price, pnl_usc_bu, pnl_usd, pnl_pct, hit
"""

import os
import uuid
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRADES_PATH   = os.path.join(_PROJECT_ROOT, "data", "paper_trades.csv")
_LAST_SIG_PATH = os.path.join(_PROJECT_ROOT, "data", "paper_last_signal.json")

TRADE_HORIZON_DAYS = 14     # días máximos antes de cerrar por timeout
ZS_CONTRACT_SIZE   = 5000   # bushels por contrato

# ── Realistic trading costs ──────────────────────────────────────────
# Slippage: market impact at entry and exit (cents/bushel per side)
SLIPPAGE_CST_PER_SIDE = 0.25   # conservative for liquid front-month ZS
# Commission: round-trip per contract (typical retail futures broker)
COMMISSION_PER_CONTRACT_RT = 5.00   # USD per round-trip

_COLUMNS = [
    "id", "signal", "entry_date", "entry_price", "stop_loss", "take_profit",
    "atr", "n_contracts", "capital_at_entry", "risk_usd",
    "status", "exit_date", "exit_price", "pnl_usc_bu", "pnl_usd", "pnl_pct", "hit",
]


# ── Persistencia ──────────────────────────────────────────────────────

def _load_trades() -> pd.DataFrame:
    if os.path.exists(_TRADES_PATH):
        try:
            df = pd.read_csv(_TRADES_PATH, parse_dates=["entry_date", "exit_date"])
            for c in _COLUMNS:
                if c not in df.columns:
                    df[c] = None
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=_COLUMNS)


def _save_trades(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(_TRADES_PATH), exist_ok=True)
    df.to_csv(_TRADES_PATH, index=False)


def _load_last_signal() -> dict:
    import json
    if os.path.exists(_LAST_SIG_PATH):
        try:
            return json.loads(open(_LAST_SIG_PATH).read())
        except Exception:
            pass
    return {}


def _save_last_signal(data: dict) -> None:
    import json
    os.makedirs(os.path.dirname(_LAST_SIG_PATH), exist_ok=True)
    open(_LAST_SIG_PATH, "w").write(json.dumps(data, indent=2))


# ── Core ──────────────────────────────────────────────────────────────

def log_trade_entry(
    signal: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    atr: float,
    n_contracts: int,
    capital: float,
    risk_usd: float,
) -> dict | None:
    """
    Registra una nueva entrada de paper trade.
    Solo registra si la señal cambió desde el último trade (evita duplicados).
    Retorna el trade registrado o None si se ignoró.
    """
    if signal == "HOLD":
        return None

    last = _load_last_signal()
    today_str = str(date.today())

    # Capa 1: máximo 1 trade por día, sin importar dirección.
    # Evita el caso "BUY a las 9, SELL a las 17" cuando la probabilidad
    # oscila cerca de los umbrales y el modelo está en zona gris.
    if last.get("date") == today_str:
        return None

    # Capa 2: si hay un trade abierto opuesto, no abrir contrario hasta cerrarlo.
    # La gestión de salida (SL/TP/timeout) decide cuándo cerrar; no se invierte
    # la posición desde acá.
    df_open = _load_trades()
    if not df_open.empty:
        opens = df_open[df_open["status"] == "open"]
        if not opens.empty:
            existing = set(opens["signal"].tolist())
            if signal == "BUY" and "SELL" in existing:
                return None
            if signal == "SELL" and "BUY" in existing:
                return None
            # Mismo lado ya abierto → no apilar
            if signal in existing:
                return None

    trade = {
        "id":              str(uuid.uuid4())[:8],
        "signal":          signal,
        "entry_date":      today_str,
        "entry_price":     round(entry_price, 2),
        "stop_loss":       round(stop_loss, 2),
        "take_profit":     round(take_profit, 2),
        "atr":             round(atr, 2),
        "n_contracts":     int(n_contracts),
        "capital_at_entry": round(capital, 2),
        "risk_usd":        round(risk_usd, 2),
        "status":          "open",
        "exit_date":       None,
        "exit_price":      None,
        "pnl_usc_bu":      None,
        "pnl_usd":         None,
        "pnl_pct":         None,
        "hit":             None,
    }

    df = _load_trades()
    df = pd.concat([df, pd.DataFrame([trade])], ignore_index=True)
    _save_trades(df)
    _save_last_signal({"signal": signal, "date": today_str})

    print(f"[PaperTrading] Nuevo trade: {signal} @ {entry_price} | SL:{stop_loss} TP:{take_profit}")
    return trade


def check_and_close_trades() -> list:
    """
    Evalúa los trades abiertos contra el precio actual.
    Cierra los que tocaron SL, TP o superaron el horizonte de 7 días.
    Retorna lista de trades cerrados en este ciclo.
    """
    df = _load_trades()
    if df.empty:
        return []

    open_trades = df[df["status"] == "open"].copy()
    if open_trades.empty:
        return []

    # Precio actual
    try:
        mkt = pd.read_csv(os.path.join(_PROJECT_ROOT, "data", "raw_market.csv"),
                          parse_dates=["Date"])
        mkt = mkt.sort_values("Date").reset_index(drop=True)
        current_price = float(mkt["Soybeans"].iloc[-1])
        current_date  = mkt["Date"].iloc[-1].date()

        # Full price history for SL/TP evaluation (not just last 10 rows)
        recent_prices = mkt[["Date", "Soybeans"]].copy()
    except Exception as e:
        print(f"[PaperTrading] Error leyendo precios: {e}")
        return []

    closed = []

    for idx, row in open_trades.iterrows():
        entry_date = pd.to_datetime(row["entry_date"]).date()
        days_open  = (current_date - entry_date).days
        signal     = row["signal"]
        sl         = float(row["stop_loss"])
        tp         = float(row["take_profit"])
        entry_px   = float(row["entry_price"])
        n_cont     = int(row["n_contracts"])

        exit_price  = None
        exit_status = None

        # Evaluar cada día desde la entrada para detectar SL/TP
        relevant = recent_prices[recent_prices["Date"].dt.date >= entry_date]
        for _, price_row in relevant.iterrows():
            px = float(price_row["Soybeans"])
            if signal == "BUY":
                if px <= sl:
                    exit_price, exit_status = px, "closed_sl"
                    break
                elif px >= tp:
                    exit_price, exit_status = px, "closed_tp"
                    break
            elif signal == "SELL":
                if px >= sl:
                    exit_price, exit_status = px, "closed_sl"
                    break
                elif px <= tp:
                    exit_price, exit_status = px, "closed_tp"
                    break

        # Timeout: 7 días sin tocar SL ni TP
        if exit_status is None and days_open >= TRADE_HORIZON_DAYS:
            exit_price  = current_price
            exit_status = "closed_timeout"

        if exit_status is not None:
            # Calcular P&L with realistic costs
            if signal == "BUY":
                pnl_usc = exit_price - entry_px
            else:
                pnl_usc = entry_px - exit_price

            # Deduct slippage (both sides: entry + exit)
            pnl_usc -= 2 * SLIPPAGE_CST_PER_SIDE

            pnl_usd = pnl_usc * n_cont * ZS_CONTRACT_SIZE / 100
            # Deduct round-trip commission per contract
            pnl_usd -= COMMISSION_PER_CONTRACT_RT * n_cont
            pnl_pct = pnl_usd / float(row["capital_at_entry"]) * 100 if row["capital_at_entry"] else 0
            hit     = pnl_usd > 0

            df.at[idx, "status"]     = exit_status
            df.at[idx, "exit_date"]  = str(current_date)
            df.at[idx, "exit_price"] = round(exit_price, 2)
            df.at[idx, "pnl_usc_bu"] = round(pnl_usc, 2)
            df.at[idx, "pnl_usd"]    = round(pnl_usd, 2)
            df.at[idx, "pnl_pct"]    = round(pnl_pct, 2)
            df.at[idx, "hit"]        = bool(hit)

            emoji = "✅" if hit else "❌"
            print(f"[PaperTrading] {emoji} Cerrado {exit_status}: {signal} "
                  f"P&L={pnl_usd:+.0f} USD ({pnl_pct:+.1f}%)")
            closed.append(df.loc[idx].to_dict())

    if closed:
        _save_trades(df)

    return closed


def get_paper_portfolio(capital: float = 10000) -> dict:
    """
    Retorna el estado completo del portfolio de paper trading.
    """
    df = _load_trades()

    if df.empty:
        return {
            "open_trades":    [],
            "closed_trades":  [],
            "summary":        _empty_summary(capital),
            "equity_curve":   [],
        }

    closed_df = df[df["status"] != "open"].copy()
    open_df   = df[df["status"] == "open"].copy()

    # Summary de trades cerrados
    if not closed_df.empty:
        closed_df["pnl_usd"] = pd.to_numeric(closed_df["pnl_usd"], errors="coerce").fillna(0)
        closed_df["hit"]     = closed_df["hit"].astype(bool)

        total_pnl   = float(closed_df["pnl_usd"].sum())
        n_closed    = len(closed_df)
        n_win       = int(closed_df["hit"].sum())
        win_rate    = round(n_win / n_closed * 100, 1) if n_closed > 0 else 0
        avg_win     = float(closed_df[closed_df["hit"]]["pnl_usd"].mean()) if n_win > 0 else 0
        n_loss      = n_closed - n_win
        avg_loss    = float(closed_df[~closed_df["hit"]]["pnl_usd"].mean()) if n_loss > 0 else 0
        expectancy  = round((win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss), 2)

        # Racha máxima pérdidas
        hits = list(closed_df.sort_values("exit_date")["hit"])
        max_consec_loss = cur = 0
        for h in hits:
            cur = cur + 1 if not h else 0
            max_consec_loss = max(max_consec_loss, cur)

        # Equity curve + drawdown
        equity      = capital
        peak_equity = capital
        max_dd      = 0.0
        curve       = [{"date": "inicio", "equity": round(capital, 2)}]
        trade_rets  = []

        for _, r in closed_df.sort_values("exit_date").iterrows():
            pnl     = float(r["pnl_usd"])
            equity += pnl
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            if dd > max_dd:
                max_dd = dd
            curve.append({"date": str(r["exit_date"])[:10], "equity": round(equity, 2)})
            trade_rets.append(pnl / capital)

        # Profit factor
        gross_win  = float(closed_df[closed_df["hit"]]["pnl_usd"].sum()) if n_win > 0 else 0
        gross_loss = abs(float(closed_df[~closed_df["hit"]]["pnl_usd"].sum())) if n_loss > 0 else 0
        profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else None

        # ── Track-record metadata para uso comercial / marketing ─────
        # inception_date: primera entrada registrada
        # days_alive: días corridos desde inception
        # is_meaningful: bandera para gating UI (n>=10 trades para ser estadísticamente válido)
        try:
            # entry_date puede venir mixto ("YYYY-MM-DD" y "YYYY-MM-DD HH:MM:SS")
            entry_dt = pd.to_datetime(df["entry_date"], errors="coerce", format="mixed").dropna()
            inception = entry_dt.min() if not entry_dt.empty else None
            if inception is not None and pd.notna(inception):
                inception_str = inception.date().isoformat()
                days_alive = (pd.Timestamp.now().normalize() - inception.normalize()).days
            else:
                inception_str, days_alive = None, 0
        except Exception as _e:
            print(f"[paper_trading] inception parse fail: {_e}")
            inception_str, days_alive = None, 0

        # Sharpe anualizado using actual trade frequency
        if len(trade_rets) >= 3:
            tr_arr = np.array(trade_rets)
            # Estimate annualization factor from actual trading period
            _trades_per_year = 52  # default fallback
            if days_alive and days_alive > 0 and n_closed > 1:
                _trades_per_year = max(1, n_closed / (days_alive / 365.25))
            sharpe = float(tr_arr.mean() / (tr_arr.std() + 1e-8) * np.sqrt(_trades_per_year))
        else:
            sharpe = None

        # ── Buy & Hold benchmark sobre el mismo período ──────────────
        # Para que el track record tenga contexto: ¿la estrategia gana
        # contra "comprar y mantener"?
        bh_return_pct = None
        try:
            entry_prices = pd.to_numeric(df["entry_price"], errors="coerce").dropna()
            sorted_df = df.dropna(subset=["entry_price"]).sort_values("entry_date")
            if len(sorted_df) >= 2:
                first_px = float(sorted_df["entry_price"].iloc[0])
                last_px  = float(sorted_df["entry_price"].iloc[-1])
                if first_px > 0:
                    bh_return_pct = round((last_px - first_px) / first_px * 100, 2)
        except Exception:
            bh_return_pct = None

        summary = {
            "inception_date":    inception_str,
            "days_alive":        int(days_alive),
            "is_meaningful":     n_closed >= 10,        # gate estadístico (no marketing)
            "needs_n_more":      max(0, 10 - n_closed),
            "bh_return_pct":     bh_return_pct,
            "alpha_vs_bh_pct":   round(round(total_pnl / capital * 100, 2) - bh_return_pct, 2) if bh_return_pct is not None else None,
            "total_pnl_usd":     round(total_pnl, 2),
            "total_return_pct":  round(total_pnl / capital * 100, 2),
            "n_closed":          n_closed,
            "n_open":            len(open_df),
            "win_rate":          win_rate,
            "n_win":             n_win,
            "n_loss":            n_loss,
            "avg_win_usd":       round(avg_win, 2),
            "avg_loss_usd":      round(avg_loss, 2),
            "expectancy_usd":    expectancy,
            "profit_factor":     profit_factor,
            "max_drawdown_pct":  round(max_dd * 100, 2),
            "sharpe_annualized": round(sharpe, 2) if sharpe else None,
            "max_consec_losses": max_consec_loss,
            "current_equity":    round(capital + total_pnl, 2),
            "closed_by_sl":      int((closed_df["status"] == "closed_sl").sum()),
            "closed_by_tp":      int((closed_df["status"] == "closed_tp").sum()),
            "closed_by_timeout": int((closed_df["status"] == "closed_timeout").sum()),
        }
    else:
        summary = _empty_summary(capital)
        curve   = []

    def trade_to_dict(r):
        d = r.to_dict()
        for k, v in d.items():
            if pd.isna(v) if not isinstance(v, (list, dict)) else False:
                d[k] = None
        return d

    return {
        "open_trades":   [trade_to_dict(r) for _, r in open_df.iterrows()],
        "closed_trades": [trade_to_dict(r) for _, r in closed_df.sort_values("exit_date", ascending=False).head(20).iterrows()],
        "summary":       summary,
        "equity_curve":  curve,
    }


def _empty_summary(capital: float) -> dict:
    return {
        "total_pnl_usd": 0, "total_return_pct": 0,
        "n_closed": 0, "n_open": 0, "win_rate": 0,
        "n_win": 0, "n_loss": 0,
        "avg_win_usd": 0, "avg_loss_usd": 0, "expectancy_usd": 0,
        "profit_factor": None, "max_drawdown_pct": 0,
        "sharpe_annualized": None, "max_consec_losses": 0,
        "current_equity": capital,
        "closed_by_sl": 0, "closed_by_tp": 0, "closed_by_timeout": 0,
    }


def _get_ie_signal() -> dict | None:
    """
    Lee el veredicto del Intelligence Engine como fuente primaria de señal.
    Retorna dict con {signal, confidence} o None si no hay veredicto fresco.
    """
    import json
    ie_path = os.path.join(_PROJECT_ROOT, "data", "intelligence_engine_verdict.json")
    if not os.path.exists(ie_path):
        return None
    try:
        with open(ie_path, "r", encoding="utf-8") as f:
            ie = json.load(f)
        verdict_data = ie.get("verdict", {})
        verdict = verdict_data.get("verdict", "HOLD")
        confidence = verdict_data.get("confidence", 0)

        # Check freshness: IE debe tener <48h para usarse
        ts = ie.get("timestamp", "")
        if ts:
            ie_dt = datetime.fromisoformat(ts)
            age_hours = (datetime.now() - ie_dt).total_seconds() / 3600
            if age_hours > 48:
                print(f"[PaperTrading] IE verdict too old ({int(age_hours)}h), ignoring")
                return None

        # Only trade if confidence > 0.5 (IE is reasonably sure)
        if confidence < 0.50:
            return None

        # Normalize STRONG_ variants
        if verdict in ("STRONG_BUY", "BUY"):
            return {"signal": "BUY", "confidence": confidence}
        elif verdict in ("STRONG_SELL", "SELL"):
            return {"signal": "SELL", "confidence": confidence}
        return None
    except Exception as e:
        print(f"[PaperTrading] Error reading IE verdict: {e}")
        return None


def run_paper_trading_cycle(capital: float = 10000, risk_pct: float = 1.0) -> dict:
    """
    Punto de entrada principal — llamado desde el pipeline en cada ciclo.
    1. Cierra trades vencidos
    2. Evalúa si hay que abrir uno nuevo (usando IE verdict como señal primaria)
    Retorna resumen del ciclo.
    """
    from src.trader.risk_manager import get_current_risk_levels

    # Paso 1: cerrar trades que corresponda
    closed = check_and_close_trades()

    # Paso 2: obtener señal del Intelligence Engine (fuente primaria)
    ie_sig = _get_ie_signal()

    # Paso 3: calcular niveles de riesgo (ATR, SL, TP) usando la señal del IE
    if ie_sig:
        signal = ie_sig["signal"]
        # Usamos get_current_risk_levels pero overrideamos la señal con la del IE
        levels = get_current_risk_levels(capital_usd=capital, risk_pct=risk_pct)
        # Recalcular con la señal correcta del IE
        from src.trader.risk_manager import compute_risk_levels, compute_atr
        import pandas as pd
        try:
            mkt = pd.read_csv(os.path.join(_PROJECT_ROOT, "data", "raw_market.csv"),
                              parse_dates=["Date"])
            mkt = mkt.sort_values("Date").reset_index(drop=True)
            entry_price = float(mkt["Soybeans"].iloc[-1])
            atr = float(compute_atr(mkt).iloc[-1])
            levels = compute_risk_levels(entry_price, atr, signal, capital, risk_pct)
            levels["signal"] = signal
        except Exception as e:
            print(f"[PaperTrading] Error computing risk levels: {e}")
            levels = {"signal": "HOLD", "stop_loss": None}
    else:
        # Fallback: usar señal del modelo ML (comportamiento anterior)
        levels = get_current_risk_levels(capital_usd=capital, risk_pct=risk_pct)
        signal = levels.get("signal", "HOLD")

    new_trade = None
    if signal in ("BUY", "SELL") and levels.get("stop_loss") is not None:
        new_trade = log_trade_entry(
            signal       = signal,
            entry_price  = levels["entry"],
            stop_loss    = levels["stop_loss"],
            take_profit  = levels["take_profit"],
            atr          = levels["atr"],
            n_contracts  = levels["n_contracts"],
            capital      = capital,
            risk_usd     = levels.get("risk_usd_total", 0),
        )

    return {
        "closed_this_cycle": len(closed),
        "new_trade_opened":  new_trade is not None,
        "new_trade":         new_trade,
        "signal":            signal,
        "signal_source":     "intelligence_engine" if ie_sig else "ml_model",
    }
