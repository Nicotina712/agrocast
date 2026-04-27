"""
src/trader/risk_manager.py
Módulo Trader — Gestión de riesgo para futuros ZS.

Calcula Stop Loss y Take Profit basados en ATR (Average True Range):
  - ATR(14): media del True Range de los últimos 14 días
  - Stop Loss  = entry - ATR_MULT_SL × ATR  (1.5× por defecto)
  - Take Profit = entry + ATR_MULT_TP × ATR  (2.5× por defecto)
  - Risk/Reward = ATR_MULT_TP / ATR_MULT_SL  (1.67:1 por defecto)

También calcula el tamaño de posición óptimo dado un capital y % de riesgo:
  - Un contrato ZS = 5000 bushels
  - Valor por punto = 5000 × 0.0001 USc/bu = USD 0.50
  - Risk por contrato = (entry - SL) × 5000 / 100  (en USD)
"""

import os
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATR_PERIOD  = 14
ATR_MULT_SL = 1.5    # Stop loss en ATR
ATR_MULT_TP = 2.5    # Take profit en ATR

# Tamaño de 1 contrato ZS en bushels
ZS_CONTRACT_SIZE = 5000  # bushels


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Calcula ATR clásico de Wilder. Usa OHLC real si está disponible."""
    close = df["Soybeans"]
    if "Soybeans_High" in df.columns and "Soybeans_Low" in df.columns:
        high = df["Soybeans_High"].astype(float)
        low  = df["Soybeans_Low"].astype(float)
        # Si hay NaN (días sin OHLC del cache viejo), fallback sintético por fila
        synth_h = close * 1.005
        synth_l = close * 0.995
        high = high.where(high.notna(), synth_h)
        low  = low.where(low.notna(),  synth_l)
    else:
        high = close * 1.005
        low  = close * 0.995
    prev_close = close.shift(1)

    tr = pd.DataFrame({
        "hl":  high - low,
        "hpc": (high - prev_close).abs(),
        "lpc": (low  - prev_close).abs(),
    }).max(axis=1)

    atr = tr.ewm(span=period, min_periods=period // 2).mean()
    return atr


def _vol_target_scalar(atr: float, entry_price: float,
                        target_annual_vol: float = 0.15) -> float:
    """
    Escala el sizing para apuntar a target_annual_vol (default 15%).
    ATR diario aproxima la volatilidad diaria absoluta.
    Si la vol realizada > target → reduce sizing; si < target → aumenta.
    """
    if entry_price <= 0 or atr <= 0:
        return 1.0
    daily_vol_pct = atr / entry_price            # ATR como % del precio
    realized_annual = daily_vol_pct * (252 ** 0.5)
    if realized_annual <= 0:
        return 1.0
    scalar = target_annual_vol / realized_annual
    # Acotamos para evitar sizing extremo
    return float(max(0.25, min(2.5, scalar)))


def compute_risk_levels(
    entry_price: float,
    atr: float,
    signal: str,
    capital_usd: float = 10000,
    risk_pct: float = 1.0,
    vol_target: bool = True,
    target_annual_vol: float = 0.15,
    expected_vol: float | None = None,
) -> dict:
    """
    Cuando expected_vol (anualizada) está disponible (head del modelo),
    overridea la vol implícita derivada de ATR para el sizing y stops:
      - daily_vol = expected_vol / sqrt(252)
      - atr_adj   = max(atr, daily_vol * entry_price)
    Esto sube los stops cuando el modelo anticipa más vol — protege contra
    estrucutra cambiante (event windows / regime shifts).
    """
    if expected_vol is not None and expected_vol > 0 and entry_price > 0:
        daily_vol_abs = (expected_vol / (252 ** 0.5)) * entry_price
        if daily_vol_abs > atr:
            atr = float(daily_vol_abs)
    """
    Calcula SL, TP y tamaño de posición dado un entry price y señal.

    Args:
        entry_price:  precio de entrada en USc/bu
        atr:          ATR(14) en USc/bu
        signal:       "BUY" | "SELL"
        capital_usd:  capital disponible en USD
        risk_pct:     porcentaje del capital a arriesgar por operación (default 1%)

    Returns:
        dict con stop_loss, take_profit, risk_reward, n_contracts, risk_usd
    """
    if signal == "BUY":
        stop_loss   = entry_price - ATR_MULT_SL * atr
        take_profit = entry_price + ATR_MULT_TP * atr
        direction   = "LONG"
    elif signal == "SELL":
        stop_loss   = entry_price + ATR_MULT_SL * atr
        take_profit = entry_price - ATR_MULT_TP * atr
        direction   = "SHORT"
    else:
        return {
            "direction":    "FLAT",
            "entry":        round(entry_price, 2),
            "stop_loss":    None,
            "take_profit":  None,
            "risk_reward":  None,
            "n_contracts":  0,
            "risk_usd":     0,
            "atr":          round(atr, 2),
        }

    risk_per_bu  = abs(entry_price - stop_loss)        # USc/bu
    risk_per_contract = risk_per_bu * ZS_CONTRACT_SIZE / 100  # USD (USc → USD)
    risk_usd     = capital_usd * risk_pct / 100

    # Vol targeting: ajusta el risk_usd efectivo según vol realizada
    vt_scalar = _vol_target_scalar(atr, entry_price, target_annual_vol) if vol_target else 1.0
    risk_usd_effective = risk_usd * vt_scalar
    n_contracts  = max(1, int(risk_usd_effective / risk_per_contract)) if risk_per_contract > 0 else 1

    rr = round(abs(take_profit - entry_price) / abs(entry_price - stop_loss), 2)

    return {
        "direction":        direction,
        "entry":            round(entry_price, 2),
        "stop_loss":        round(stop_loss, 2),
        "take_profit":      round(take_profit, 2),
        "risk_reward":      rr,
        "atr":              round(atr, 2),
        "risk_per_bu":      round(risk_per_bu, 2),
        "risk_per_contract_usd": round(risk_per_contract, 2),
        "n_contracts":      n_contracts,
        "risk_usd_total":   round(n_contracts * risk_per_contract, 2),
        "potential_profit_usd": round(n_contracts * abs(take_profit - entry_price) * ZS_CONTRACT_SIZE / 100, 2),
        "vol_target_scalar": round(vt_scalar, 3),
        "annual_vol_estimate": round(atr / entry_price * (252 ** 0.5), 3) if entry_price > 0 else None,
    }


def get_current_risk_levels(capital_usd: float = 10000, risk_pct: float = 1.0) -> dict:
    """
    Lee el precio actual y los signals.csv para generar los niveles de riesgo
    del trade actual.
    """
    try:
        mkt = pd.read_csv(os.path.join(_PROJECT_ROOT, "data", "raw_market.csv"),
                          parse_dates=["Date"])
        mkt = mkt.sort_values("Date").reset_index(drop=True)
        entry_price = float(mkt["Soybeans"].iloc[-1])
        atr_series  = compute_atr(mkt)
        atr         = float(atr_series.iloc[-1])
    except Exception as e:
        return {"error": f"No se pudo leer raw_market.csv: {e}"}

    try:
        sig_df  = pd.read_csv(os.path.join(_PROJECT_ROOT, "artifacts", "signals.csv"))
        signal  = str(sig_df["signal"].iloc[-1])
        conf    = float(sig_df["confidence"].iloc[-1])
        exp_vol = float(sig_df["expected_vol"].iloc[-1]) if "expected_vol" in sig_df.columns else None
        if exp_vol is not None and (np.isnan(exp_vol) or exp_vol <= 0):
            exp_vol = None
    except Exception:
        signal, conf, exp_vol = "HOLD", 0.0, None

    levels = compute_risk_levels(entry_price, atr, signal, capital_usd, risk_pct,
                                  expected_vol=exp_vol)
    if exp_vol is not None:
        levels["expected_vol_annual"] = round(exp_vol, 4)
    levels["signal"]     = signal
    levels["confidence"] = round(conf, 3)
    return levels
