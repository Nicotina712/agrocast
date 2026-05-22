"""
QuantAgent V3 Backtest — Price Action + Fundamental Context Integration.

Compares:
  A) V2 baseline (pure price action) — same as backtest_rules_v2.py
  B) V3 enhanced (price action + fundamental overlay)

Fundamental data sources (available historically):
  - CME daily bars (daily OHLCV, OI, volume) -> daily trend/momentum
  - CVOL / Implied Volatility -> vol regime signal
  - WASDE (monthly) -> stocks/supply signal
  - Multi-commodity signals (ZC, ZW correlations)
  - Swing context bridge (daily model bias)
  - Event calendar (WASDE proximity, roll windows)
  - COT weekly positioning (commercial/speculator net)

Timing analysis:
  - Intraday bars (60m): ~15-20 min delay via yfinance
  - CME daily: available at open next day (T+1)
  - CVOL: computed from daily data, available T+0 post-settlement
  - WASDE: monthly (known dates), impact immediate at 11:00 CT release day
  - COT: weekly (Tues release for prior Fri), 3-day lag
  - Multi-commodity: daily, same delay as CME data
  - Event calendar: deterministic (known in advance)

Integration strategy:
  The fundamental context acts as a FILTER + BIAS, not a signal generator.
  - If fundamental bias aligns with price action -> boost confidence
  - If fundamental bias opposes price action -> reduce confidence or veto
  - If high-impact event imminent -> widen stops or skip trade

Usage:
  python -m src.quantagent.backtest_fundamental_v3 [--days 200] [--capital 1000]
"""

import os
import sys
import json
import argparse
import io
from datetime import datetime, timedelta
from copy import deepcopy

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.intraday.data.tick_feed import fetch_intraday_bars
from src.intraday.features.microstructure import build_intraday_features

# Import V2 components (reuse, don't duplicate)
from src.quantagent.backtest_rules_v2 import (
    _detect_trend_v2,
    _detect_setup_v2,
    rule_risk_agent_v2,
    _evaluate_trade_v2,
)

_OUT_DIR = os.path.join(_ROOT, "artifacts", "quantagent")
_DATA_DIR = os.path.join(_ROOT, "data")
_ARTIFACTS_DIR = os.path.join(_ROOT, "artifacts")


# ═══════════════════════════════════════════════════════════════════════
#  FUNDAMENTAL CONTEXT LOADER
# ═══════════════════════════════════════════════════════════════════════

def _load_daily_context() -> pd.DataFrame:
    """Load daily market data for swing context. Prefers raw_market.csv (10yr history)."""
    # Prefer raw_market.csv (2500+ rows, 10yr history) over cme_history.csv (recent only)
    path = os.path.join(_DATA_DIR, "raw_market.csv")
    if not os.path.exists(path):
        path = os.path.join(_DATA_DIR, "cme_history.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["Date"] if "Date" in pd.read_csv(path, nrows=0).columns else [0])
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _load_cot_data() -> pd.DataFrame:
    """Load COT weekly positioning data."""
    path = os.path.join(_DATA_DIR, "cot_soybeans.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        return df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _load_cvol_data() -> pd.DataFrame:
    """Load implied volatility history."""
    path = os.path.join(_DATA_DIR, "cvol_history.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        return df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _load_wasde_data() -> dict:
    """Load WASDE data and known report dates."""
    data = {}
    path = os.path.join(_DATA_DIR, "wasde_official.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data["latest"] = json.load(f)
        except Exception:
            pass

    hist_path = os.path.join(_DATA_DIR, "wasde_history.json")
    if os.path.exists(hist_path):
        try:
            with open(hist_path) as f:
                data["history"] = json.load(f)
        except Exception:
            pass
    return data


def _load_signals_csv() -> pd.DataFrame:
    """Load daily model signals (swing context)."""
    path = os.path.join(_ARTIFACTS_DIR, "signals.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # Handle both 'date' and 'Date' column names
        date_col = "date" if "date" in df.columns else "Date"
        df[date_col] = pd.to_datetime(df[date_col])
        return df.sort_values(date_col).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _load_multi_commodity() -> dict:
    """Load multi-commodity signals."""
    path = os.path.join(_DATA_DIR, "multi_commodity.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_regime() -> dict:
    """Load current regime detection."""
    path = os.path.join(_ARTIFACTS_DIR, "regime.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════
#  FUNDAMENTAL OVERLAY ENGINE
# ═══════════════════════════════════════════════════════════════════════

class FundamentalOverlay:
    """
    Computes a fundamental bias for each trading day.

    Output: a score from -3 to +3 where:
      -3 = strongly bearish fundamentals
       0 = neutral
      +3 = strongly bullish fundamentals

    Plus flags:
      - event_imminent: bool (WASDE within 2 days)
      - vol_regime: str (from CVOL)
      - cot_extreme: bool (positioning at extremes)
    """

    def __init__(self):
        print("[V3] Loading fundamental data...")
        self.daily = _load_daily_context()
        self.cot = _load_cot_data()
        self.cvol = _load_cvol_data()
        self.wasde = _load_wasde_data()
        self.signals = _load_signals_csv()
        self.regime = _load_regime()

        self._daily_features = self._precompute_daily()
        self._cot_features = self._precompute_cot()
        self._cvol_features = self._precompute_cvol()

        n_daily = len(self._daily_features)
        n_cot = len(self._cot_features)
        n_cvol = len(self._cvol_features)
        n_signals = len(self.signals)
        print(f"[V3] Loaded: {n_daily} daily, {n_cot} COT, {n_cvol} CVOL, {n_signals} signals")

    def _precompute_daily(self) -> dict:
        """Compute daily swing indicators: 5/20 SMA cross, 20d momentum."""
        if self.daily.empty:
            return {}

        df = self.daily.copy()
        # Try different column name patterns
        close_col = None
        for col in ["Close", "close", "Settle", "Last", "Soybeans", "front_settle"]:
            if col in df.columns:
                close_col = col
                break
        if close_col is None:
            # If multi-asset, try to find soybeans
            soy_cols = [c for c in df.columns if "soy" in c.lower() or "zs" in c.lower()]
            if soy_cols:
                close_col = soy_cols[0]
            else:
                return {}

        df["sma5"] = df[close_col].rolling(5).mean()
        df["sma20"] = df[close_col].rolling(20).mean()
        df["sma_cross"] = (df["sma5"] - df["sma20"]) / df[close_col]  # normalized
        df["momentum_20d"] = df[close_col].pct_change(20)
        df["momentum_5d"] = df[close_col].pct_change(5)
        df["daily_ret"] = df[close_col].pct_change()
        df["vol_20d"] = df["daily_ret"].rolling(20).std()

        result = {}
        for _, row in df.iterrows():
            d = row["Date"]
            if pd.isna(d):
                continue
            result[d.strftime("%Y-%m-%d")] = {
                "close": float(row[close_col]) if pd.notna(row[close_col]) else None,
                "sma_cross": float(row["sma_cross"]) if pd.notna(row["sma_cross"]) else 0,
                "momentum_20d": float(row["momentum_20d"]) if pd.notna(row["momentum_20d"]) else 0,
                "momentum_5d": float(row["momentum_5d"]) if pd.notna(row["momentum_5d"]) else 0,
                "vol_20d": float(row["vol_20d"]) if pd.notna(row["vol_20d"]) else 0,
            }
        return result

    def _precompute_cot(self) -> dict:
        """Process COT data into trading signals."""
        if self.cot.empty:
            return {}

        result = {}
        df = self.cot.copy()

        # Look for commercial/speculator net columns
        comm_col = None
        spec_col = None
        for col in df.columns:
            cl = col.lower()
            if "commercial" in cl and "net" in cl:
                comm_col = col
            elif ("noncommercial" in cl or "non_commercial" in cl or "speculator" in cl) and "net" in cl:
                spec_col = col
            elif "cot_index" in cl:
                pass  # Use if available

        idx_col = None
        for col in df.columns:
            if "cot_index" in col.lower() or "index" in col.lower():
                idx_col = col
                break

        for _, row in df.iterrows():
            d = row["Date"]
            if pd.isna(d):
                continue
            entry = {}
            if idx_col and pd.notna(row.get(idx_col)):
                cot_idx = float(row[idx_col])
                entry["cot_index"] = cot_idx
                # Extremes: <20 = bearish positioning, >80 = bullish positioning
                entry["cot_extreme_bear"] = cot_idx < 20
                entry["cot_extreme_bull"] = cot_idx > 80
            if comm_col and pd.notna(row.get(comm_col)):
                entry["commercial_net"] = float(row[comm_col])
            if spec_col and pd.notna(row.get(spec_col)):
                entry["speculator_net"] = float(row[spec_col])

            # COT is reported on Tuesday for prior Friday — we make it available the following week
            # For backtest realism, shift by 3 business days
            available_date = d + timedelta(days=3)
            result[available_date.strftime("%Y-%m-%d")] = entry

        return result

    def _precompute_cvol(self) -> dict:
        """Process CVOL data into volatility regime signals."""
        if self.cvol.empty:
            return {}

        result = {}
        df = self.cvol.copy()

        iv_col = None
        for col in df.columns:
            cl = col.lower()
            if "iv" in cl or "cvol" in cl or "implied" in cl or "atm" in cl:
                iv_col = col
                break
        if iv_col is None and len(df.columns) > 1:
            # Try second column (first is Date)
            iv_col = df.columns[1]

        if iv_col is None:
            return {}

        df["iv_sma20"] = df[iv_col].rolling(20).mean()
        df["iv_zscore"] = (df[iv_col] - df["iv_sma20"]) / df[iv_col].rolling(20).std()

        for _, row in df.iterrows():
            d = row["Date"]
            if pd.isna(d):
                continue
            iv_z = float(row["iv_zscore"]) if pd.notna(row["iv_zscore"]) else 0
            result[d.strftime("%Y-%m-%d")] = {
                "iv": float(row[iv_col]) if pd.notna(row[iv_col]) else None,
                "iv_zscore": iv_z,
                "iv_regime": "extreme" if iv_z > 2 else ("elevated" if iv_z > 1 else ("low" if iv_z < -1 else "normal")),
            }
        return result

    def _get_signal_bias(self, date_str: str) -> float:
        """Get daily model signal bias for a date."""
        if self.signals.empty:
            return 0.0
        try:
            # Handle both 'date' and 'Date' column names
            date_col = "date" if "date" in self.signals.columns else "Date"
            mask = self.signals[date_col].dt.strftime("%Y-%m-%d") == date_str
            if mask.any():
                row = self.signals[mask].iloc[-1]
                sig = str(row.get("signal", "")).upper()
                conf = float(row.get("confidence", 0.5))
                if sig == "BUY":
                    return conf
                elif sig == "SELL":
                    return -conf
        except Exception:
            pass
        return 0.0

    def get_context(self, date_str: str) -> dict:
        """
        Get fundamental context for a given date.

        Returns dict with:
          - fundamental_score: float [-3, +3]
          - components: dict with individual scores
          - flags: dict with event_imminent, vol_regime, etc.
          - alignment_with: function to check alignment with a direction
        """
        components = {}
        flags = {}

        # 1. Daily swing bias (SMA cross + momentum)
        daily = self._daily_features.get(date_str, {})
        swing_score = 0.0
        if daily:
            sma_c = daily.get("sma_cross", 0)
            mom_20 = daily.get("momentum_20d", 0)
            mom_5 = daily.get("momentum_5d", 0)

            # SMA cross: strong trend indicator
            if sma_c > 0.005:
                swing_score += 1.0
            elif sma_c > 0.001:
                swing_score += 0.5
            elif sma_c < -0.005:
                swing_score -= 1.0
            elif sma_c < -0.001:
                swing_score -= 0.5

            # 20d momentum
            if mom_20 > 0.05:
                swing_score += 0.5
            elif mom_20 < -0.05:
                swing_score -= 0.5

            # 5d momentum (more recent, higher weight for intraday)
            if mom_5 > 0.02:
                swing_score += 0.5
            elif mom_5 < -0.02:
                swing_score -= 0.5

        components["swing"] = round(swing_score, 2)

        # 2. Daily model signal
        signal_bias = self._get_signal_bias(date_str)
        components["model_signal"] = round(signal_bias, 2)

        # 3. COT positioning (find most recent available)
        cot_score = 0.0
        cot_data = None
        for days_back in range(0, 10):
            check_date = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=days_back)).strftime("%Y-%m-%d")
            if check_date in self._cot_features:
                cot_data = self._cot_features[check_date]
                break

        if cot_data:
            # COT contrarian: extreme speculator positioning = contrarian signal
            if cot_data.get("cot_extreme_bull"):
                cot_score = -0.5  # Contrarian: too bullish = bearish signal
                flags["cot_extreme"] = "BULL_EXTREME"
            elif cot_data.get("cot_extreme_bear"):
                cot_score = 0.5   # Contrarian: too bearish = bullish signal
                flags["cot_extreme"] = "BEAR_EXTREME"
            cot_idx = cot_data.get("cot_index")
            if cot_idx is not None:
                # Moderate positioning signal
                if 30 < cot_idx < 70:
                    pass  # neutral
                elif cot_idx >= 70:
                    cot_score = max(cot_score, -0.3)  # mild contrarian bearish
                elif cot_idx <= 30:
                    cot_score = min(cot_score, 0.3)   # mild contrarian bullish

        components["cot"] = round(cot_score, 2)

        # 4. Implied volatility regime
        cvol_data = None
        for days_back in range(0, 5):
            check_date = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=days_back)).strftime("%Y-%m-%d")
            if check_date in self._cvol_features:
                cvol_data = self._cvol_features[check_date]
                break

        if cvol_data:
            flags["iv_regime"] = cvol_data.get("iv_regime", "normal")
            flags["iv_zscore"] = cvol_data.get("iv_zscore", 0)
        else:
            flags["iv_regime"] = "unknown"

        # 5. Event proximity (simplified — use WASDE dates)
        flags["event_imminent"] = False
        # Known WASDE dates (2nd Tuesday/Thursday of each month) — simplified check
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            # WASDE is typically 10th-12th of month
            if 9 <= dt.day <= 13:
                flags["event_imminent"] = True
                flags["event_type"] = "WASDE"
        except Exception:
            pass

        # === TOTAL SCORE ===
        # Weights: swing context is most relevant for intraday alignment
        total = (
            components["swing"] * 1.0 +      # Daily trend alignment (highest weight)
            components["model_signal"] * 0.8 + # ML model prediction
            components["cot"] * 0.5            # COT contrarian
        )
        total = max(-3.0, min(3.0, total))

        return {
            "fundamental_score": round(total, 2),
            "components": components,
            "flags": flags,
            "date": date_str,
        }


# ═══════════════════════════════════════════════════════════════════════
#  V3 ENHANCED SETUP DETECTION
# ═══════════════════════════════════════════════════════════════════════

def _apply_fundamental_overlay(setup: dict, risk: dict, fund_ctx: dict, capital: float) -> tuple:
    """
    Apply fundamental context to modify the V2 signal.

    Rules:
    1. ALIGNMENT BOOST: If fund bias aligns with price action direction ->
       upgrade confidence (LOW->MEDIUM, MEDIUM->HIGH)
    2. CONFLICT FILTER: If fund bias opposes price action direction ->
       downgrade confidence or veto
    3. EVENT CAUTION: If WASDE imminent -> widen SL by 20% or skip if LOW conf
    4. VOL REGIME: If IV elevated + realized vol high -> reduce size

    Returns: modified (setup, risk) — does NOT change the underlying logic,
             only adjusts confidence and sizing.
    """
    setup_v3 = deepcopy(setup)
    risk_v3 = deepcopy(risk)

    fund_score = fund_ctx.get("fundamental_score", 0)
    flags = fund_ctx.get("flags", {})
    direction = setup_v3.get("direction", "FLAT")
    confidence = setup_v3.get("confidence", "LOW")

    if direction == "FLAT":
        return setup_v3, risk_v3

    # === 1. ALIGNMENT CHECK ===
    is_long = direction in ("LONG", "BUY")
    is_short = direction in ("SHORT", "SELL")
    fund_bullish = fund_score > 0.5
    fund_bearish = fund_score < -0.5
    fund_strong_bullish = fund_score > 1.5
    fund_strong_bearish = fund_score < -1.5
    fund_neutral = abs(fund_score) <= 0.5

    aligned = (is_long and fund_bullish) or (is_short and fund_bearish)
    strong_aligned = (is_long and fund_strong_bullish) or (is_short and fund_strong_bearish)
    conflicting = (is_long and fund_bearish) or (is_short and fund_bullish)
    strong_conflicting = (is_long and fund_strong_bearish) or (is_short and fund_strong_bullish)

    # Boost confidence if aligned
    if strong_aligned:
        if confidence == "LOW":
            setup_v3["confidence"] = "MEDIUM"
            setup_v3["description"] += f" [V3: FUND STRONGLY ALIGNED score={fund_score:+.1f}, upgraded LOW->MED]"
        elif confidence == "MEDIUM":
            setup_v3["confidence"] = "HIGH"
            setup_v3["description"] += f" [V3: FUND STRONGLY ALIGNED score={fund_score:+.1f}, upgraded MED->HIGH]"
    elif aligned:
        if confidence == "LOW":
            setup_v3["confidence"] = "MEDIUM"
            setup_v3["description"] += f" [V3: FUND ALIGNED score={fund_score:+.1f}, upgraded LOW->MED]"

    # Downgrade or veto if conflicting
    if strong_conflicting:
        if confidence == "LOW" or confidence == "MEDIUM":
            setup_v3["direction"] = "FLAT"
            risk_v3["trade_viable"] = False
            risk_v3["veto_reason"] = f"V3 VETO: fund score {fund_score:+.1f} strongly opposes {direction}"
            setup_v3["description"] += f" [V3: VETOED — fund strongly opposes]"
        elif confidence == "HIGH":
            setup_v3["confidence"] = "MEDIUM"
            setup_v3["description"] += f" [V3: FUND CONFLICTS score={fund_score:+.1f}, downgraded HIGH->MED]"
    elif conflicting:
        if confidence == "LOW":
            setup_v3["direction"] = "FLAT"
            risk_v3["trade_viable"] = False
            risk_v3["veto_reason"] = f"V3 VETO: fund score {fund_score:+.1f} opposes {direction} (LOW conf)"
        elif confidence == "MEDIUM":
            setup_v3["confidence"] = "LOW"
            setup_v3["description"] += f" [V3: FUND CONFLICTS score={fund_score:+.1f}, downgraded MED->LOW]"

    # === 2. EVENT CAUTION ===
    if flags.get("event_imminent"):
        event = flags.get("event_type", "unknown")
        if setup_v3.get("confidence") == "LOW":
            setup_v3["direction"] = "FLAT"
            risk_v3["trade_viable"] = False
            risk_v3["veto_reason"] = f"V3 VETO: {event} imminent + LOW confidence"
        else:
            # Widen SL by 20% for event risk
            sl = risk_v3.get("stop_loss", {})
            if sl.get("price") and sl.get("risk_usd"):
                risk_v3["stop_loss"]["risk_usd"] = round(sl["risk_usd"] * 1.2, 2)
                setup_v3["description"] += f" [V3: {event} imminent — SL widened 20%]"

    # === 3. IV REGIME CHECK ===
    iv_regime = flags.get("iv_regime", "normal")
    if iv_regime in ("extreme", "elevated"):
        pos = risk_v3.get("position_size", {})
        contracts = pos.get("contracts_mzs", 1)
        if iv_regime == "extreme" and contracts > 1:
            risk_v3["position_size"]["contracts_mzs"] = max(1, contracts // 2)
            setup_v3["description"] += f" [V3: IV {iv_regime} — size halved]"
        elif iv_regime == "elevated":
            # Just note it, V2 already handles realized vol
            pass

    return setup_v3, risk_v3


# ═══════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════

def run_backtest_v3(days=200, capital=1000.0, sample_hour=11, verbose=True):
    """
    Run side-by-side V2 vs V3 backtest.
    Same bars, same entry logic, but V3 applies fundamental overlay.
    """
    os.makedirs(_OUT_DIR, exist_ok=True)

    if verbose:
        print(f"{'='*80}")
        print(f"  QUANTAGENT V3 BACKTEST — Price Action + Fundamentals")
        print(f"  {days} days, ${capital:,.0f}, sample h{sample_hour} CT")
        print(f"{'='*80}\n")

    # Load fundamental context
    fund = FundamentalOverlay()

    # Load intraday bars
    print("[V3] Fetching 60m bars...")
    bars = fetch_intraday_bars(interval="60m", use_cache=True, cache_max_age_min=120)
    if bars.empty:
        return {"error": "No bars"}

    feat = build_intraday_features(bars, interval="60m")
    rth = feat[feat["is_rth"] == 1].copy() if "is_rth" in feat.columns else feat.copy()
    unique_days = sorted(rth["date_ct"].unique())
    test_days = unique_days[-(days + 5):-5] if len(unique_days) > days + 5 else unique_days[:-5]

    if not test_days:
        return {"error": "Not enough days"}

    if verbose:
        print(f"[V3] Test period: {test_days[0]} to {test_days[-1]} ({len(test_days)} days)\n")

    # -- Run both strategies --
    v2_trades = []
    v3_trades = []
    v2_capital = capital
    v3_capital = capital
    v2_peak = capital
    v3_peak = capital
    v2_max_dd = 0
    v3_max_dd = 0

    v2_equity = [{"day": "start", "capital": capital}]
    v3_equity = [{"day": "start", "capital": capital}]

    # Track fundamental overlay stats
    overlay_stats = {
        "upgrades": 0,
        "downgrades": 0,
        "vetoes": 0,
        "event_adjustments": 0,
        "aligned_trades": 0,
        "conflicting_trades": 0,
        "neutral_trades": 0,
    }

    for i, day in enumerate(test_days):
        day_str = str(day)
        day_bars = rth[rth["date_ct"] == day]
        sample_bars = day_bars[day_bars["hour_ct"] == sample_hour]
        if sample_bars.empty:
            for h in [sample_hour - 1, sample_hour + 1, sample_hour + 2]:
                sample_bars = day_bars[day_bars["hour_ct"] == h]
                if not sample_bars.empty:
                    break
        if sample_bars.empty:
            continue

        row = sample_bars.iloc[-1]
        sample_idx = feat.index.get_loc(sample_bars.index[-1])
        session_bars = day_bars[day_bars.index <= sample_bars.index[-1]]
        close = float(row["close"])
        atr = float(row.get("atr_14", 5) or 5)

        # -- V2: Pure price action --
        trend_dir, trend_str, trend_score = _detect_trend_v2(row)
        setup_v2 = _detect_setup_v2(row, session_bars, trend_dir, trend_score)
        trend_out = {"trend": trend_dir, "trend_strength": trend_str, "setup": setup_v2, "_score": trend_score}
        risk_v2 = rule_risk_agent_v2(row, trend_out, v2_capital)

        if not risk_v2["trade_viable"] or setup_v2["direction"] == "FLAT":
            signal_v2 = {"signal": "FLAT", "entry": None, "stop_loss": None,
                         "take_profit": None, "contracts": 0, "confidence": "LOW",
                         "max_hold_bars": 3, "_atr": atr}
        else:
            signal_v2 = {
                "signal": setup_v2["direction"],
                "entry": round(close, 2),
                "stop_loss": risk_v2["stop_loss"]["price"],
                "take_profit": risk_v2["take_profit"]["price"],
                "contracts": risk_v2["position_size"]["contracts_mzs"],
                "confidence": setup_v2["confidence"],
                "max_hold_bars": risk_v2["max_hold_bars"],
                "_atr": atr,
            }

        # -- V3: Price action + fundamental overlay --
        fund_ctx = fund.get_context(day_str)
        setup_v3, risk_v3 = _apply_fundamental_overlay(
            deepcopy(setup_v2), deepcopy(risk_v2), fund_ctx, v3_capital
        )

        # Re-build signal for V3
        trend_out_v3 = {"trend": trend_dir, "trend_strength": trend_str, "setup": setup_v3}
        # Need to re-run risk with possibly modified confidence
        if setup_v3["confidence"] == "LOW" and setup_v3["direction"] != "FLAT":
            # V2 risk vetoes all LOW — check if V3 changed it
            risk_v3_recheck = rule_risk_agent_v2(row, trend_out_v3, v3_capital)
            # But override with V3's decisions
            if risk_v3.get("trade_viable") and not risk_v3_recheck.get("trade_viable"):
                # V3 overlay didn't veto but V2 risk does (LOW conf veto)
                risk_v3 = risk_v3_recheck
            elif not risk_v3.get("trade_viable"):
                pass  # V3 already vetoed
            else:
                risk_v3 = risk_v3_recheck

        if not risk_v3.get("trade_viable") or setup_v3["direction"] == "FLAT":
            signal_v3 = {"signal": "FLAT", "entry": None, "stop_loss": None,
                         "take_profit": None, "contracts": 0, "confidence": "LOW",
                         "max_hold_bars": 3, "_atr": atr}
        else:
            signal_v3 = {
                "signal": setup_v3["direction"],
                "entry": round(close, 2),
                "stop_loss": risk_v3["stop_loss"]["price"],
                "take_profit": risk_v3["take_profit"]["price"],
                "contracts": risk_v3["position_size"]["contracts_mzs"],
                "confidence": setup_v3["confidence"],
                "max_hold_bars": risk_v3["max_hold_bars"],
                "_atr": atr,
            }

        # Track overlay changes
        v2_dir = signal_v2["signal"]
        v3_dir = signal_v3["signal"]
        v2_conf = signal_v2.get("confidence", "LOW")
        v3_conf = signal_v3.get("confidence", "LOW")
        fund_score = fund_ctx["fundamental_score"]

        if v2_dir != "FLAT":
            is_long = v2_dir in ("LONG", "BUY")
            if (is_long and fund_score > 0.5) or (not is_long and fund_score < -0.5):
                overlay_stats["aligned_trades"] += 1
            elif (is_long and fund_score < -0.5) or (not is_long and fund_score > 0.5):
                overlay_stats["conflicting_trades"] += 1
            else:
                overlay_stats["neutral_trades"] += 1

        if v2_dir != "FLAT" and v3_dir == "FLAT":
            overlay_stats["vetoes"] += 1
        elif v2_conf != v3_conf:
            conf_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            if conf_order.get(v3_conf, 0) > conf_order.get(v2_conf, 0):
                overlay_stats["upgrades"] += 1
            else:
                overlay_stats["downgrades"] += 1
        if fund_ctx["flags"].get("event_imminent") and v2_dir != "FLAT":
            overlay_stats["event_adjustments"] += 1

        # -- Evaluate both --
        future = feat.iloc[sample_idx + 1:sample_idx + 20]

        eval_v2 = _evaluate_trade_v2(signal_v2, future, v2_capital)
        eval_v3 = _evaluate_trade_v2(signal_v3, future, v3_capital)

        if eval_v2.get("traded"):
            v2_capital = round(v2_capital + eval_v2["pnl_usd"], 2)
            v2_peak = max(v2_peak, v2_capital)
            v2_max_dd = max(v2_max_dd, (v2_peak - v2_capital) / v2_peak * 100)

        if eval_v3.get("traded"):
            v3_capital = round(v3_capital + eval_v3["pnl_usd"], 2)
            v3_peak = max(v3_peak, v3_capital)
            v3_max_dd = max(v3_max_dd, (v3_peak - v3_capital) / v3_peak * 100)

        v2_equity.append({"day": day_str, "capital": v2_capital})
        v3_equity.append({"day": day_str, "capital": v3_capital})

        v2_trades.append({
            "day": day_str, "price": close, "signal": v2_dir, "conf": v2_conf,
            "setup": setup_v2["type"], "eval": eval_v2, "cap": v2_capital,
        })
        v3_trades.append({
            "day": day_str, "price": close, "signal": v3_dir, "conf": v3_conf,
            "setup": setup_v3["type"], "eval": eval_v3, "cap": v3_capital,
            "fund_score": fund_score, "fund_flags": fund_ctx["flags"],
            "fund_components": fund_ctx["components"],
        })

        if verbose:
            v2_pnl = eval_v2.get("pnl_usd", 0)
            v3_pnl = eval_v3.get("pnl_usd", 0)
            v2_icon = "  " if not eval_v2.get("traded") else (" W" if v2_pnl > 0 else " L")
            v3_icon = "  " if not eval_v3.get("traded") else (" W" if v3_pnl > 0 else " L")

            changed = ""
            if v2_dir != v3_dir or v2_conf != v3_conf:
                changed = f" ← V3:{v3_dir[:1]}/{v3_conf[:1]} fund={fund_score:+.1f}"

            print(
                f"  [{i+1:>3}/{len(test_days)}] {day_str} "
                f"| {close:>8.2f} "
                f"| V2:{v2_dir:>5}{v2_icon} ${v2_pnl:>+7.2f} "
                f"| V3:{v3_dir:>5}{v3_icon} ${v3_pnl:>+7.2f} "
                f"| fund={fund_score:>+5.2f}"
                f"{changed}"
            )

    # ═══════════════════════════════════════════════════════════════════
    #  STATISTICS
    # ═══════════════════════════════════════════════════════════════════

    def _calc_stats(trades_list, final_cap, max_dd, label):
        active = [t for t in trades_list if t["eval"].get("traded")]
        flat = [t for t in trades_list if not t["eval"].get("traded")]
        wins = [t for t in active if t["eval"]["pnl_usd"] > 0]
        losses = [t for t in active if t["eval"]["pnl_usd"] <= 0]
        total_pnl = sum(t["eval"]["pnl_usd"] for t in active)
        avg_win = np.mean([t["eval"]["pnl_usd"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["eval"]["pnl_usd"] for t in losses]) if losses else 0
        gross_w = sum(t["eval"]["pnl_usd"] for t in wins)
        gross_l = abs(sum(t["eval"]["pnl_usd"] for t in losses))
        pf = round(gross_w / gross_l, 2) if gross_l > 0 else float("inf")

        daily_rets = [t["eval"]["pnl_pct"] / 100 for t in active]
        sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if len(daily_rets) > 1 and np.std(daily_rets) > 0 else 0

        # Exit type breakdown
        exit_counts = {}
        for t in active:
            et = t["eval"].get("exit_type", "timeout")
            exit_counts[et] = exit_counts.get(et, 0) + 1

        # Setup breakdown
        setup_stats = {}
        for t in active:
            st = t["setup"]
            if st not in setup_stats:
                setup_stats[st] = {"n": 0, "w": 0, "pnl": 0}
            setup_stats[st]["n"] += 1
            setup_stats[st]["pnl"] = round(setup_stats[st]["pnl"] + t["eval"]["pnl_usd"], 2)
            if t["eval"]["pnl_usd"] > 0:
                setup_stats[st]["w"] += 1

        return {
            "label": label,
            "capital_final": final_cap,
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round((final_cap - capital) / capital * 100, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_samples": len(trades_list),
            "active": len(active),
            "flat": len(flat),
            "flat_rate": round(len(flat) / len(trades_list) * 100, 1) if trades_list else 0,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(active) * 100, 1) if active else 0,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": pf if pf != float("inf") else "inf",
            "sharpe": round(sharpe, 2),
            "exits": exit_counts,
            "by_setup": setup_stats,
        }

    stats_v2 = _calc_stats(v2_trades, v2_capital, v2_max_dd, "V2 (Price Only)")
    stats_v3 = _calc_stats(v3_trades, v3_capital, v3_max_dd, "V3 (Price + Fund)")

    # ═══════════════════════════════════════════════════════════════════
    #  OUTPUT
    # ═══════════════════════════════════════════════════════════════════

    if verbose:
        print(f"\n{'='*80}")
        print(f"  RESULTADOS: V2 (Price Only) vs V3 (Price + Fundamentals)")
        print(f"  Periodo: {test_days[0]} -> {test_days[-1]} ({len(test_days)} dias)")
        print(f"{'='*80}")

        header = f"  {'Metrica':<25} {'V2 (Price)':>15} {'V3 (Price+Fund)':>15} {'Delta':>12}"
        print(header)
        print(f"  {'-'*67}")

        def _fmt(val):
            if isinstance(val, float):
                return f"{val:,.2f}"
            return str(val)

        rows = [
            ("Capital final", f"${stats_v2['capital_final']:,.2f}", f"${stats_v3['capital_final']:,.2f}",
             f"${stats_v3['capital_final'] - stats_v2['capital_final']:+,.2f}"),
            ("Retorno total", f"{stats_v2['total_return_pct']:+.1f}%", f"{stats_v3['total_return_pct']:+.1f}%",
             f"{stats_v3['total_return_pct'] - stats_v2['total_return_pct']:+.1f}pp"),
            ("Max Drawdown", f"{stats_v2['max_drawdown_pct']:.1f}%", f"{stats_v3['max_drawdown_pct']:.1f}%",
             f"{stats_v3['max_drawdown_pct'] - stats_v2['max_drawdown_pct']:+.1f}pp"),
            ("Trades activos", f"{stats_v2['active']}", f"{stats_v3['active']}",
             f"{stats_v3['active'] - stats_v2['active']:+d}"),
            ("Win Rate", f"{stats_v2['win_rate']:.1f}%", f"{stats_v3['win_rate']:.1f}%",
             f"{stats_v3['win_rate'] - stats_v2['win_rate']:+.1f}pp"),
            ("Avg Win", f"${stats_v2['avg_win']:+.2f}", f"${stats_v3['avg_win']:+.2f}", ""),
            ("Avg Loss", f"${stats_v2['avg_loss']:+.2f}", f"${stats_v3['avg_loss']:+.2f}", ""),
            ("Profit Factor", f"{stats_v2['profit_factor']}", f"{stats_v3['profit_factor']}", ""),
            ("Sharpe", f"{stats_v2['sharpe']:.2f}", f"{stats_v3['sharpe']:.2f}", ""),
            ("Flat rate", f"{stats_v2['flat_rate']:.0f}%", f"{stats_v3['flat_rate']:.0f}%", ""),
        ]

        for label, v2_val, v3_val, delta in rows:
            print(f"  {label:<25} {v2_val:>15} {v3_val:>15} {delta:>12}")

        print(f"\n  -- Fundamental Overlay Impact --")
        print(f"  Confidence upgrades:   {overlay_stats['upgrades']}")
        print(f"  Confidence downgrades: {overlay_stats['downgrades']}")
        print(f"  Trades vetoed:         {overlay_stats['vetoes']}")
        print(f"  Event adjustments:     {overlay_stats['event_adjustments']}")
        print(f"  Aligned trades:        {overlay_stats['aligned_trades']}")
        print(f"  Conflicting trades:    {overlay_stats['conflicting_trades']}")
        print(f"  Neutral trades:        {overlay_stats['neutral_trades']}")

        # Analyze vetoed trades: were they good or bad vetoes?
        v3_vetoed = [
            (v2_trades[i], v3_trades[i])
            for i in range(len(v2_trades))
            if v2_trades[i]["eval"].get("traded") and not v3_trades[i]["eval"].get("traded")
        ]
        if v3_vetoed:
            vetoed_pnl = sum(t[0]["eval"]["pnl_usd"] for t in v3_vetoed)
            vetoed_wins = sum(1 for t in v3_vetoed if t[0]["eval"]["pnl_usd"] > 0)
            vetoed_losses = len(v3_vetoed) - vetoed_wins
            print(f"\n  -- Trades vetados por V3 (habrían sido en V2) --")
            print(f"  Total: {len(v3_vetoed)} trades vetados")
            print(f"  Habrían sido: {vetoed_wins}W / {vetoed_losses}L = ${vetoed_pnl:+.2f}")
            if vetoed_pnl < 0:
                print(f"  ✅ BUEN VETO: Evitó ${abs(vetoed_pnl):.2f} en pérdidas")
            else:
                print(f"  ⚠️  Vetó trades que habrían dado ${vetoed_pnl:.2f} de ganancia")

        # Analyze upgraded trades
        v3_new = [
            (v2_trades[i], v3_trades[i])
            for i in range(len(v2_trades))
            if not v2_trades[i]["eval"].get("traded") and v3_trades[i]["eval"].get("traded")
        ]
        if v3_new:
            new_pnl = sum(t[1]["eval"]["pnl_usd"] for t in v3_new)
            new_wins = sum(1 for t in v3_new if t[1]["eval"]["pnl_usd"] > 0)
            new_losses = len(v3_new) - new_wins
            print(f"\n  -- Trades nuevos por V3 (no estaban en V2) --")
            print(f"  Total: {len(v3_new)} trades nuevos (confidence upgrade)")
            print(f"  Resultado: {new_wins}W / {new_losses}L = ${new_pnl:+.2f}")
            if new_pnl > 0:
                print(f"  ✅ BUEN UPGRADE: Ganó ${new_pnl:.2f} extra")
            else:
                print(f"  ⚠️  Trades nuevos perdieron ${abs(new_pnl):.2f}")

        # Verdict
        print(f"\n  {'='*67}")
        print(f"  VEREDICTO:")
        v3_better = stats_v3["total_return_pct"] > stats_v2["total_return_pct"]
        v3_safer = stats_v3["max_drawdown_pct"] < stats_v2["max_drawdown_pct"]
        v3_higher_wr = stats_v3["win_rate"] > stats_v2["win_rate"]

        if v3_better and v3_safer:
            print(f"  ✅ V3 SUPERIOR: Mayor retorno Y menor drawdown.")
            print(f"     RECOMENDACIÓN: Integrar fundamentals al QuantAgent.")
        elif v3_better:
            print(f"  ✅ V3 más rentable (+{stats_v3['total_return_pct'] - stats_v2['total_return_pct']:.1f}pp)")
            if stats_v3["max_drawdown_pct"] - stats_v2["max_drawdown_pct"] < 5:
                print(f"     DD ligeramente mayor pero aceptable.")
                print(f"     RECOMENDACIÓN: Integrar con monitoreo de DD.")
            else:
                print(f"     ⚠️  Pero DD significativamente mayor. Ajustar sizing.")
        elif v3_safer:
            print(f"  🛡️  V3 más seguro (DD {stats_v3['max_drawdown_pct']:.1f}% vs {stats_v2['max_drawdown_pct']:.1f}%)")
            if stats_v2["total_return_pct"] - stats_v3["total_return_pct"] < 3:
                print(f"     Retorno similar con mejor gestión de riesgo.")
                print(f"     RECOMENDACIÓN: Integrar — mejor risk-adjusted return.")
            else:
                print(f"     Pero retorno significativamente menor. Evaluar trade-off.")
        elif v3_higher_wr:
            print(f"  📊 V3 mejor win rate ({stats_v3['win_rate']:.1f}% vs {stats_v2['win_rate']:.1f}%)")
            print(f"     Menos trades pero más precisos.")
        else:
            print(f"  ❌ V3 no mejora V2 en este periodo.")
            print(f"     Fundamentals no agregaron valor con estas reglas.")
            print(f"     Considerar: ajustar pesos o integrar news/sentiment live.")

        print(f"  {'='*67}\n")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "type": "v2_vs_v3_comparison",
        "config": {"days": days, "capital": capital, "sample_hour": sample_hour},
        "period": {"start": str(test_days[0]), "end": str(test_days[-1])},
        "v2_stats": stats_v2,
        "v3_stats": stats_v3,
        "overlay_impact": overlay_stats,
        "v2_equity": v2_equity,
        "v3_equity": v3_equity,
        "v3_trades": v3_trades,  # Include fund context for analysis
    }

    out_file = os.path.join(_OUT_DIR, "backtest_v3_comparison.json")
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        if verbose:
            print(f"  Saved: {out_file}")
    except Exception as e:
        print(f"  Save error: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--capital", type=float, default=1000)
    parser.add_argument("--sample-hour", type=int, default=11)
    args = parser.parse_args()
    run_backtest_v3(days=args.days, capital=args.capital, sample_hour=args.sample_hour)
