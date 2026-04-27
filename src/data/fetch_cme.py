"""
src/data/fetch_cme.py
Scraper de CME Group para Soybean Futures (ZS).

Obtiene datos diarios públicos del endpoint de settlements de cmegroup.com:
  - Settlement price del front-month
  - Volume del front-month
  - Open Interest del front-month
  - OI total agregado (suma de todos los vencimientos listados)
  - Spread front/next (validación contra curve_features)

Persiste histórico en data/cme_history.csv (append-only por fecha).
Usa solo los endpoints JSON expuestos en la web pública de CME — sin auth.

Notas:
  - El endpoint de settlements responde HTML+JSON embebido. Para mantenerlo
    simple, usamos el endpoint /CmeWS/mvc/Quotes/Future/G/* que devuelve JSON.
  - ZS = Soybean futures (5000 bu). Producto ID = 320, exchange = G (CBOT).
  - Si el fetch falla, el módulo NO rompe el pipeline: devuelve None.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT_PATH     = os.path.join(_PROJECT_ROOT, "data", "cme_history.csv")

# Soybean Futures (ZS) en CBOT — productId=320 según CME public APIs
_QUOTES_URL = "https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/320/G"
_REFERER    = "https://www.cmegroup.com/markets/agriculture/oilseeds/soybean.quotes.html"

# Endpoints de respaldo (Yahoo Finance — sin auth, sin CORS)
# ZS=F → front continuo de Soybean Futures
_YF_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
_YF_SYMBOLS   = ["ZS=F", "ZSN26.CBT", "ZSX26.CBT"]  # front + 2 vencimientos como aproximación

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept":           "application/json, text/plain, */*",
    "Accept-Language":  "en-US,en;q=0.9",
    "Accept-Encoding":  "gzip, deflate, br",
    "Referer":          _REFERER,
    "Origin":           "https://www.cmegroup.com",
    "Sec-Fetch-Dest":   "empty",
    "Sec-Fetch-Mode":   "cors",
    "Sec-Fetch-Site":   "same-origin",
    "Sec-Ch-Ua":        '"Chromium";v="124", "Google Chrome";v="124"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
}

_COLUMNS = [
    "Date", "front_month", "front_settle", "front_volume", "front_oi",
    "next_month", "next_settle", "next_volume", "next_oi",
    "total_oi", "n_contracts_listed", "front_next_spread_cme",
]


def _to_float(x) -> Optional[float]:
    try:
        if x is None or x == "" or x == "-":
            return None
        return float(str(x).replace(",", "").replace("'", ""))
    except Exception:
        return None


def _parse_quotes(payload: dict) -> Optional[dict]:
    """Extrae los datos relevantes del JSON de CME."""
    quotes = payload.get("quotes") or []
    if not quotes:
        return None

    rows = []
    for q in quotes:
        rows.append({
            "month":   q.get("expirationCode") or q.get("expirationMonth"),
            "label":   q.get("expirationDate"),
            "settle":  _to_float(q.get("priorSettle") or q.get("last")),
            "volume":  _to_float(q.get("volume")),
            "oi":      _to_float(q.get("openInterest")),
        })

    df = pd.DataFrame(rows).dropna(subset=["settle"])
    if df.empty:
        return None

    front = df.iloc[0]
    nxt   = df.iloc[1] if len(df) > 1 else None

    total_oi = float(df["oi"].fillna(0).sum())

    out = {
        "Date":             str(date.today()),
        "front_month":      str(front.get("label") or front.get("month")),
        "front_settle":     float(front["settle"]) if pd.notna(front["settle"]) else None,
        "front_volume":     float(front["volume"]) if pd.notna(front["volume"]) else None,
        "front_oi":         float(front["oi"])     if pd.notna(front["oi"])     else None,
        "next_month":       str(nxt.get("label") or nxt.get("month")) if nxt is not None else None,
        "next_settle":      float(nxt["settle"]) if nxt is not None and pd.notna(nxt["settle"]) else None,
        "next_volume":      float(nxt["volume"]) if nxt is not None and pd.notna(nxt["volume"]) else None,
        "next_oi":          float(nxt["oi"])     if nxt is not None and pd.notna(nxt["oi"])     else None,
        "total_oi":         total_oi,
        "n_contracts_listed": int(len(df)),
    }
    if out["front_settle"] is not None and out["next_settle"] is not None:
        out["front_next_spread_cme"] = round(out["next_settle"] - out["front_settle"], 4)
    else:
        out["front_next_spread_cme"] = None
    return out


def _try_cme(timeout: int = 15) -> Optional[dict]:
    """Intento 1: API pública CME con sesión que primero visita la página."""
    try:
        sess = requests.Session()
        sess.headers.update(_HEADERS)
        # Warm-up: cargar la página soja para recibir cookies (incl. Akamai bm_sv)
        sess.get(_REFERER, timeout=timeout)
        r = sess.get(_QUOTES_URL, timeout=timeout)
        r.raise_for_status()
        return _parse_quotes(r.json())
    except Exception as e:
        print(f"[CME] CME directo fallo: {e}")
        return None


def _try_yahoo(timeout: int = 15) -> Optional[dict]:
    """Fallback: yfinance — entrega settle, volume y openInterest sin auth."""
    try:
        import yfinance as yf
    except ImportError:
        print("[CME] yfinance no instalado")
        return None

    rows = []
    for sym in _YF_SYMBOLS:
        try:
            t    = yf.Ticker(sym)
            info = t.fast_info
            hist = t.history(period="2d")
            if hist.empty:
                continue
            last_close = float(hist["Close"].iloc[-1])
            last_vol   = float(hist["Volume"].iloc[-1]) if "Volume" in hist else None
            # OI no siempre disponible en yfinance — intentamos info dict
            oi = None
            try:
                oi = float(t.info.get("openInterest")) if t.info.get("openInterest") else None
            except Exception:
                pass
            rows.append({
                "label":  sym,
                "settle": last_close,
                "volume": last_vol,
                "oi":     oi,
            })
        except Exception as e:
            print(f"[CME] yfinance {sym} fallo: {e}")
            continue

    if not rows:
        return None
    df = pd.DataFrame(rows).dropna(subset=["settle"])
    if df.empty:
        return None

    front = df.iloc[0]
    nxt   = df.iloc[1] if len(df) > 1 else None
    total_oi = float(df["oi"].fillna(0).sum())

    out = {
        "Date":          str(date.today()),
        "front_month":   str(front["label"]),
        "front_settle":  float(front["settle"]),
        "front_volume":  float(front["volume"]) if pd.notna(front["volume"]) else None,
        "front_oi":      float(front["oi"])     if pd.notna(front["oi"])     else None,
        "next_month":    str(nxt["label"]) if nxt is not None else None,
        "next_settle":   float(nxt["settle"]) if nxt is not None else None,
        "next_volume":   float(nxt["volume"]) if nxt is not None and pd.notna(nxt["volume"]) else None,
        "next_oi":       float(nxt["oi"])     if nxt is not None and pd.notna(nxt["oi"])     else None,
        "total_oi":      total_oi,
        "n_contracts_listed": int(len(df)),
    }
    if out["front_settle"] is not None and out["next_settle"] is not None:
        out["front_next_spread_cme"] = round(out["next_settle"] - out["front_settle"], 4)
    else:
        out["front_next_spread_cme"] = None
    out["_source"] = "yahoo"
    return out


def fetch_cme_snapshot(timeout: int = 15) -> Optional[dict]:
    """
    Descarga snapshot. Intenta CME directo; si falla cae a Yahoo Finance.
    None si todas las fuentes fallan.
    """
    snap = _try_cme(timeout)
    if snap is not None:
        snap["_source"] = "cme"
        return snap
    print("[CME] cayendo a Yahoo Finance fallback…")
    return _try_yahoo(timeout)


def append_snapshot(snap: dict) -> None:
    """Append-only: si ya existe la fecha, la sobrescribe."""
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    if os.path.exists(_OUT_PATH):
        try:
            hist = pd.read_csv(_OUT_PATH)
        except Exception:
            hist = pd.DataFrame(columns=_COLUMNS)
    else:
        hist = pd.DataFrame(columns=_COLUMNS)

    hist = hist[hist["Date"] != snap["Date"]]
    row  = {c: snap.get(c) for c in _COLUMNS}
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist = hist.sort_values("Date").reset_index(drop=True)
    hist.to_csv(_OUT_PATH, index=False)


def load_cme_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mergea cme_history.csv contra features.csv por Date con ffill.
    Añade columnas: cme_front_oi, cme_total_oi, cme_volume,
                    cme_oi_change_5d, cme_oi_change_pct_5d,
                    cme_volume_zscore_20d, cme_spread.
    """
    if not os.path.exists(_OUT_PATH):
        return features_df

    try:
        hist = pd.read_csv(_OUT_PATH)
    except Exception:
        return features_df

    if hist.empty:
        return features_df

    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist.sort_values("Date").reset_index(drop=True)

    hist["cme_front_oi"]            = hist["front_oi"]
    hist["cme_total_oi"]            = hist["total_oi"]
    hist["cme_volume"]              = hist["front_volume"]
    hist["cme_spread"]              = hist["front_next_spread_cme"]
    hist["cme_oi_change_5d"]        = hist["cme_total_oi"].diff(5)
    hist["cme_oi_change_pct_5d"]    = hist["cme_total_oi"].pct_change(5)
    vol_mean = hist["cme_volume"].rolling(20, min_periods=5).mean()
    vol_std  = hist["cme_volume"].rolling(20, min_periods=5).std()
    hist["cme_volume_zscore_20d"]   = (hist["cme_volume"] - vol_mean) / (vol_std + 1e-9)

    keep = ["Date", "cme_front_oi", "cme_total_oi", "cme_volume",
            "cme_oi_change_5d", "cme_oi_change_pct_5d",
            "cme_volume_zscore_20d", "cme_spread"]
    h = hist[keep].copy()

    df = features_df.copy()
    if "Date" not in df.columns:
        return features_df
    df["Date"] = pd.to_datetime(df["Date"])

    merged = pd.merge_asof(
        df.sort_values("Date"),
        h.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    merged[keep[1:]] = merged[keep[1:]].ffill().fillna(0)
    return merged


def run_cme_snapshot() -> Optional[dict]:
    """Punto de entrada para pipeline: snapshot + persistencia."""
    snap = fetch_cme_snapshot()
    if snap is None:
        return None
    append_snapshot(snap)
    print(f"[CME] {snap['Date']} front={snap['front_month']} settle={snap['front_settle']} "
          f"OI_total={snap['total_oi']:.0f}")
    return snap


if __name__ == "__main__":
    out = run_cme_snapshot()
    print(json.dumps(out, indent=2, default=str))
