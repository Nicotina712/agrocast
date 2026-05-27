"""
src/data/load_data.py
Descarga datos de mercado desde Yahoo Finance y los cachea en disco.

✅ MEJORA: los datos se guardan en data/raw_market.csv para no
   re-descargar 10 años cada vez que corre el pipeline.
   Solo re-descarga si el archivo tiene más de 1 día de antigüedad.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# Ubicación del caché (relativo a la raíz del proyecto)
_SCRIPT_DIR  = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")
_ROLLOVER_PATH = os.path.join(_PROJECT_ROOT, "data", "rollover_flags.csv")
_CACHE_TTL_H  = 24  # horas


def _cache_is_fresh() -> bool:
    """
    True si el caché existe y los DATOS son recientes.

    IMPORTANTE: No usar mtime del archivo (se actualiza con git operations).
    En su lugar, verificar la última fecha dentro del CSV.
    Considerar fresco si incluye el último día hábil (Mon-Fri).
    """
    if not os.path.exists(_CACHE_PATH):
        return False
    try:
        df = pd.read_csv(_CACHE_PATH, parse_dates=["Date"], usecols=["Date"])
        if df.empty:
            return False
        last_date = df["Date"].max().date()
        today = datetime.now().date()
        # Find the most recent business day (Mon-Fri)
        latest_biz = today
        while latest_biz.weekday() >= 5:  # Sat=5, Sun=6
            latest_biz -= timedelta(days=1)
        # Fresh only if data includes the latest business day
        # (allow 1 day grace for market data delay — e.g. today's data
        #  may not be available until after market close)
        days_stale = (latest_biz - last_date).days
        return days_stale <= 1
    except Exception:
        # If we can't read/parse the file, treat as stale to force refresh
        return False


def _detect_and_flag_rollovers(df: pd.DataFrame) -> set:
    """
    Detecta días de rollover en la serie continua ZS=F.

    Un rollover ocurre cuando el contrato front-month cambia (ej. MAY26 → JUL26)
    y la serie continua muestra un salto de precio artificial.

    Criterios de detección:
      1. |soy_ret| > 1.5% AND |soy_ret| > 2.5 * |corn_ret| AND |corn_ret| < 1%
         (soja se mueve mucho más que maíz — señal de rollover, no de mercado)
      2. |soy_ret| > 2.5% (movimiento solo muy grande, independiente del maíz)

    Guarda los resultados en data/rollover_flags.csv.
    Retorna un set de fechas (pd.Timestamp) marcadas como rollover.
    """
    if "Soybeans" not in df.columns:
        return set()

    work = df.copy().sort_values("Date").reset_index(drop=True)
    work["soy_ret"]  = work["Soybeans"].pct_change(fill_method=None)
    work["corn_ret"] = work["Maize"].pct_change(fill_method=None) if "Maize" in work.columns else 0.0

    # Criterio 1: soja se mueve mucho más que maíz (patrón de rollover)
    crit1 = (
        (work["soy_ret"].abs() > 0.015) &
        (work["soy_ret"].abs() > 2.5 * work["corn_ret"].abs()) &
        (work["corn_ret"].abs() < 0.01)
    )
    # Criterio 2: movimiento solo muy grande en soja
    crit2 = work["soy_ret"].abs() > 0.025

    work["flagged"] = crit1 | crit2

    # Guardar flags
    flags_df = work[["Date", "soy_ret", "corn_ret", "flagged"]].copy()
    os.makedirs(os.path.dirname(_ROLLOVER_PATH), exist_ok=True)
    flags_df.to_csv(_ROLLOVER_PATH, index=False)

    rollover_dates = set(work.loc[work["flagged"], "Date"].tolist())
    n_flagged = len(rollover_dates)
    if n_flagged > 0:
        print(f"[Rollover] {n_flagged} dias flaggeados -> {_ROLLOVER_PATH}")
    else:
        print("[Rollover] Sin rollovers detectados en la serie de soja.")

    return rollover_dates


def _clean_ticker(df: pd.DataFrame, name: str, with_ohlc: bool = False) -> pd.DataFrame:
    """Normaliza un DataFrame de yfinance. Si with_ohlc, expone Open/High/Low/Close."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    df[name] = df["Close"].astype(float)
    cols = ["Date", name]
    if with_ohlc:
        for src, dst in [("Open", f"{name}_Open"), ("High", f"{name}_High"),
                         ("Low", f"{name}_Low")]:
            if src in df.columns:
                df[dst] = df[src].astype(float)
                cols.append(dst)
    return df[cols]


def load_all_data() -> pd.DataFrame:
    """
    Descarga (o carga desde caché) los precios de:
      - Soja      ZS=F
      - Maíz      ZC=F
      - Petróleo  CL=F
      - Dólar     DX-Y.NYB

    Retorna un DataFrame con columnas: Date, Soybeans, Maize, Oil, Dollar.
    """

    # ── Intentar leer caché ──────────────────────────────────────
    if _cache_is_fresh():
        print(f"[Cache] Cargando datos desde cache ({_CACHE_PATH})")
        df = pd.read_csv(_CACHE_PATH, parse_dates=["Date"])
        print(f"[OK] Dataset cacheado: {df.shape}")
        return df

    # ── Descargar desde Yahoo Finance ────────────────────────────
    print("[Data] Descargando datos de mercado (Yahoo Finance)...")

    tickers = {
        "Soybeans":    "ZS=F",
        "Maize":       "ZC=F",
        "Wheat":       "ZW=F",   # trigo CBOT
        "Oil":         "CL=F",
        "Dollar":      "DX-Y.NYB",
        "SoybeanMeal": "ZM=F",   # harina de soja — proxy de demanda proteínica
        "SoybeanOil":  "ZL=F",   # aceite de soja — proxy de demanda biocombustible
    }

    frames = {}
    for name, ticker in tickers.items():
        try:
            raw = yf.download(ticker, period="10y", interval="1d", progress=False)
            if raw.empty:
                print(f"[WARN] {ticker} devolvio datos vacios")
                continue
            frames[name] = _clean_ticker(raw, name, with_ohlc=(name == "Soybeans"))
        except Exception as e:
            print(f"❌ Error descargando {ticker}: {e}")

    if "Soybeans" not in frames:
        # ── Fallback: usar caché stale si existe ─────────────────
        # yfinance falla a veces en CI/CD (IPs bloqueadas por Yahoo Finance,
        # rate limiting, o cambios de API). Si hay datos en caché (aunque sean
        # stale), los usamos para que el pipeline pueda continuar con datos
        # ligeramente desactualizados en lugar de crashear.
        if os.path.exists(_CACHE_PATH):
            try:
                df_stale = pd.read_csv(_CACHE_PATH, parse_dates=["Date"])
                stale_days = (
                    pd.Timestamp.today().date() - df_stale["Date"].max().date()
                ).days
                print(f"⚠️  yfinance falló (ZS=F). Usando caché stale ({stale_days}d antigua).")
                print(f"   ACCION: revisar si Yahoo Finance está bloqueando las IPs del runner.")
                return df_stale
            except Exception as _fe:
                pass
        raise RuntimeError(
            "No se pudieron descargar datos de soja (ZS=F) y no hay caché disponible. "
            "Revisa la conexión o configura YFINANCE_FALLBACK_PATH."
        )

    # ── Merge ────────────────────────────────────────────────────
    df = frames["Soybeans"]
    for name in ("Maize", "Wheat", "Oil", "Dollar", "SoybeanMeal", "SoybeanOil"):
        if name in frames:
            df = df.merge(frames[name], on="Date", how="left")
        else:
            df[name] = float("nan")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # ── Detectar rollovers ───────────────────────────────────────
    _detect_and_flag_rollovers(df)

    # ── Guardar caché ────────────────────────────────────────────
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    df.to_csv(_CACHE_PATH, index=False)
    print(f"[OK] Datos guardados en {_CACHE_PATH}")
    print(f"[OK] Dataset final: {df.shape}")
    print(df[['Date','Soybeans','Maize']].tail(3).to_string())

    return df
