"""
src/data/fetch_gdelt.py
Descarga el tono histórico de noticias agrícolas desde GDELT 2.0 Timeline API.

GDELT (Global Database of Events, Language, and Tone) analiza millones de
artículos diariamente y calcula un score de tono (negativo = pesimista,
positivo = optimista) para cualquier consulta temática.

Caché: data/gdelt_tone.csv — TTL 7 días.
"""

import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests


def _safe_print(msg: str) -> None:
    """Print sin crashear en consolas que no soportan UTF-8 (Windows cp1252)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            enc = sys.stdout.encoding or "ascii"
            print(msg.encode(enc, errors="replace").decode(enc, errors="replace"))
        except Exception:
            print(msg.encode("ascii", errors="ignore").decode("ascii"))

_SCRIPT_DIR    = os.path.dirname(__file__)
_PROJECT_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_CACHE_PATH    = os.path.join(_PROJECT_ROOT, "data", "gdelt_tone.csv")
_FAIL_PATH     = os.path.join(_PROJECT_ROOT, "data", "gdelt_tone_fail.txt")
_TTL_DAYS      = 7
_FAIL_TTL_HRS  = 6   # no reintentar GDELT por 6h si falló

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
# Query con OR — GDELT requiere paréntesis y comillas para frases multi-palabra
QUERY     = '(soybean OR soja OR "soybean oil" OR "crop report")'


def _cache_fresh() -> bool:
    if not os.path.exists(_CACHE_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(_CACHE_PATH))
    return age < timedelta(days=_TTL_DAYS)


def _recently_failed() -> bool:
    """Devuelve True si GDELT falló en las últimas _FAIL_TTL_HRS horas."""
    if not os.path.exists(_FAIL_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(_FAIL_PATH))
    return age < timedelta(hours=_FAIL_TTL_HRS)


def _mark_failed() -> None:
    os.makedirs(os.path.dirname(_FAIL_PATH), exist_ok=True)
    with open(_FAIL_PATH, "w") as f:
        f.write(datetime.now().isoformat())


def _clear_fail() -> None:
    if os.path.exists(_FAIL_PATH):
        os.remove(_FAIL_PATH)


def _parse_date(raw: str) -> pd.Timestamp | None:
    """Parsea '20180115000000' → Timestamp."""
    try:
        return pd.Timestamp(str(raw)[:8])   # YYYYMMDD es suficiente
    except Exception:
        return None


def _fetch_gdelt(start: str, end: str, retries: int = 2) -> pd.DataFrame:
    """
    Llama a GDELT con retries y backoff. La API es notoriamente lenta/inestable:
    timeouts de connect ocurren con frecuencia, especialmente para ventanas
    grandes. Connect timeout corto + read timeout largo + retry.
    """
    params = {
        "query":          QUERY,
        "mode":           "TimelineTone",
        "format":         "json",
        "startdatetime":  start,
        "enddatetime":    end,
        "timelinesmooth": 7,
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                GDELT_URL,
                params=params,
                timeout=(10, 60),  # (connect, read)
                headers={"User-Agent": "Mozilla/5.0 AgroCastBot/1.0"},
            )

            if resp.status_code == 429:
                _safe_print("  [WARN] GDELT rate limit (429) - se reintentara en 6h")
                _mark_failed()
                return pd.DataFrame()

            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}"
                _safe_print(f"  [WARN] GDELT {last_err} (intento {attempt + 1}/{retries + 1})")
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                return pd.DataFrame()

            body = resp.text.strip()
            if not body:
                _safe_print("  [WARN] GDELT respuesta vacia")
                return pd.DataFrame()

            try:
                data = resp.json()
            except Exception:
                _safe_print("  [WARN] GDELT respuesta no es JSON")
                return pd.DataFrame()

            break  # éxito, salir del loop de reintentos

        except (requests.ConnectTimeout, requests.ReadTimeout, requests.ConnectionError) as e:
            last_err = type(e).__name__
            _safe_print(f"  [WARN] GDELT timeout/conn {last_err} (intento {attempt + 1}/{retries + 1})")
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return pd.DataFrame()
        except Exception as e:
            _safe_print(f"  [ERR] GDELT error: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    records = []
    for series in data.get("timeline", []):
        for item in series.get("data", []):
            dt = _parse_date(item.get("date", ""))
            if dt is None:
                continue
            try:
                records.append({"Date": dt, "tone": float(item["value"])})
            except (KeyError, ValueError, TypeError):
                continue

    if records:
        _clear_fail()  # descarga exitosa → limpiar marca de fallo

    return pd.DataFrame(records)


def fetch_gdelt_tone(start_year: int = 2018) -> pd.DataFrame:
    """
    Retorna DataFrame con columnas: Date, gdelt_tone.
    gdelt_tone: tono promedio de cobertura noticiosa agrícola (negativo=pesimista).

    GDELT TimelineTone limita a ~1 año por consulta. Iteramos por año y
    concatenamos los resultados. Si una ventana falla, seguimos con la siguiente.
    """
    if _cache_fresh():
        _safe_print(f"[GDELT] desde cache ({_CACHE_PATH})")
        return pd.read_csv(_CACHE_PATH, parse_dates=["Date"])

    if _recently_failed():
        _safe_print("[GDELT] omitido (fallo hace menos de 6h - reintentando mas tarde)")
        return pd.DataFrame(columns=["Date", "gdelt_tone"])

    _safe_print("[GDELT] Descargando datos historicos por ano...")
    current_year = datetime.now().year
    all_chunks = []
    failed_years = []

    for year in range(max(start_year, 2018), current_year + 1):
        start = f"{year}0101000000"
        if year == current_year:
            end = datetime.now().strftime("%Y%m%d%H%M%S")
        else:
            end = f"{year}1231235959"

        chunk = _fetch_gdelt(start, end)
        if not chunk.empty:
            all_chunks.append(chunk)
            _safe_print(f"   [GDELT] {year}: {len(chunk)} registros")
        else:
            failed_years.append(year)
            _safe_print(f"   [GDELT] {year}: vacio")
        time.sleep(1.5)  # cortesía con la API

    if not all_chunks:
        _safe_print("[GDELT] sin datos en ningun ano - marcando fallo")
        _mark_failed()
        # Fallback: si tenemos cache vieja en disco (TTL vencido), la usamos
        # antes que devolver vacio - es preferible data stale a data nula.
        if os.path.exists(_CACHE_PATH):
            _safe_print("[GDELT] usando cache vieja como fallback")
            return pd.read_csv(_CACHE_PATH, parse_dates=["Date"])
        return pd.DataFrame(columns=["Date", "gdelt_tone"])

    if failed_years:
        _safe_print(f"[GDELT] anos sin datos: {failed_years} (parcialmente OK)")

    df = pd.concat(all_chunks, ignore_index=True)

    result = (
        df.groupby("Date")["tone"]
        .mean()
        .reset_index()
        .rename(columns={"tone": "gdelt_tone"})
        .sort_values("Date")
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    result.to_csv(_CACHE_PATH, index=False)
    _clear_fail()  # parcial OK también limpia el flag
    _safe_print(f"[GDELT] OK: {len(result)} registros unicos guardados en cache")

    return result
