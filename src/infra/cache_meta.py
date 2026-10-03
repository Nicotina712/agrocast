"""
Frescura de cachés JSON basada en CONTENIDO, no en mtime del archivo.

Por qué: en GitHub Actions el checkout pone el mtime de todos los archivos
en "ahora", así que un TTL por os.path.getmtime() siempre daba "fresco" y
los módulos nunca re-fetcheaban (basis/china/brasil/WASDE congelados desde
junio 2026). Mismo bug que ya se había arreglado en current_contract.json.

Uso:
    from src.infra.cache_meta import is_fresh, stamp
    if is_fresh(_CACHE_PATH, _TTL_HOURS): ...
    json.dump(stamp(result), f)
"""
import json
import os
from datetime import datetime, timezone

_TS_KEYS = ("_cached_at", "_generated_at", "generated_at", "timestamp")


def _parse(ts: str):
    t = datetime.fromisoformat(str(ts))
    if t.tzinfo is None:          # legacy naive → hora local de quien lo escribió
        t = t.astimezone()
    return t


def is_fresh(path: str, ttl_hours: float) -> bool:
    """True si el JSON tiene un timestamp de generación más nuevo que ttl_hours.
    Sin timestamp legible → False (fuerza refresh; el próximo write lo estampa)."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for k in _TS_KEYS:
            if isinstance(data, dict) and data.get(k):
                age = datetime.now(timezone.utc) - _parse(data[k])
                return age.total_seconds() < ttl_hours * 3600
    except Exception:
        pass
    return False


def stamp(data: dict) -> dict:
    """Agrega _cached_at (UTC ISO) al dict antes de persistirlo."""
    if isinstance(data, dict):
        data["_cached_at"] = datetime.now(timezone.utc).isoformat()
    return data
