"""
GDELT client robusto para el portfolio de robots.

Estrategia multi-capa para evitar 429:
  1. gdeltdoc library (maneja sesion y headers correctos)
  2. Exponential backoff con jitter en cada intento
  3. Cache local de 2h para no re-pedir lo mismo
  4. Fallback a requests directo si gdeltdoc falla
  5. Espaciado de requests entre instrumentos

Uso:
  from gdelt_client import GDELTClient
  client = GDELTClient()
  articles = client.fetch(query="FTSE 100 UK economy", timespan="1d")
"""

import json
import os
import random
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

_HERE     = Path(__file__).resolve().parent
_MVP_ROOT = _HERE.parent.parent
_CACHE_DIR = _MVP_ROOT / "data" / "gdelt_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

GDELT_V2_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Tiempos de espera (segundos)
RETRY_BASE      = 5      # base para backoff exponencial
RETRY_MAX       = 60     # maximo entre reintentos
INTER_REQ_SLEEP = 5      # pausa minima entre requests al mismo query
MAX_RETRIES     = 2      # solo 2 intentos; si falla, RSS cubre el gap
CACHE_TTL_SECS  = 43200  # 12 horas de cache — GDELT solo se refresca 2x/dia max


class GDELTClient:
    """
    Cliente GDELT con manejo robusto de rate limits.
    Usa gdeltdoc cuando esta disponible, fallback a requests directo.
    """

    def __init__(self):
        self._session    = None
        self._gdeltdoc   = None
        self._last_req   = 0.0
        self._init_backends()

    def _init_backends(self):
        """Inicializa gdeltdoc y requests session."""
        # Backend 1: gdeltdoc (preferido)
        try:
            from gdeltdoc import GdeltDoc, Filters
            self._gdeltdoc = GdeltDoc()
            self._Filters  = Filters
            print("[GDELT] Backend: gdeltdoc OK")
        except ImportError:
            print("[GDELT] gdeltdoc no disponible, usando requests directo")

        # Backend 2: requests con headers correctos
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept":          "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection":      "keep-alive",
        })
        # urllib3 retry solo para 503/504, NO para 429 (lo manejamos manualmente)
        retry = Retry(
            total=2,
            backoff_factor=1,
            status_forcelist=[503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        from requests.adapters import HTTPAdapter
        session.mount("https://", HTTPAdapter(max_retries=retry))
        self._session = session

    def _cache_key(self, query: str, timespan: str, mode: str) -> str:
        raw = f"{query}|{timespan}|{mode}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    def _cache_get(self, key: str) -> list | None:
        f = _CACHE_DIR / f"{key}.json"
        if not f.exists():
            return None
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            age  = time.time() - data.get("ts", 0)
            if age > CACHE_TTL_SECS:
                return None
            return data.get("articles", [])
        except Exception:
            return None

    def _cache_set(self, key: str, articles: list):
        f = _CACHE_DIR / f"{key}.json"
        f.write_text(
            json.dumps({"ts": time.time(), "articles": articles}, ensure_ascii=False),
            encoding="utf-8"
        )

    def _enforce_rate_limit(self):
        """Asegura un minimo de INTER_REQ_SLEEP segundos entre requests."""
        elapsed = time.time() - self._last_req
        if elapsed < INTER_REQ_SLEEP:
            time.sleep(INTER_REQ_SLEEP - elapsed + random.uniform(0.1, 0.5))
        self._last_req = time.time()

    def _backoff(self, attempt: int) -> float:
        """Calcula tiempo de espera con jitter exponencial."""
        wait = min(RETRY_BASE * (2 ** attempt), RETRY_MAX)
        jitter = random.uniform(0, wait * 0.3)
        return wait + jitter

    # ── Backend 1: gdeltdoc ──────────────────────────────────────────────────

    def _simplify_query(self, query: str) -> str:
        """Simplifica query si es demasiado compleja para gdeltdoc."""
        # gdeltdoc rechaza queries con demasiados OR o comillas anidadas
        # Limitamos a los primeros 3 terminos
        import re
        terms = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
        if len(terms) > 3:
            terms = terms[:3]
        # Eliminar comillas dobles (gdeltdoc las maneja internamente)
        simplified = " OR ".join(t.strip('"') for t in terms)
        return simplified

    def _fetch_gdeltdoc(self, query: str, timespan: str, max_records: int) -> list:
        """Usa la libreria gdeltdoc para el fetch."""
        if self._gdeltdoc is None:
            raise ImportError("gdeltdoc no disponible")

        # Si la query es muy compleja, simplificarla para gdeltdoc
        simple_q = self._simplify_query(query)
        filters = self._Filters(
            keyword   = simple_q,
            timespan  = timespan,
            num_records = min(max_records, 250),
        )
        df = self._gdeltdoc.article_search(filters)
        if df is None or df.empty:
            return []

        articles = []
        for _, row in df.iterrows():
            articles.append({
                "title":       str(row.get("title", "")),
                "url":         str(row.get("url", "")),
                "source":      str(row.get("domain", "")),
                "publishedAt": str(row.get("seendate", "")),
                "description": str(row.get("title", "")),
                "language":    str(row.get("language", "English")),
                "socialimage": str(row.get("socialimage", "")),
            })
        return articles

    # ── Backend 2: requests directo ──────────────────────────────────────────

    def _fetch_requests(self, query: str, timespan: str, max_records: int) -> list:
        """Fallback: requests directo a la API v2."""
        import urllib.parse
        params = {
            "query":      query,
            "mode":       "ArtList",
            "maxrecords": str(min(max_records, 250)),
            "format":     "json",
            "timespan":   timespan,
            "sort":       "DateDesc",
        }
        url  = GDELT_V2_URL + "?" + urllib.parse.urlencode(params)
        resp = self._session.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        arts = data.get("articles") or []
        return [
            {
                "title":       a.get("title", ""),
                "url":         a.get("url", ""),
                "source":      a.get("domain", ""),
                "publishedAt": a.get("seendate", ""),
                "description": a.get("title", ""),
            }
            for a in arts if a.get("title")
        ]

    # ── Metodo publico principal ─────────────────────────────────────────────

    def fetch(
        self,
        query:       str,
        timespan:    str  = "1d",
        max_records: int  = 50,
        use_cache:   bool = True,
    ) -> list:
        """
        Fetch noticias de GDELT con retry robusto y cache.

        Args:
            query:       Query string (keywords, operadores AND/OR)
            timespan:    "1d", "6h", "3d", etc.
            max_records: Max articulos a retornar
            use_cache:   Usar cache local de 2h

        Returns:
            Lista de dicts con title, url, source, publishedAt, description
        """
        query = query.strip()
        key   = self._cache_key(query, timespan, "ArtList")

        # Cache hit
        if use_cache:
            cached = self._cache_get(key)
            if cached is not None:
                print(f"  [GDELT] Cache hit ({len(cached)} arts) para: {query[:60]}")
                return cached

        self._enforce_rate_limit()

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                # Intentar requests directo primero (mas fiable, evita RateLimitError de gdeltdoc)
                # gdeltdoc como fallback solo si requests falla con error no-429
                articles = self._fetch_requests(query, timespan, max_records)

                if use_cache:
                    self._cache_set(key, articles)
                print(f"  [GDELT] OK: {len(articles)} arts (intento {attempt+1})")
                return articles

            except Exception as e:
                last_error  = e
                err_str     = str(e)
                is_429      = "429" in err_str or "Too Many" in err_str or "RateLimit" in type(e).__name__
                is_timeout  = "timeout" in err_str.lower() or "timed out" in err_str.lower()

                if attempt == MAX_RETRIES - 1:
                    break

                wait = self._backoff(attempt)
                if is_429:
                    wait = max(wait, 20)  # minimo 20s en 429
                    print(f"  [GDELT] 429 rate limit — esperando {wait:.1f}s (intento {attempt+1}/{MAX_RETRIES})")
                    # En 429, intentar gdeltdoc como alternativa
                    if self._gdeltdoc is not None:
                        try:
                            time.sleep(3)
                            articles = self._fetch_gdeltdoc(query, timespan, max_records)
                            if use_cache:
                                self._cache_set(key, articles)
                            print(f"  [GDELT] OK via gdeltdoc: {len(articles)} arts")
                            return articles
                        except Exception:
                            pass  # gdeltdoc tambien fallo, seguir con backoff
                elif is_timeout:
                    print(f"  [GDELT] Timeout — esperando {wait:.1f}s (intento {attempt+1}/{MAX_RETRIES})")
                else:
                    print(f"  [GDELT] Error: {err_str[:80]} — esperando {wait:.1f}s")

                time.sleep(wait)

        print(f"  [GDELT] FAIL tras {MAX_RETRIES} intentos: {str(last_error)[:100]}")
        return []


# ── Instancia global compartida ──────────────────────────────────────────────
_client = None


def get_client() -> GDELTClient:
    global _client
    if _client is None:
        _client = GDELTClient()
    return _client


def fetch_news(query: str, timespan: str = "1d", max_records: int = 50) -> list:
    """Shortcut para usar el cliente global."""
    return get_client().fetch(query, timespan, max_records)


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = GDELTClient()
    print("\nTest 1: UK100 / FTSE")
    arts = client.fetch('FTSE "UK100" OR "Bank of England" OR "UK economy"', timespan="1d")
    for a in arts[:3]:
        print(f"  - {a['title'][:80]}")

    print("\nTest 2: Gold / XAUUSD")
    arts = client.fetch('"gold price" OR "XAU" OR "precious metals" inflation', timespan="1d")
    for a in arts[:3]:
        print(f"  - {a['title'][:80]}")

    print("\nTest 3: Bitcoin")
    arts = client.fetch("bitcoin BTC crypto regulation", timespan="1d")
    for a in arts[:3]:
        print(f"  - {a['title'][:80]}")
