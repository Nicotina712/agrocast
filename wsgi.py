"""
wsgi.py — Production entrypoint for AgroCast PRO.

Used by gunicorn (Render, Heroku, etc):
    gunicorn wsgi:app --workers 1 --threads 8 --timeout 120

Why a separate wsgi.py:
  - The Flask app lives in `MVP lectura de noticias/news_server.py`.
    The directory name has a space → can't be imported as a module from
    gunicorn directly. This file fixes sys.path and exposes `app`.
  - The pipeline bootstrap and APScheduler live inside
    `if __name__ == "__main__"` in news_server.py, so they never run
    under gunicorn. We replicate the bits that matter here, gated by
    env vars so local dev (running news_server.py directly) is unchanged.

Env vars consumed:
  RUN_SCHEDULER       "1" → start APScheduler (use ONLY with --workers 1)
  BOOTSTRAP_PIPELINE  "1" → run full pipeline on first boot if no artifacts
  CORS_ORIGINS        comma-separated list, default "*"
  DATA_DIR            override data path (default: ./data)
  ARTIFACTS_DIR       override artifacts path (default: ./artifacts)
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MVP lectura de noticias"))

# ── .env (optional) ───────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    env_file = ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
        print(f"[wsgi] .env loaded from {env_file}")
except ImportError:
    pass

# ── Stdout encoding (Windows / Render parity) ─────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Import Flask app ──────────────────────────────────────────────────
# news_server.py auto-imports multi_news_fetcher from the same dir,
# which is now on sys.path.
from importlib import import_module
news_server = import_module("news_server")
app = news_server.app
print("[wsgi] Flask app imported")

# ── CORS ──────────────────────────────────────────────────────────────
try:
    from flask_cors import CORS
    origins_raw = os.getenv("CORS_ORIGINS", "*").strip()
    origins = [o.strip() for o in origins_raw.split(",")] if origins_raw != "*" else "*"
    CORS(app, resources={r"/api/*": {"origins": origins}})
    print(f"[wsgi] CORS enabled: origins={origins}")
except ImportError:
    print("[wsgi] flask-cors not installed; CORS disabled")

# ── Healthcheck endpoint (Render uses this) ───────────────────────────
@app.route("/healthz")
def _healthz():
    return {"status": "ok", "service": "agrocast"}, 200

# ── Bootstrap pipeline on first deploy (no artifacts on disk) ─────────
if os.getenv("BOOTSTRAP_PIPELINE", "0") == "1":
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
    forecast_csv = artifacts_dir / "forecast.csv"
    if not forecast_csv.exists():
        print("[wsgi] no artifacts on disk → running pipeline (one-shot, blocking)")
        try:
            subprocess.run(
                [sys.executable, "-m", "src.pipeline"],
                cwd=str(ROOT), timeout=900, check=False,
            )
            print("[wsgi] bootstrap pipeline finished")
        except Exception as e:
            print(f"[wsgi] bootstrap pipeline failed: {e}")
    else:
        print(f"[wsgi] artifacts present at {forecast_csv} → skipping bootstrap")

# ── Scheduler (APScheduler) ───────────────────────────────────────────
# IMPORTANT: only enable with gunicorn --workers 1, otherwise the
# pipeline will run N times in parallel.
if os.getenv("RUN_SCHEDULER", "0") == "1":
    try:
        from src.infra.scheduler import start_scheduler
        start_scheduler()
        print("[wsgi] scheduler started")
    except Exception as e:
        print(f"[wsgi] scheduler not started: {e}")
else:
    print("[wsgi] RUN_SCHEDULER!=1 → scheduler disabled (use external cron or set RUN_SCHEDULER=1)")
