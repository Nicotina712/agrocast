"""
src/intraday/live/retrainer.py
Retraining periódico del modelo intraday.

Plan:
  - Cron semanal (Domingo 17:00 CT antes de Globex open)
  - Re-fetch barras 60 días
  - Train walk-forward
  - Si AUC fold-mean nuevo > AUC actual − 0.02 → swap modelo
  - Sino → mantener modelo viejo y alertar

Estado: Fase 1 — retraining manual con logging.
       Fase 2 — automatización completa con swap condicional.
"""

from __future__ import annotations

import os
import time
import json
from datetime import datetime


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_ARTIFACTS = os.path.join(_ROOT, "artifacts", "intraday")
_META_PATH = os.path.join(_ARTIFACTS, "retrain_meta.json")


def weekly_retrain(force: bool = False) -> dict:
    """Retrain the intraday model with latest data.

    Fase 1: logs retrain metadata (timestamps, data range, metrics).
    Returns dict with retrain results or skip reason.
    """
    os.makedirs(_ARTIFACTS, exist_ok=True)

    # Check if retrain is needed (skip if <7 days since last)
    if not force and os.path.exists(_META_PATH):
        try:
            with open(_META_PATH) as f:
                meta = json.load(f)
            last = meta.get("last_retrain", "")
            if last:
                age_days = (datetime.now() - datetime.fromisoformat(last)).total_seconds() / 86400
                if age_days < 7:
                    return {"skipped": True, "reason": f"Last retrain {age_days:.1f}d ago (< 7d)"}
        except Exception:
            pass

    t0 = time.time()

    try:
        from src.intraday.data.tick_feed import fetch_intraday_bars
        from src.intraday.features.microstructure import build_intraday_features
        from src.intraday.model.train_intraday import train_intraday_model

        # Fetch fresh 60-day bars
        bars = fetch_intraday_bars(interval="60m", use_cache=False)
        if bars.empty:
            return {"skipped": True, "reason": "No bars available"}

        feat = build_intraday_features(bars, interval="60m")
        result = train_intraday_model(feat)

        elapsed = time.time() - t0

        meta = {
            "last_retrain": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "n_bars": len(bars),
            "n_features": len(feat.columns),
            "metrics": result if isinstance(result, dict) else {},
        }

        with open(_META_PATH, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"[Retrainer] Retrained in {elapsed:.1f}s with {len(bars)} bars")
        return {"skipped": False, **meta}

    except ImportError as e:
        return {"skipped": True, "reason": f"Missing dependency: {e}"}
    except Exception as e:
        return {"skipped": True, "reason": f"Retrain failed: {e}"}
