"""
src/intraday/live/retrainer.py
Retraining periódico (Fase 2 — STUB).

Plan:
  - Cron semanal (Domingo 17:00 CT antes de Globex open)
  - Re-fetch barras 60 días
  - Train walk-forward
  - Si AUC fold-mean nuevo > AUC actual − 0.02 → swap modelo
  - Sino → mantener modelo viejo y alertar
"""

from __future__ import annotations


def weekly_retrain():
    raise NotImplementedError("Fase 2")
