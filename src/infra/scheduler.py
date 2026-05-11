"""
src/infra/scheduler.py
Cron scheduler para AgroCast PRO.

Corre en un thread de background cuando el servidor Flask arranca.
No requiere servicios externos — usa APScheduler (pip install apscheduler).

Jobs configurados:
  - Pipeline completo: cada 6 horas (o configurable via PIPELINE_INTERVAL_H)
  - Brief semanal: lunes a las 8:00 AM (hora local)
  - Cleanup de caché expirada: cada 24 horas

Si APScheduler no está instalado, el scheduler no arranca (falla silenciosa).
El pipeline sigue corriendo en modo reactivo (al visitar el dashboard).
"""

import logging
import os
import subprocess
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PIPELINE_SCRIPT = os.path.join(_PROJECT_ROOT, "src", "pipeline.py")

logger = logging.getLogger("agrocast.scheduler")

# Configuración via env vars
PIPELINE_INTERVAL_H  = int(os.getenv("PIPELINE_INTERVAL_H", "6"))
BRIEF_HOUR           = int(os.getenv("BRIEF_HOUR", "8"))
ROLLING_WINDOW_YEARS = int(os.getenv("ROLLING_WINDOW_YEARS", "3"))


def _run_pipeline():
    """Corre el pipeline completo como subprocess."""
    logger.info(f"[Scheduler] Iniciando pipeline: {datetime.now().strftime('%H:%M')}")
    try:
        result = subprocess.run(
            [sys.executable, _PIPELINE_SCRIPT],
            capture_output=True, text=True, timeout=600,
            cwd=_PROJECT_ROOT,
            encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            logger.info("[Scheduler] Pipeline OK")
        else:
            logger.warning(f"[Scheduler] Pipeline error: {result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        logger.error("[Scheduler] Pipeline timeout (>10 min)")
    except Exception as e:
        logger.error(f"[Scheduler] Pipeline exception: {e}")


def _run_weekly_brief():
    """Genera y envía el brief semanal si es lunes."""
    logger.info("[Scheduler] Ejecutando brief semanal…")
    try:
        sys.path.insert(0, _PROJECT_ROOT)
        from src.alerts.weekly_brief import generate_weekly_brief
        result = generate_weekly_brief()
        if result:
            logger.info("[Scheduler] Brief enviado ✅")
        else:
            logger.info("[Scheduler] Brief omitido (no es lunes o ya enviado)")
    except Exception as e:
        logger.error(f"[Scheduler] Brief error: {e}")


def _check_wasde_alert():
    """Verifica si el próximo WASDE está en 48h y envía alerta Telegram."""
    logger.info("[Scheduler] Verificando proximidad WASDE...")
    try:
        sys.path.insert(0, _PROJECT_ROOT)
        from src.alerts.telegram_bot import check_and_alert_wasde_upcoming
        sent = check_and_alert_wasde_upcoming()
        if sent:
            logger.info("[Scheduler] Alerta WASDE enviada")
    except Exception as e:
        logger.error(f"[Scheduler] WASDE alert error: {e}")


def _run_monthly_rolling_retrain():
    """
    Re-entrena modelos con ventana deslizante de ROLLING_WINDOW_YEARS años.
    Corre el primer lunes de cada mes.
    Esto evita que el modelo se ancle a regímenes de precios muy antiguos
    (el drift monitor lo confirmó: AUC degrada con datos acumulativos).
    """
    logger.info(f"[Scheduler] Rolling retrain mensual (ventana {ROLLING_WINDOW_YEARS}a)…")
    try:
        result = subprocess.run(
            [sys.executable, _PIPELINE_SCRIPT],
            capture_output=True, text=True, timeout=600,
            cwd=_PROJECT_ROOT,
            encoding="utf-8", errors="replace",
            env={**os.environ, "ROLLING_WINDOW_YEARS": str(ROLLING_WINDOW_YEARS)},
        )
        if result.returncode == 0:
            logger.info("[Scheduler] Rolling retrain OK")
        else:
            logger.warning(f"[Scheduler] Rolling retrain error: {result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        logger.error("[Scheduler] Rolling retrain timeout (>10 min)")
    except Exception as e:
        logger.error(f"[Scheduler] Rolling retrain exception: {e}")


def _cleanup_cache():
    """Elimina archivos de caché muy antiguos (>7 días)."""
    import time
    data_dir = os.path.join(_PROJECT_ROOT, "data")
    if not os.path.exists(data_dir):
        return
    cutoff = time.time() - 7 * 86400
    cleaned = 0
    for fname in os.listdir(data_dir):
        if fname.endswith(".json") and "history" not in fname:
            fpath = os.path.join(data_dir, fname)
            try:
                if os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
                    cleaned += 1
            except Exception:
                pass
    if cleaned:
        logger.info(f"[Scheduler] Cleanup: {cleaned} archivos eliminados")


def start_scheduler():
    """
    Inicia el scheduler en background. Debe llamarse una sola vez al arrancar Flask.
    Falla silenciosamente si APScheduler no está instalado.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        logger.warning(
            "[Scheduler] APScheduler no instalado — scheduler desactivado. "
            "Instalar: pip install apscheduler"
        )
        return None

    scheduler = BackgroundScheduler(
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 3600},
        timezone="America/Montevideo",
    )

    # Pipeline cada N horas
    scheduler.add_job(
        _run_pipeline,
        trigger=IntervalTrigger(hours=PIPELINE_INTERVAL_H),
        id="pipeline",
        name=f"Pipeline AgroCast (cada {PIPELINE_INTERVAL_H}h)",
        replace_existing=True,
    )

    # Brief semanal — lunes 8:00 AM
    scheduler.add_job(
        _run_weekly_brief,
        trigger=CronTrigger(day_of_week="mon", hour=BRIEF_HOUR, minute=0),
        id="weekly_brief",
        name="Brief Semanal (lunes 8am)",
        replace_existing=True,
    )

    # Alerta WASDE — diaria 9:00 AM (48h antes del reporte)
    scheduler.add_job(
        _check_wasde_alert,
        trigger=CronTrigger(hour=9, minute=0),
        id="wasde_alert",
        name="Alerta WASDE 48h anticipada",
        replace_existing=True,
    )

    # Cleanup diario — 3:00 AM
    scheduler.add_job(
        _cleanup_cache,
        trigger=CronTrigger(hour=3, minute=0),
        id="cleanup",
        name="Cleanup caché",
        replace_existing=True,
    )

    # Rolling retrain mensual — primer lunes de cada mes a las 2:00 AM
    # Usa ventana deslizante ROLLING_WINDOW_YEARS (default 3 años) para
    # evitar que el modelo se ancle a regímenes de precios históricos.
    scheduler.add_job(
        _run_monthly_rolling_retrain,
        trigger=CronTrigger(day="1-7", day_of_week="mon", hour=2, minute=0),
        id="monthly_rolling_retrain",
        name=f"Rolling Retrain Mensual ({ROLLING_WINDOW_YEARS}a ventana)",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        f"[Scheduler] Iniciado ✅ | Pipeline cada {PIPELINE_INTERVAL_H}h | "
        f"Brief lunes {BRIEF_HOUR}:00 | TZ=America/Montevideo"
    )
    print(f"[Scheduler] ✅ Pipeline automático cada {PIPELINE_INTERVAL_H}h | Brief lunes {BRIEF_HOUR}:00h")
    return scheduler
