"""
Pipeline completo: fetch + análisis + alertas para todos los instrumentos.
Diseñado para correr cada hora via Task Scheduler o cron.

Uso:
  python run_news_pipeline.py              # pipeline completo
  python run_news_pipeline.py --dry-run   # sin enviar alertas ni LLM
  python run_news_pipeline.py --instrument UK100  # solo un instrumento
  python run_news_pipeline.py --report    # muestra resumen de últimas alertas
"""

import sys, argparse, json, time
from pathlib import Path
from datetime import datetime

_HERE     = Path(__file__).resolve().parent
_MVP_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_MVP_ROOT))

from templates_nuevos_mercados.news_intelligence.news_fetcher_multi import (
    fetch_for_instrument, save_results, INSTRUMENT_PROFILES
)
from templates_nuevos_mercados.news_intelligence.alert_engine import run_all

_DATA_DIR   = _MVP_ROOT / "data" / "news_portfolio"
_REPORT_LOG = _DATA_DIR / "pipeline_log.jsonl"


def run_pipeline(instruments=None, dry_run=False):
    instruments = instruments or list(INSTRUMENT_PROFILES.keys())
    started_at  = datetime.now()
    print(f"\n{'='*65}")
    print(f"  Portfolio News Pipeline — {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Instrumentos: {len(instruments)} | dry-run: {dry_run}")
    print(f"{'='*65}")

    # PASO 1: Fetch y análisis de sentimiento
    # Pausa de 45s entre instrumentos para respetar rate limits de GDELT
    GDELT_INTER_INSTRUMENT_SLEEP = 45
    fetch_results = {}
    for i, inst in enumerate(instruments):
        if i > 0:
            print(f"  [GDELT] Pausa {GDELT_INTER_INSTRUMENT_SLEEP}s entre instrumentos...")
            time.sleep(GDELT_INTER_INSTRUMENT_SLEEP)
        articles = fetch_for_instrument(inst, dry_run=dry_run)
        if articles:
            save_results(inst, articles)
            fetch_results[inst] = len(articles)

    # PASO 2: Evaluación y alertas
    alerts = run_all(instruments=instruments, dry_run=dry_run)

    # PASO 3: Log del pipeline
    duration = (datetime.now() - started_at).total_seconds()
    log_entry = {
        "timestamp":    started_at.isoformat(),
        "instruments":  instruments,
        "fetch_counts": fetch_results,
        "alerts_count": len(alerts),
        "alerts":       [{"inst": a["instrument"], "title": a.get("title","")[:80]} for a in alerts],
        "duration_s":   round(duration, 1),
        "dry_run":      dry_run,
    }
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_REPORT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    print(f"\n{'='*65}")
    print(f"  Pipeline completado en {duration:.1f}s")
    print(f"  Noticias fetched: {sum(fetch_results.values())}")
    print(f"  Alertas enviadas: {len(alerts)}")
    print(f"{'='*65}\n")
    return log_entry


def show_report(last_n=20):
    """Muestra resumen del historial de alertas."""
    if not _REPORT_LOG.exists():
        print("Sin historial de pipeline todavía.")
        return
    lines = _REPORT_LOG.read_text(encoding="utf-8").strip().split("\n")
    entries = [json.loads(l) for l in lines[-last_n:] if l.strip()]
    print(f"\n=== Últimas {len(entries)} corridas del pipeline ===")
    for e in entries:
        ts      = e["timestamp"][:16]
        n_fetch = sum(e.get("fetch_counts", {}).values())
        n_alert = e.get("alerts_count", 0)
        dur     = e.get("duration_s", "?")
        dr      = "[dry]" if e.get("dry_run") else ""
        print(f"  {ts} | fetch={n_fetch} | alertas={n_alert} | {dur}s {dr}")
        for a in e.get("alerts", []):
            print(f"    → [{a['inst']}] {a['title'][:70]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", help="Solo un instrumento")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--report",   action="store_true", help="Muestra historial")
    args = parser.parse_args()

    if args.report:
        show_report()
        return

    instruments = [args.instrument] if args.instrument else None
    run_pipeline(instruments=instruments, dry_run=args.dry_run)

    # 2026-06-24 HILO 1: snapshot del net news-intel vs senal viva (forward-validation del veto)
    if not args.dry_run and not args.instrument:
        try:
            import subprocess, os as _o
            _root = _o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
            subprocess.run(["py", "-3", "news_context_shadow.py"], cwd=_root, timeout=120)
            print("[shadow] news_context snapshot OK")
        except Exception as _e:
            print(f"[shadow] no critico: {_e}")


if __name__ == "__main__":
    main()
