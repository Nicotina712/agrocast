"""
Motor de alertas de noticias para el portfolio de robots.
Lee los resultados de news_fetcher_multi, detecta noticias de alto impacto
y envía alertas por Telegram reutilizando la infraestructura de AgroCast.

Lógica de alerta:
  - magnitude >= threshold del instrumento
  - confidence >= threshold del instrumento
  - No repetir la misma noticia (dedup por hash de título)
  - Max 3 alertas por instrumento por día (evita spam)
  - Cooldown de 30min entre alertas del mismo instrumento

Uso:
  python alert_engine.py              # evalúa todos los instrumentos
  python alert_engine.py --instrument UK100
  python alert_engine.py --dry-run    # muestra alertas sin enviar
"""

import os, sys, json, hashlib, argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HERE     = Path(__file__).resolve().parent
_MVP_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_MVP_ROOT))

from dotenv import load_dotenv
load_dotenv(_MVP_ROOT / ".env", override=True)

from templates_nuevos_mercados.news_intelligence.instrument_profiles import INSTRUMENT_PROFILES

_DATA_DIR   = _MVP_ROOT / "data" / "news_portfolio"
_STATE_FILE = _DATA_DIR / "alert_state.json"

MAX_ALERTS_PER_DAY  = 3
COOLDOWN_MINUTES    = 30


# ── Telegram (reutiliza config de AgroCast) ─────────────────────────────────

def _send_telegram(text: str) -> bool:
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[Telegram] No configurado — saltando envío")
        return False
    import urllib.request, urllib.parse
    payload = json.dumps({
        "chat_id": chat_id, "text": text, "parse_mode": "HTML"
    }).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception as e:
        print(f"[Telegram] Error: {e}")
        return False


# ── Estado de alertas ────────────────────────────────────────────────────────

def _load_state() -> dict:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(state: dict):
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _article_hash(title: str) -> str:
    return hashlib.sha1(title.lower().strip()[:100].encode()).hexdigest()[:12]


def _should_alert(instrument: str, article_hash: str, state: dict) -> tuple[bool, str]:
    """Retorna (should_alert, reason_if_not)."""
    now     = datetime.now(timezone.utc)
    today   = now.strftime("%Y-%m-%d")
    inst_st = state.get(instrument, {})

    # 1. Dedup: ya se alertó esta noticia
    seen = inst_st.get("seen_hashes", [])
    if article_hash in seen:
        return False, "ya alertada"

    # 2. Max alertas por día
    alerts_today = inst_st.get("alerts_today", {})
    if alerts_today.get("date") == today and alerts_today.get("count", 0) >= MAX_ALERTS_PER_DAY:
        return False, f"limite diario ({MAX_ALERTS_PER_DAY}) alcanzado"

    # 3. Cooldown
    last_alert = inst_st.get("last_alert_utc")
    if last_alert:
        last_dt = datetime.fromisoformat(last_alert)
        if (now - last_dt).total_seconds() < COOLDOWN_MINUTES * 60:
            mins_left = int(COOLDOWN_MINUTES - (now - last_dt).total_seconds() / 60)
            return False, f"cooldown ({mins_left}min restantes)"

    return True, ""


def _update_state(state: dict, instrument: str, article_hash: str):
    now   = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    inst_st = state.setdefault(instrument, {})

    # Seen hashes (keep last 200)
    seen = inst_st.get("seen_hashes", [])
    seen.append(article_hash)
    inst_st["seen_hashes"] = seen[-200:]

    # Alerts today counter
    alerts_today = inst_st.get("alerts_today", {})
    if alerts_today.get("date") != today:
        alerts_today = {"date": today, "count": 0}
    alerts_today["count"] += 1
    inst_st["alerts_today"] = alerts_today

    # Last alert timestamp
    inst_st["last_alert_utc"] = now.isoformat()
    state[instrument] = inst_st


# ── Formato de alerta ─────────────────────────────────────────────────────────

IMPACT_EMOJI = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}
MAG_LABEL    = {1: "ruido", 2: "leve", 3: "relevante", 4: "importante", 5: "SHOCK"}

def _format_alert(instrument: str, profile: dict, article: dict, intel: dict) -> str:
    impact   = intel.get("price_impact", "neutral")
    mag      = intel.get("magnitude", 1)
    conf     = intel.get("confidence", 0)
    horizon  = intel.get("horizon", "?")
    drivers  = ", ".join(intel.get("drivers", ["other"]))
    rationale= intel.get("rationale", "")
    key_quote= intel.get("key_quote", "")
    title    = article.get("title", "")[:200]
    source   = article.get("source", "")
    url      = article.get("url", "")

    emoji = IMPACT_EMOJI.get(impact, "➡️")
    mag_label = MAG_LABEL.get(mag, str(mag))

    lines = [
        f"{emoji} <b>[{instrument}] Alerta de Noticias</b>",
        f"<b>{profile['name']}</b> — {impact.upper()} | Magnitud: {mag}/5 ({mag_label}) | Horizonte: {horizon}",
        f"",
        f"📰 <b>{title}</b>",
        f"Fuente: {source}",
        f"",
        f"🎯 <b>Drivers:</b> {drivers}",
        f"💡 <b>Impacto:</b> {rationale}",
    ]
    if key_quote:
        lines.append(f'📌 <i>"{key_quote}"</i>')
    lines += [
        f"",
        f"Confianza: {conf:.0%} | {datetime.now().strftime('%H:%M UTC')}",
    ]
    if url:
        lines.append(f"🔗 <a href='{url}'>Ver noticia</a>")

    return "\n".join(lines)


# ── Motor principal ───────────────────────────────────────────────────────────

def evaluate_instrument(instrument: str, state: dict, dry_run: bool = False) -> list:
    """Evalúa noticias de un instrumento y genera alertas si corresponde."""
    profile   = INSTRUMENT_PROFILES.get(instrument)
    latest_f  = _DATA_DIR / f"{instrument}_latest.json"

    if not profile or not latest_f.exists():
        return []

    try:
        data = json.loads(latest_f.read_text(encoding="utf-8"))
    except Exception:
        return []

    articles   = data.get("articles", [])
    mag_th     = profile["magnitude_threshold"]
    conf_th    = profile["confidence_threshold"]
    alerts_sent= []

    for art in articles:
        intel = art.get("intel", {})
        if not intel:
            continue

        mag  = intel.get("magnitude", 1)
        conf = intel.get("confidence", 0)
        impact = intel.get("price_impact", "neutral")

        # Filtro de calidad
        if impact == "neutral" and mag < 4:
            continue
        if mag < mag_th or conf < conf_th:
            continue

        a_hash = _article_hash(art.get("title", ""))
        should, reason = _should_alert(instrument, a_hash, state)

        if not should:
            print(f"  [{instrument}] Skipped '{art['title'][:60]}...' — {reason}")
            continue

        # Generar y enviar alerta
        msg = _format_alert(instrument, profile, art, intel)

        if dry_run:
            print(f"\n{'='*60}")
            print(f"[DRY-RUN] Alerta para {instrument}:")
            print(msg)
            print(f"{'='*60}")
            alerts_sent.append({"instrument": instrument, "title": art.get("title"), "dry_run": True})
        else:
            print(f"  [{instrument}] Enviando alerta: {art['title'][:60]}...")
            sent = _send_telegram(msg)
            if sent:
                _update_state(state, instrument, a_hash)
                alerts_sent.append({"instrument": instrument, "title": art.get("title"), "sent": True})
                print(f"  [{instrument}] OK Alerta enviada")
            else:
                print(f"  [{instrument}] FAIL Error al enviar")

    return alerts_sent


def run_all(instruments: list = None, dry_run: bool = False):
    instruments = instruments or list(INSTRUMENT_PROFILES.keys())
    state       = _load_state()
    all_alerts  = []

    print(f"\n=== Alert Engine — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    print(f"Evaluando {len(instruments)} instrumentos...")

    for inst in instruments:
        alerts = evaluate_instrument(inst, state, dry_run=dry_run)
        all_alerts.extend(alerts)

    if not dry_run and all_alerts:
        _save_state(state)

    print(f"\nResumen: {len(all_alerts)} alertas {'(dry-run)' if dry_run else 'enviadas'}")
    for a in all_alerts:
        print(f"  {a['instrument']}: {a.get('title', '')[:70]}")

    return all_alerts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", help="Solo un instrumento")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else None
    run_all(instruments=instruments, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

